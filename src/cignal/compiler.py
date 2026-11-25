"""Compile a `Kernel` subclass, and the kernels it holds as fields, to a C99 header and source.

The module is imported in CPython to read build-time values (field defaults,
parameters, module constants). Method bodies are re-parsed and lowered to C in
one typed walk, which also reports every unsupported construct.
"""

import ast
import builtins
import importlib.util
import inspect
import itertools
import math
import operator
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import NoReturn

from .diagnostics import CompileError, Diagnostic, DiagnosticSeverity, RuleId, SourceSpan
from .kernel import SMOOTH_SNAP, Array, Delay, Kernel, Param, Spec, Table, event

Number = bool | int | float
Ty = str | tuple[str, ...]  # a scalar type name, or a tuple of scalar type names

FLOAT_TYPES = ("f32", "f64")
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
MAX_BUFFER = 2**24  # largest Array/Delay length: indices and `f32(N - 1)` stay exact in float
FLT_MAX = 3.4028234663852886e38
NONE = "none"  # the type of a call to a method that returns None

_MATH_FUNCTIONS = {
    "sin": 1, "cos": 1, "tan": 1, "tanh": 1, "exp": 1, "log": 1,
    "sqrt": 1, "fabs": 1, "floor": 1, "ceil": 1, "atan2": 2, "pow": 2,
    "fmod": 2, "log2": 1, "log10": 1, "copysign": 2,
}  # fmt: skip
_BUILTINS = ("abs", "min", "max")
_MATH_CONSTANTS = {"pi": math.pi, "tau": math.tau, "e": math.e}
_CASTS = {"f32": float, "f64": float, "i32": int}
_LIFECYCLE = ("init", "control", "tick")
_RESERVED_METHODS = ("process", "set_param", "reset", "target", "event")
# Pd's built-in selectors and the Pd shim's own methods: no event may take these names, and no parameter
# with one gets a `<name> value` message.
PD_SELECTORS = frozenset({"anything", "bang", "dsp", "float", "list", "pointer", "reset", "signal", "symbol"})


def _py_mod(a: Number, b: Number) -> Number:
    return a % b


_BINOPS: dict[type[ast.operator], tuple[str, Callable[[Number, Number], Number]]] = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
    ast.Mod: ("%", _py_mod),
    ast.FloorDiv: ("//", operator.floordiv),
    ast.Pow: ("**", operator.pow),
    ast.BitAnd: ("&", operator.and_),
    ast.BitOr: ("|", operator.or_),
    ast.BitXor: ("^", operator.xor),
    ast.LShift: ("<<", operator.lshift),
    ast.RShift: (">>", operator.rshift),
}
_BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
# C precedence of `&`, `^`, `|`; every non-atom operand of these is parenthesised (GCC's -Wparentheses).
_BIT_PREC = {ast.BitAnd: 8, ast.BitXor: 7, ast.BitOr: 6}
_COMPARES: dict[type[ast.cmpop], tuple[str, Callable[[Number, Number], bool]]] = {
    ast.Lt: ("<", operator.lt),
    ast.LtE: ("<=", operator.le),
    ast.Gt: (">", operator.gt),
    ast.GtE: (">=", operator.ge),
    ast.Eq: ("==", operator.eq),
    ast.NotEq: ("!=", operator.ne),
}

# C99 keywords and the libm names generated code calls. User code never shares a scope with `K_process`,
# so its locals (`in`, `out`, `n`, `i`) need no escaping.
_RESERVED = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if inline int "
    "long register restrict return short signed sizeof static struct switch typedef union unsigned void "
    "volatile while _Bool _Complex _Imaginary bool true false".split()
) | frozenset(name + suffix for name in _MATH_FUNCTIONS for suffix in ("", "f"))

# Python's `%`: the result takes the sign of the divisor. The float version follows CPython's float_rem.
_HELPERS = {
    "mod_float": """static inline float cignal_mod_float(float x, float y) {
    float m = fmodf(x, y);
    if (m != 0.0f) {
        if ((y < 0.0f) != (m < 0.0f)) m += y;
    } else {
        m = copysignf(0.0f, y);
    }
    return m;
}""",
    "mod_double": """static inline double cignal_mod_double(double x, double y) {
    double m = fmod(x, y);
    if (m != 0.0) {
        if ((y < 0.0) != (m < 0.0)) m += y;
    } else {
        m = copysign(0.0, y);
    }
    return m;
}""",
    # CPython raises on a zero divisor; real-time code cannot, so it returns 0. INT32_MIN % -1 traps in C.
    # i32 arithmetic wraps, as the spec promises: signed overflow is undefined in C, unsigned is not.
    # Converting an out-of-range uint32_t back to int32_t is implementation-defined; GCC and clang wrap.
    "add_i32": """static inline int32_t cignal_add_i32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a + (uint32_t)b);
}""",
    "sub_i32": """static inline int32_t cignal_sub_i32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a - (uint32_t)b);
}""",
    "mul_i32": """static inline int32_t cignal_mul_i32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a * (uint32_t)b);
}""",
    "neg_i32": """static inline int32_t cignal_neg_i32(int32_t a) {
    return (int32_t)(0u - (uint32_t)a);
}""",
    # Python floors; C truncates. A zero divisor gives 0 (CPython raises); -1 wraps like negation.
    "floordiv_int32_t": """static inline int32_t cignal_floordiv_int32_t(int32_t a, int32_t b) {
    if (b == 0) return 0;
    if (b == -1) return (int32_t)(0u - (uint32_t)a);
    int32_t q = a / b;
    if (a % b != 0 && ((a < 0) != (b < 0))) q -= 1;
    return q;
}""",
    # CPython's float_floor_div, step for step.
    "floordiv_float": """static inline float cignal_floordiv_float(float x, float y) {
    float m = fmodf(x, y);
    float d = (x - m) / y;
    if (m != 0.0f) {
        if ((y < 0.0f) != (m < 0.0f)) d -= 1.0f;
    }
    if (d != 0.0f) {
        float f = floorf(d);
        if (d - f > 0.5f) f += 1.0f;
        return f;
    }
    return copysignf(0.0f, x / y);
}""",
    "floordiv_double": """static inline double cignal_floordiv_double(double x, double y) {
    double m = fmod(x, y);
    double d = (x - m) / y;
    if (m != 0.0) {
        if ((y < 0.0) != (m < 0.0)) d -= 1.0;
    }
    if (d != 0.0) {
        double f = floor(d);
        if (d - f > 0.5) f += 1.0;
        return f;
    }
    return copysign(0.0, x / y);
}""",
    # Counts outside 0..31 make Python's result leave i32 (or raise); these return the limiting value.
    "shl_int32_t": """static inline int32_t cignal_shl_int32_t(int32_t a, int32_t b) {
    if (b < 0 || b > 31) return 0;
    return (int32_t)((uint32_t)a << b);
}""",
    # Python's >> floors. ~a is non-negative when a is negative, so no shift of a negative value occurs.
    "shr_int32_t": """static inline int32_t cignal_shr_int32_t(int32_t a, int32_t b) {
    if (b < 0) b = 0;
    if (b > 31) b = 31;
    return a < 0 ? ~(~a >> b) : a >> b;
}""",
    # Exponentiation by squaring with wrapping multiplies; the exponent is a non-negative constant.
    "pow_int32_t": """static inline int32_t cignal_pow_int32_t(int32_t a, int32_t e) {
    uint32_t result = 1u, base = (uint32_t)a;
    while (e > 0) {
        if (e & 1) result *= base;
        base *= base;
        e >>= 1;
    }
    return (int32_t)result;
}""",
    # A constant divisor m > 0 needs neither the 0 nor the -1 guard; C's remainder takes the dividend's sign.
    "mod_int32_t_positive": """static inline int32_t cignal_mod_int32_t_positive(int32_t a, int32_t m) {
    int32_t r = a % m;
    return r < 0 ? r + m : r;
}""",
    "mod_int32_t": """static inline int32_t cignal_mod_int32_t(int32_t a, int32_t b) {
    if (b == 0 || b == -1) return 0;
    int32_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}""",
}


def _delay_helpers(t: str) -> dict[str, str]:
    """Ring-buffer operations for element type `t`.

    `kernel.Delay` defines the semantics; these follow it operation for operation. Each is a separate
    helper because clang's -Wunused-function flags an unused static function in a .c file.
    """
    return {
        f"delay_write_{t}": f"""static inline void cignal_delay_write_{t}({t} *buf, int32_t n, int32_t *pos, {t} x) {{
    buf[*pos] = x;
    *pos += 1;
    if (*pos == n) *pos = 0;
}}""",
        f"delay_read_{t}": f"""static inline {t} cignal_delay_read_{t}(const {t} *buf, int32_t n, int32_t pos,
        int32_t d) {{
    if (d < 1) d = 1;
    else if (d > n) d = n;
    int32_t index = pos - d;
    if (index < 0) index += n;
    return buf[index];
}}""",
        # The position is double for any element type: at 1600 samples an f32 position resolves only 1.2e-4.
        f"delay_read_frac_{t}": f"""static inline {t} cignal_delay_read_frac_{t}(const {t} *buf, int32_t n, int32_t pos,
        double d) {{
    double hi = (double)(n - 1);
    if (!(d >= 1.0)) d = 1.0;
    else if (d > hi) d = hi;
    int32_t i = (int32_t)d;
    {t} frac = ({t})(d - (double)i);
    {t} a = cignal_delay_read_{t}(buf, n, pos, i);
    {t} b = cignal_delay_read_{t}(buf, n, pos, i + 1);
    return a + frac * (b - a);
}}""",
        f"delay_clear_{t}": f"""static inline void cignal_delay_clear_{t}({t} *buf, int32_t n, int32_t *pos) {{
    for (int32_t k = 0; k < n; k++) buf[k] = 0;
    *pos = 0;
}}""",
    }


for _t in ("float", "double"):
    _HELPERS.update(_delay_helpers(_t))
# CPython's min and max keep the running result unless a later argument compares strictly less (greater).
# That fixes the NaN and signed-zero cases, where fmin and fmax differ: min(nan, 1.0) is nan, min(0.0, -0.0) is 0.0.
for _t in ("float", "double", "int32_t"):
    for _name, _op in (("min", "<"), ("max", ">")):
        _HELPERS[f"{_name}_{_t}"] = f"""static inline {_t} cignal_{_name}_{_t}({_t} a, {_t} b) {{
    return b {_op} a ? b : a;
}}"""
# C's abs(INT32_MIN) is undefined; this wraps, as the spec promises for i32.
_HELPERS["abs_int32_t"] = """static inline int32_t cignal_abs_int32_t(int32_t a) {
    return a < 0 ? (int32_t)(0u - (uint32_t)a) : a;
}"""
# GCC -O2 does not unroll a loop over sub-kernels, which serialises their dependency chains: 3.9x slower on
# an 8-biquad bank (docs/dev/simd.md). clang unrolls these already, and forcing it slowed the ladders by 12%.
# `i32()` of a float and event arguments, as kernel.event_i32: C's float-to-int conversion is undefined for
# NaN and out-of-range values (C99 6.3.1.4).
_HELPERS["to_i32"] = """static inline int32_t cignal_to_i32(double v) {
    if (isnan(v)) return 0;
    if (v <= -2147483648.0) return INT32_MIN;
    if (v >= 2147483647.0) return INT32_MAX;
    return (int32_t)v;
}"""
_HELPERS["unroll"] = """#ifndef CIGNAL_UNROLL
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8
#define CIGNAL_PRAGMA(x) _Pragma(#x)
#define CIGNAL_UNROLL(n) CIGNAL_PRAGMA(GCC unroll n)
#else
#define CIGNAL_UNROLL(n)
#endif
#endif"""
MAX_UNROLL = 16  # trip counts up to this are fully unrolled under GCC
# Helpers a helper calls, emitted before it.
_HELPER_DEPS = {"delay_read_frac_float": ["delay_read_float"], "delay_read_frac_double": ["delay_read_double"]}
_HELPER_SYMBOLS = {
    "mod_float": {"fmodf", "copysignf"},
    "mod_double": {"fmod", "copysign"},
    "floordiv_float": {"fmodf", "floorf", "copysignf"},
    "floordiv_double": {"fmod", "floor", "copysign"},
}
_DELAY_METHODS = {"write": 1, "read": 1, "read_frac": 1, "clear": 0}


def c_name(name: str) -> str:
    """Escape a Python identifier that is reserved in C or in the generated code.

    Appending `_` to every reserved name, and to every name already ending in `_`, keeps the mapping
    one-to-one: `int` becomes `int_`, so a user's `int_` must become `int__`. Generated identifiers
    start with `cignal_` and never end in `_`, so user names with that prefix are escaped too.
    """
    escape = name in _RESERVED or name.startswith(("_", "cignal_")) or name.endswith("_")
    return f"{name}_" if escape else name


@dataclass(frozen=True)
class Target:
    """How cignal types map to C. `f32_ctype="double"` builds the reference variant."""

    f32_ctype: str = "float"

    def ctype(self, t: Ty) -> str:
        if isinstance(t, tuple):
            return "cignal_tuple_" + "_".join(self.ctype(e).removesuffix("_t") for e in t)
        if t == NONE:
            return "void"
        return {"f32": self.f32_ctype, "f64": "double", "i32": "int32_t", "bool": "bool"}[t]


@dataclass
class Field:
    """A kernel field: a scalar (`type`), a sub-kernel (`kernel`), an `array` or a `delay`."""

    name: str
    type: str | None
    default: Number = 0
    kernel: str | None = None
    param: tuple[float, float] | None = None  # (lo, hi) for a host parameter
    smooth: float = 0.0  # one-pole time constant in seconds, for a smoothed parameter
    array: tuple[str, int] | None = None  # (element type, length)
    delay: tuple[str, int] | None = None  # (element type, length)
    count: int = 0  # > 0: an `Array` of `count` sub-kernels or delay lines

    @property
    def c_suffix(self) -> str:
        return f"[{self.count}]" if self.count else ""

    @property
    def is_scalar(self) -> bool:
        return self.type is not None


@dataclass
class Signature:
    params: dict[str, str]
    returns: Ty | None
    node: ast.FunctionDef
    failed: bool = False
    event: bool = False  # `@event def on_<name>`: on the exported kernel, a function hosts call


@dataclass
class CKernel:
    """Generated C for one exported kernel."""

    name: str
    header_name: str
    header: str
    source: str
    n_inputs: int
    n_outputs: int
    params: list[tuple[str, float, float, float]]  # (name, lo, hi, default)
    sample_ctype: str
    math_symbols: frozenset[str]
    c_name: str = ""  # the C identifier of the kernel: `--prefix` plus its name
    events: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)  # (host name, argument types)
    warnings: list[Diagnostic] = field(default_factory=list)


@dataclass
class _Expr:
    """A lowered expression: C code and type, or an untyped build-time constant."""

    code: str
    type: Ty | None  # None: untyped constant, see `value`
    value: Number | None = None
    prec: int = 16  # C precedence of the outermost operator; 16 for atoms, calls and member access

    @property
    def is_const(self) -> bool:
        return self.type is None


# C precedence levels used by the emitter.
_PREC_UNARY, _PREC_MUL, _PREC_ADD, _PREC_REL, _PREC_EQ, _PREC_AND, _PREC_OR, _PREC_COND = 14, 13, 12, 10, 9, 5, 4, 3
_PREC_AND_BITS = 8  # `&`


def _wrap(expr: _Expr, code: str, parent: int, right: bool = False) -> str:
    """Parenthesise `code` (the C form of `expr`) only where C's grammar would parse it differently.

    A right operand keeps parentheses at equal precedence: float `a - (b - c)` and `a + (b + c)`
    must not be reassociated.
    """
    prec = expr.prec
    if expr.is_const:
        # A negative literal such as `-2.0f` parses as unary minus.
        prec = _PREC_UNARY if isinstance(expr.value, (int, float)) and expr.value < 0 else 16
    return f"({code})" if prec < parent or (right and prec == parent) else code


class _Errors:
    def __init__(self) -> None:
        self.items: list[Diagnostic] = []

    def add(
        self,
        rule: str,
        message: str,
        node: ast.AST | None = None,
        remediation: str | None = None,
        severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
    ) -> None:
        span = SourceSpan.from_node(node) if node is not None else None
        node_type = type(node).__name__ if node is not None else None
        self.items.append(Diagnostic(rule, severity, message, span, node_type, None, remediation))

    @property
    def failed(self) -> bool:
        return any(d.is_error for d in self.items)


class _Abort(Exception):
    """Stops lowering of the current statement after a diagnostic is recorded."""


def _literal(value: Number, t: str, target: Target, errors: _Errors, node: ast.AST) -> str:
    """Render a constant as a C literal of type `t`."""
    if t == "bool":
        if not isinstance(value, bool):
            errors.add(RuleId.TYPE_MISMATCH, f"expected bool, got {type(value).__name__} constant", node)
            raise _Abort
        return "true" if value else "false"
    if isinstance(value, bool):
        errors.add(RuleId.TYPE_MISMATCH, f"expected {t}, got bool constant", node)
        raise _Abort
    if t == "i32":
        if isinstance(value, float):
            errors.add(RuleId.MIXED_NUMERIC, "float constant used as i32", node, "cast with i32(...)")
            raise _Abort
        if not INT32_MIN <= value <= INT32_MAX:
            errors.add(RuleId.CONSTANT_RANGE, f"{value} does not fit in i32", node)
            raise _Abort
        return "(-2147483647 - 1)" if value == INT32_MIN else str(value)
    fvalue = float(value)
    ctype = target.ctype(t)
    if not math.isfinite(fvalue) or (ctype == "float" and abs(fvalue) > FLT_MAX):
        errors.add(RuleId.CONSTANT_RANGE, f"{value!r} is not a finite {t}", node)
        raise _Abort
    return repr(fvalue) + ("f" if ctype == "float" else "")


def _type_text(t: Ty | None) -> str:
    if isinstance(t, tuple):
        return "tuple[" + ", ".join(t) + "]"
    return t or "constant"


class _FunctionLowerer:
    """Lower one method body to C statements."""

    def __init__(self, kernel: "_KernelModel", method: str, signature: Signature, free: bool = False):
        """`free` lowers a module-level function: no `self`, and `kernel` only supplies the module scope."""
        self.kernel = kernel
        self.program = kernel.program
        self.target = kernel.program.target
        self.errors = kernel.errors
        self.method = method
        self.has_self = not free
        self.key = ("", method) if free else (kernel.name, method)  # this function in the call graph
        self.loop_depth = 0
        self.params = signature.params
        self.returns = signature.returns
        self.locals: dict[str, str] = {}
        self.inline: set[str] = set()  # locals declared at their first assignment, not hoisted
        self.used: set[str] = set()
        self.uses_self = False
        self.lines: list[str] = []
        self.temps = itertools.count()
        # Live `for` variables and their proven value range (None when the bounds are not constant).
        self.loop_vars: dict[str, tuple[int, int] | None] = {}
        self.local_ranges: dict[str, tuple[int, int]] = {}  # locals every assignment keeps in a proven range
        self.assigned: set[str] = set()  # names the body assigns: never module constants, even before declared
        self.helpers: set[str] = set()  # emitted only if this function is
        self.smoothed_reads: list[ast.AST] = []  # reads of smoothed parameters, which `control` may not do
        self.depth = 1  # indentation of the statement being lowered, where `_leaf` emits its temporaries
        self.sequenced = False  # the statement's evaluation order is observable: see `_order_sensitive`

    # -- statements -------------------------------------------------------

    def lower_body(self, body: list[ast.stmt]) -> list[str]:
        self.local_ranges = self._infer_local_ranges(body)
        self._block(body, 1)
        # A local first assigned inside a branch is hoisted, so every later read sees a declaration.
        decls = [
            f"    {self.target.ctype(t)} {c_name(n)} = {self._zero(t)};"
            for n, t in self.locals.items()
            if n not in self.inline
        ]
        unused = [] if self.uses_self else ["    (void)self;"]
        unused += [f"    (void){c_name(p)};" for p in self.params if p not in self.used]
        return unused + decls + self.lines

    def _zero(self, t: str) -> str:
        return _literal(False if t == "bool" else 0, t, self.target, self.errors, ast.Pass())

    def _block(self, body: list[ast.stmt], depth: int) -> None:
        for index, stmt in enumerate(body):
            if index == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    continue  # docstring
            try:
                self._stmt(stmt, depth)
            except _Abort:
                pass

    def _emit(self, depth: int, text: str) -> None:
        self.lines.append("    " * depth + text)

    def _stmt(self, stmt: ast.stmt, depth: int) -> None:
        if isinstance(stmt, ast.Pass):
            return
        self.depth, self.sequenced = depth, self._order_sensitive(_statement_exprs(stmt))
        if isinstance(stmt, ast.AnnAssign):
            self._ann_assign(stmt, depth)
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                self._unsupported(stmt, "chained assignment")
            if isinstance(stmt.targets[0], ast.Tuple):
                self._unpack(stmt.targets[0], stmt.value, depth)
            else:
                self._assign(stmt.targets[0], stmt.value, depth)
        elif isinstance(stmt, ast.AugAssign):
            self._aug_assign(stmt, depth)
        elif isinstance(stmt, ast.Return):
            self._return(stmt, depth)
        elif isinstance(stmt, ast.If):
            self._if(stmt, depth, prefix="if")
        elif isinstance(stmt, ast.For):
            self._for(stmt, depth)
        elif isinstance(stmt, ast.While):
            self._while(stmt, depth)
        elif isinstance(stmt, (ast.Break, ast.Continue)):
            if not self.loop_depth:
                self._unsupported(stmt, f"`{type(stmt).__name__.lower()}` outside a loop")
            self._emit(depth, f"{type(stmt).__name__.lower()};")
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            expr = self._expr(stmt.value)
            self._emit(depth, f"{expr.code};" if expr.type == NONE else f"(void){expr.code};")
        else:
            self._unsupported(stmt, f"`{type(stmt).__name__}` statement")

    def _ann_assign(self, stmt: ast.AnnAssign, depth: int) -> None:
        if not isinstance(stmt.target, ast.Name):
            self._unsupported(stmt, "annotated assignment to a non-name")
        t = self.kernel.resolve_scalar(stmt.annotation)
        if t is None:
            raise _Abort
        name = stmt.target.id
        if name in self.params or (name in self.locals and self.locals[name] != t):
            self.errors.add(RuleId.TYPE_MISMATCH, f"`{name}` is already declared with another type", stmt)
            raise _Abort
        created = name not in self.locals
        self.locals.setdefault(name, t)
        if stmt.value is not None:
            value = self._coerce(self._expr(stmt.value), t, stmt.value)
            self._emit(depth, self._store(c_name(name), t, value, name if created else None, depth))

    def _assign(self, target: ast.expr, value: ast.expr, depth: int) -> None:
        expr = self._expr(value)
        if isinstance(expr.type, tuple) or expr.type == NONE:
            self.errors.add(RuleId.TYPE_MISMATCH, f"cannot assign a {_type_text(expr.type)} to one name", value,
                            "unpack it: `a, b = ...`" if isinstance(expr.type, tuple) else None)  # fmt: skip
            raise _Abort
        lhs, t, created = self._target(target, expr.type or self._const_type(expr))
        self._emit(depth, self._store(lhs, t, self._coerce(expr, t, value), created, depth))

    def _store(self, lhs: str, t: str, value: str, created: str | None, depth: int) -> str:
        """An assignment, or a declaration if it creates a local at the top level of the function."""
        if created is not None and depth == 1:
            self.inline.add(created)
            return f"{self.target.ctype(t)} {lhs} = {value};"
        return f"{lhs} = {value};"

    def _unpack(self, target: ast.Tuple, value: ast.expr, depth: int) -> None:
        expr = self._expr(value)
        if not isinstance(expr.type, tuple) or len(expr.type) != len(target.elts):
            self.errors.add(RuleId.TYPE_MISMATCH, f"cannot unpack {_type_text(expr.type)} into "
                            f"{len(target.elts)} names", value)  # fmt: skip
            raise _Abort
        temp = f"cignal_t{next(self.temps)}"
        lines = [f"{self.target.ctype(expr.type)} {temp} = {expr.code};"]
        for index, (elt, t) in enumerate(zip(target.elts, expr.type, strict=True)):
            lhs, lhs_type, created = self._target(elt, t)
            self._coerce(_Expr("", t), lhs_type, elt)
            lines.append(self._store(lhs, lhs_type, f"{temp}.v{index}", created, depth))
        # At the top level the temporary is an ordinary local; deeper, braces keep its name unique per scope.
        if depth == 1:
            for line in lines:
                self._emit(depth, line)
        else:
            self._emit(depth, "{ " + " ".join(lines) + " }")

    def _target(self, target: ast.expr, new_type: str | None) -> tuple[str, str, str | None]:
        """Resolve an assignment target to (C lvalue, type, name of a newly created local or None).

        `new_type` declares an unknown name; without it, an unknown name is an error.
        """
        if isinstance(target, ast.Name):
            name = target.id
            if name in self.loop_vars:
                self._unsupported(target, f"assigning the loop variable `{name}`")
            if name in self.params:
                self.used.add(name)
                return c_name(name), self.params[name], None
            if name not in self.locals:
                if new_type is None:
                    self.errors.add(RuleId.UNKNOWN_NAME, f"`{name}` is not defined", target)
                    raise _Abort
                self.locals[name] = new_type
                return c_name(name), new_type, name
            return c_name(name), self.locals[name], None
        if isinstance(target, ast.Attribute):
            code, field_ = self._field(target)
            if field_.param is not None and self.kernel.exported:
                self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"assigning parameter `{field_.name}`", target,
                                "parameters are set by the host; copy the value into another field")  # fmt: skip
                raise _Abort
            assert field_.type is not None
            return code, field_.type, None
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Attribute) and self._member(target.value)[1].param is not None and \
                    self.kernel.exported:  # fmt: skip
                self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"assigning parameter `{ast.unparse(target.value)}`",
                                target, "parameters are set by the host; copy the value into a field")  # fmt: skip
                raise _Abort
            code, t = self._subscript(target)
            return code, t, None
        self._unsupported(target, f"assignment to `{ast.unparse(target)}`")

    def _aug_assign(self, stmt: ast.AugAssign, depth: int) -> None:
        lhs, t, _ = self._target(stmt.target, None)
        current = self._leaf(_Expr(lhs, t))  # Python reads the target before it evaluates the value
        result = self._binop(current, stmt.op, self._expr(stmt.value), stmt)
        self._emit(depth, f"{lhs} = {self._coerce(result, t, stmt)};")

    def _return(self, stmt: ast.Return, depth: int) -> None:
        if self.returns is None:
            if stmt.value is not None:
                self.errors.add(RuleId.BAD_SIGNATURE, "this method returns None", stmt)
                raise _Abort
            self._emit(depth, "return;")
            return
        if stmt.value is None:
            self.errors.add(RuleId.BAD_SIGNATURE, f"must return {_type_text(self.returns)}", stmt)
            raise _Abort
        if isinstance(self.returns, tuple):
            self._emit(depth, f"return {self._tuple_value(stmt.value, self.returns)};")
        else:
            self._emit(depth, f"return {self._coerce(self._expr(stmt.value), self.returns, stmt.value)};")

    def _tuple_value(self, node: ast.expr, t: tuple[str, ...]) -> str:
        if isinstance(node, ast.Tuple):
            if len(node.elts) != len(t):
                self.errors.add(RuleId.BAD_SIGNATURE, f"expected {len(t)} values, got {len(node.elts)}", node)
                raise _Abort
            values = [self._coerce(self._expr(e), et, e) for e, et in zip(node.elts, t, strict=True)]
            self.program.tuples.add(t)
            return f"({self.target.ctype(t)}){{{', '.join(values)}}}"
        expr = self._expr(node)
        if expr.type != t:
            self.errors.add(RuleId.TYPE_MISMATCH, f"expected {_type_text(t)}, got {_type_text(expr.type)}", node)
            raise _Abort
        return expr.code

    def _if(self, stmt: ast.If, depth: int, prefix: str) -> None:
        self.depth, self.sequenced = depth, self._order_sensitive([stmt.test])
        test = self._expr(stmt.test)
        if test.type != "bool" and not (test.is_const and isinstance(test.value, bool)):
            self.errors.add(RuleId.TYPE_MISMATCH, "an `if` condition must be bool", stmt.test, "compare explicitly")
            raise _Abort
        cond = self._coerce(test, "bool", stmt.test)
        self._emit(depth, f"{prefix} ({cond}) {{")
        self._block(stmt.body, depth + 1)
        # An `elif` test with temporaries is nested in `else`, so they are evaluated only when it is.
        orelse = stmt.orelse[0] if len(stmt.orelse) == 1 else None
        if isinstance(orelse, ast.If) and not self._order_sensitive([orelse.test]):
            self._if(orelse, depth, prefix="} else if")
            return
        if stmt.orelse:
            self._emit(depth, "} else {")
            self._block(stmt.orelse, depth + 1)
        self._emit(depth, "}")

    def _for(self, stmt: ast.For, depth: int) -> None:
        """`for i in range(...)`: bounds evaluated once, as in Python; `i` is read-only in the body."""
        call = stmt.iter
        is_range = (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and getattr(self.kernel.module, "range", builtins.range) is builtins.range
        )
        if stmt.orelse or not isinstance(stmt.target, ast.Name) or not is_range:
            self._unsupported(stmt, "a `for` loop other than `for name in range(...)` without `else`")
        assert isinstance(call, ast.Call)
        if call.keywords or not 1 <= len(call.args) <= 3:
            self.errors.add(RuleId.UNSUPPORTED_CALL, "range() takes 1 to 3 positional arguments", call)
            raise _Abort
        name = stmt.target.id
        if name in self.params or name in self.locals or name in self.loop_vars:
            self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"loop variable `{name}` reuses a name in scope", stmt.target,
                            "use a new name for each loop variable")  # fmt: skip
            raise _Abort
        args = [self._expr(a) for a in call.args]
        for arg, node in zip(args, call.args, strict=True):
            self._coerce(arg, "i32", node)  # reports a non-i32 bound
        start, stop = (args[0], args[1]) if len(args) >= 2 else (_Expr("", None, 0), args[0])
        step = args[2].value if len(args) == 3 else 1
        if len(args) == 3 and not (args[2].is_const and isinstance(step, int) and step != 0):
            self.errors.add(RuleId.UNSUPPORTED_SYNTAX, "range() step must be a non-zero integer constant", call.args[2])
            raise _Abort
        assert isinstance(step, int)
        proven: tuple[int, int] | None = None
        trips = 0
        if start.is_const and stop.is_const:
            values = range(int(start.value or 0), int(stop.value or 0), step)
            trips = len(values)
            proven = (min(values), max(values)) if values else (0, 0)  # an empty loop body never runs
        var = c_name(name)
        first = self._coerce(start, "i32", call)
        bound = self._coerce(stop, "i32", call)
        if not stop.is_const:
            self._emit(depth, "{")
            if not start.is_const:  # Python evaluates the start first
                temp = f"cignal_t{next(self.temps)}"
                self._emit(depth + 1, f"const int32_t {temp} = {first};")
                first = temp
            temp = f"cignal_t{next(self.temps)}"
            self._emit(depth + 1, f"const int32_t {temp} = {bound};")
            bound, depth = temp, depth + 1
        cmp = "<" if step > 0 else ">"
        increment = f"{var}++" if step == 1 else f"{var} += {step}" if step > 0 else f"{var} -= {-step}"
        unroll = f"CIGNAL_UNROLL({trips}) " if 1 < trips <= MAX_UNROLL else ""
        if unroll:
            self.helpers.add("unroll")
        self._emit(depth, f"{unroll}for (int32_t {var} = {first}; {var} {cmp} {bound}; "
                   f"{increment}) {{")  # fmt: skip
        self.loop_vars[name] = proven
        self.loop_depth += 1
        try:
            self._block(stmt.body, depth + 1)
        finally:
            del self.loop_vars[name]
            self.loop_depth -= 1
        self._emit(depth, "}")
        if not stop.is_const:
            self._emit(depth - 1, "}")

    def _while(self, stmt: ast.While, depth: int) -> None:
        if stmt.orelse:
            self._unsupported(stmt, "`while ... else`")
        # A test with temporaries is evaluated at the top of each pass, as Python does.
        hoisted = self.sequenced
        if hoisted:
            self._emit(depth, "for (;;) {")
            self.depth = depth + 1
        test = self._expr(stmt.test)
        if test.type != "bool" and not (test.is_const and isinstance(test.value, bool)):
            self.errors.add(RuleId.TYPE_MISMATCH, "a `while` condition must be bool", stmt.test)
            raise _Abort
        self.errors.add(RuleId.UNBOUNDED_LOOP, "termination of this `while` loop is not checked; a loop that "
                        "runs too long misses the audio deadline", stmt,
                        "prefer `for i in range(N)` with `break` when a bound is known",
                        severity=DiagnosticSeverity.WARNING)  # fmt: skip
        cond = self._coerce(test, "bool", stmt.test)
        if hoisted:
            self._emit(depth + 1, f"if (!{_wrap(test, cond, _PREC_UNARY, right=True)}) break;")
        else:
            self._emit(depth, f"while ({cond}) {{")
        self.loop_depth += 1
        try:
            self._block(stmt.body, depth + 1)
        finally:
            self.loop_depth -= 1
        self._emit(depth, "}")

    # -- expressions ------------------------------------------------------

    def _expr(self, node: ast.expr) -> _Expr:
        if self.sequenced and _is_conditional(node) and _mentions_self(node):
            return self._conditional_leaf(node)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)):
                return _Expr("", None, node.value)
            self._unsupported(node, f"{type(node.value).__name__} constant")
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.BinOp):
            return self._binop(self._expr(node.left), node.op, self._expr(node.right), node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.BoolOp):
            return self._boolop(node)
        if isinstance(node, ast.IfExp):
            return self._ifexp(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and self._is_module_name(node.value.id):
                found = self.program.const_tuple(node.value.id)
                if found is not None:
                    return self._tuple_index(node.value.id, *found, node)
            code, t = self._subscript(node)
            return self._leaf(_Expr(code, t))
        if isinstance(node, ast.Tuple):
            self._unsupported(node, "a tuple outside `return` or unpacking")
        self._unsupported(node, f"`{type(node).__name__}` expression")

    def _name(self, node: ast.Name) -> _Expr:
        name = node.id
        if name in self.loop_vars:
            return _Expr(c_name(name), "i32")
        if name in self.params:
            self.used.add(name)
            return _Expr(c_name(name), self.params[name])
        if name in self.locals:
            return _Expr(c_name(name), self.locals[name])
        value = self.kernel.constant(name)
        if value is not None:
            return _Expr("", None, value)
        self.errors.add(RuleId.UNKNOWN_NAME, f"`{name}` is not a local, parameter or numeric module constant", node)
        raise _Abort

    def _attribute(self, node: ast.Attribute) -> _Expr:
        if self.kernel.is_module(node.value, math) and node.attr in _MATH_CONSTANTS:
            return _Expr("", None, _MATH_CONSTANTS[node.attr])
        code, field_ = self._field(node)
        if field_.smooth > 0 and self._receiver(node.value) == ("self->", "self", self.kernel) and self.kernel.exported:
            self.smoothed_reads.append(node)
        return self._leaf(_Expr(code, field_.type))

    def _receiver(self, node: ast.expr) -> tuple[str, str, "_KernelModel"] | None:
        """For `self` or a chain of sub-kernel fields: (field-access prefix, pointer, kernel model)."""
        if isinstance(node, ast.Name) and node.id == "self" and self.has_self:
            self.uses_self = True
            return "self->", "self", self.kernel
        if isinstance(node, ast.Attribute):
            base = self._receiver(node.value)
            if base is None:
                return None
            prefix, _, model = base
            field_ = model.fields.get(node.attr)
            if field_ is not None and field_.kernel is not None and not field_.count:
                member = f"{prefix}{c_name(node.attr)}"
                return member + ".", f"&{member}", self.program.model(field_.kernel)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            found = self._element(node)
            if found is not None and found[1].kernel is not None:
                member = found[0]
                return member + ".", f"&{member}", self.program.model(found[1].kernel)
        return None

    def _element(self, node: ast.Subscript) -> tuple[str, Field] | None:
        """`self.xs[i]` for an `Array` of sub-kernels or delay lines: (C lvalue of the element, field)."""
        assert isinstance(node.value, ast.Attribute)
        base = self._receiver(node.value.value)
        if base is None:
            return None
        prefix, _, model = base
        field_ = model.fields.get(node.value.attr)
        if field_ is None or not field_.count:
            return None
        index = self._proven_index(node.slice, self._expr(node.slice), field_.count)
        return f"{prefix}{c_name(node.value.attr)}[{index}]", field_

    def _member(self, node: ast.Attribute) -> tuple[str, Field]:
        """Resolve `self.x` or `self.sub.x` to (C lvalue, field) for a field of any kind."""
        base = self._receiver(node.value)
        if base is None:
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "self" and not self.has_self:
                self.errors.add(RuleId.UNKNOWN_NAME, f"`self` is not defined in module-level function "
                                f"`{self.method}`", node, "pass the values it needs as arguments")  # fmt: skip
                raise _Abort
            self._unsupported(node, f"attribute `{ast.unparse(node)}`")
        prefix, _, model = base
        field_ = model.fields.get(node.attr)
        if field_ is None:
            self.errors.add(RuleId.UNDECLARED_STATE, f"`{ast.unparse(node)}` is not a declared field", node,
                            f"declare it in the class body, e.g. `{node.attr}: f32 = 0.0`")  # fmt: skip
            raise _Abort
        return f"{prefix}{c_name(node.attr)}", field_

    def _field(self, node: ast.Attribute) -> tuple[str, Field]:
        """Resolve `self.x` or `self.sub.x` to (C lvalue, scalar field)."""
        code, field_ = self._member(node)
        if not field_.is_scalar:
            kind = "sub-kernel" if field_.kernel else "array" if field_.array else "delay line"
            hint = {
                "sub-kernel": "call one of its methods or read one of its fields",
                "array": "index it: `self.a[i]`",
                "delay line": "call write/read/read_frac/clear",
            }[kind]
            self.errors.add(RuleId.TYPE_MISMATCH, f"{kind} `{ast.unparse(node)}` used as a value", node, hint)
            raise _Abort
        return code, field_

    def _subscript(self, node: ast.Subscript) -> tuple[str, str]:
        """`self.a[e]` for an `Array` field, with `e` proven to be in range. Returns (C lvalue, element type)."""
        if not isinstance(node.value, ast.Attribute):
            self._unsupported(node, "subscript of anything but an `Array` field")
        code, field_ = self._member(node.value)
        if field_.array is None:
            self._unsupported(node, f"subscript of `{ast.unparse(node.value)}`, which is not an `Array`")
        element, size = field_.array
        if field_.smooth > 0 and self._receiver(node.value.value) == ("self->", "self", self.kernel) and \
                self.kernel.exported:  # fmt: skip
            self.smoothed_reads.append(node)
        return f"{code}[{self._proven_index(node.slice, self._expr(node.slice), size)}]", element

    def _proven_index(self, node: ast.expr, index: _Expr, size: int) -> str:
        """The C code for `index`, which must be provably in [0, size) (section 5.4)."""
        if any(isinstance(n, ast.Call) for n in ast.walk(node)):
            self.errors.add(RuleId.UNPROVEN_INDEX, "an index may not contain calls", node,
                            "compute the index into a local first")  # fmt: skip
            raise _Abort
        code = self._coerce(index, "i32", node)
        bounds = self._index_range(node)
        if bounds is None or bounds[0] < 0 or bounds[1] >= size:
            found = "unknown" if bounds is None else f"[{bounds[0]}, {bounds[1]}]"
            self.errors.add(RuleId.UNPROVEN_INDEX, f"index `{ast.unparse(node)}` is not provably in "
                            f"[0, {size - 1}] (found {found})", node,
                            f"use a constant, a `range` variable with constant bounds, `e % {size}`, or a local "
                            "assigned only such values")  # fmt: skip
            raise _Abort
        return code

    def _is_module_name(self, name: str) -> bool:
        return name not in self.loop_vars and name not in self.params and name not in self.locals

    def _tuple_index(self, name: str, t: str, values: tuple[Number, ...], node: ast.Subscript) -> _Expr:
        """`T[e]` for a module-level tuple `T`: folded for a constant `e`, else read from a `static const` array."""
        index = self._expr(node.slice)
        if index.is_const:
            i = index.value
            if isinstance(i, bool) or not isinstance(i, int) or not -len(values) <= i < len(values):
                self.errors.add(RuleId.CONSTANT_RANGE, f"`{name}[{i!r}]` is outside a {len(values)}-element tuple",
                                node.slice)  # fmt: skip
                raise _Abort
            return _Expr("", None, values[i])
        code = self._proven_index(node.slice, index, len(values))
        self.program.const_literals(name, t, values, node)
        self.helpers.add(f"const_{name}")
        return _Expr(f"cignal_const_{name}[{code}]", t)

    def _infer_local_ranges(self, body: list[ast.stmt]) -> dict[str, tuple[int, int]]:
        """Ranges of locals whose every assignment is provably in range, so `pick = i` can index after the loop.

        Flow-insensitive: a local's range is the union over all its assignments, plus 0, the value of a hoisted
        declaration. A local assigned any other way, or in a cycle (`k = k + 1`), gets no range.
        """
        self.assigned = {n.id for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Name)
                         and isinstance(n.ctx, ast.Store)}  # fmt: skip
        sites: dict[str, list[tuple[ast.expr | None, dict[str, tuple[int, int] | None]]]] = {}
        unknown: set[str] = set()

        def visit(stmts: list[ast.stmt], loops: dict[str, tuple[int, int] | None]) -> None:
            for stmt in stmts:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    sites.setdefault(stmt.targets[0].id, []).append((stmt.value, loops))
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    sites.setdefault(stmt.target.id, []).append((stmt.value, loops))
                elif isinstance(stmt, (ast.Assign, ast.AugAssign)):
                    unknown.update(n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)
                                   and isinstance(n.ctx, ast.Store))  # fmt: skip
                if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                    visit(stmt.body, {**loops, stmt.target.id: self._range_bounds(stmt.iter)})
                elif isinstance(stmt, (ast.If, ast.While)):
                    visit(stmt.body, loops)
                    visit(stmt.orelse, loops)

        visit(body, {})
        candidates = {n: s for n, s in sites.items() if n not in unknown and n not in self.params}
        ranges: dict[str, tuple[int, int]] = {}
        saved = self.loop_vars, self.local_ranges
        try:
            for _ in range(len(candidates) + 1):  # each pass settles at least one more local, or none ever will
                found: dict[str, tuple[int, int]] = {}
                for name, assigned in candidates.items():
                    lo, hi = 0, 0
                    for value, loops in assigned:
                        self.loop_vars, self.local_ranges = loops, ranges
                        r = (0, 0) if value is None else self._index_range(value, locals_known=True)
                        if r is None:
                            break
                        lo, hi = min(lo, r[0]), max(hi, r[1])
                    else:
                        found[name] = (lo, hi)
                if found == ranges:
                    break
                ranges = found
        finally:
            self.loop_vars, self.local_ranges = saved
        return ranges

    def _range_bounds(self, call: ast.expr) -> tuple[int, int] | None:
        """The values of `range(...)` with constant arguments, as `_for` proves them."""
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"):
            return None
        points = [self._index_range(a, locals_known=True) for a in call.args]
        if not 1 <= len(points) <= 3 or any(p is None or p[0] != p[1] for p in points):
            return None
        args = [p[0] for p in points if p is not None]
        if len(args) == 3 and args[2] == 0:
            return None
        values = range(*args)
        return (min(values), max(values)) if values else (0, 0)

    def _index_range(self, node: ast.expr, locals_known: bool = False) -> tuple[int, int] | None:
        """The interval an i32 index expression is proven to lie in, or None.

        `locals_known` applies `local_ranges` before the locals are declared, while they are inferred.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value, node.value
        if isinstance(node, ast.Name):
            if node.id in self.loop_vars:
                return self.loop_vars[node.id]
            if node.id in self.local_ranges and (locals_known or node.id in self.locals):
                return self.local_ranges[node.id]
            if node.id in self.params or node.id in self.locals or (locals_known and node.id in self.assigned):
                return None  # a runtime value, even if a module constant has the same name
            value = self.kernel.constant(node.id)
            if isinstance(value, int) and not isinstance(value, bool):
                return value, value
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._index_range(node.operand, locals_known)
            return (-inner[1], -inner[0]) if inner else None
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mod):
                right = self._index_range(node.right, locals_known)
                # Floor-mod by a positive constant lands in [0, M - 1] whatever the left operand.
                return (0, right[0] - 1) if right and right[0] == right[1] and right[0] > 0 else None
            left, right = self._index_range(node.left, locals_known), self._index_range(node.right, locals_known)
            if left is None or right is None:
                return None
            # Add and Sub are exact mod 2**32, so a range inside [0, size) holds under C's wrapping too.
            if isinstance(node.op, ast.Add):
                return left[0] + right[0], left[1] + right[1]
            if isinstance(node.op, ast.Sub):
                return left[0] - right[1], left[1] - right[0]
        return None

    def _binop(self, left: _Expr, op: ast.operator, right: _Expr, node: ast.AST) -> _Expr:
        if type(op) not in _BINOPS:
            self._unsupported(node, f"operator `{_op_text(op)}`")
        symbol, fn = _BINOPS[type(op)]
        for operand in (left, right):
            if isinstance(operand.type, tuple) or operand.type == NONE:
                self.errors.add(RuleId.TYPE_MISMATCH, f"`{symbol}` on {_type_text(operand.type)}", node)
                raise _Abort
        t = self._unify(left, right, node=node)
        if t == "bool" or any(isinstance(e.value, bool) for e in (left, right)):
            self.errors.add(RuleId.TYPE_MISMATCH, f"`{symbol}` on bool", node, "use `and`, `or`, `not` on bools")
            raise _Abort
        integral = t == "i32" or (t is None and _all_int(left, right))
        if isinstance(op, ast.Div) and integral:
            self.errors.add(RuleId.MIXED_NUMERIC, "`/` on i32 operands", node, "cast an operand with f32(...)")
            raise _Abort
        if isinstance(op, _BITWISE) and not integral:
            self.errors.add(RuleId.TYPE_MISMATCH, f"`{symbol}` needs i32 operands", node)
            raise _Abort
        if t is None:
            assert left.value is not None and right.value is not None
            try:
                return _Expr("", None, fn(left.value, right.value))
            except (ZeroDivisionError, OverflowError, ValueError) as exc:
                self.errors.add(RuleId.CONSTANT_RANGE, f"constant `{symbol}` fails: {exc}", node)
                raise _Abort from None
        assert isinstance(t, str)
        if isinstance(op, ast.Pow):
            return self._pow(left, right, t, node)
        a, b = self._coerce(left, t, node), self._coerce(right, t, node)
        if isinstance(op, ast.Mod):
            divisor = right.value if right.is_const else None
            if t == "i32" and isinstance(divisor, int) and not isinstance(divisor, bool) and divisor > 0:
                if divisor & (divisor - 1) == 0:
                    # Floor-mod by 2**k is the low k bits; int32_t is two's complement by definition.
                    return _Expr(f"{_wrap(left, a, 15)} & {divisor - 1}", t, prec=_PREC_AND_BITS)
                self.helpers.add("mod_int32_t_positive")
                return _Expr(f"cignal_mod_int32_t_positive({a}, {b})", t)
            return self._helper_call("mod_" + self.target.ctype(t), a, b, t)
        if isinstance(op, ast.FloorDiv):
            return self._helper_call("floordiv_" + self.target.ctype(t), a, b, t)
        if isinstance(op, (ast.LShift, ast.RShift)):
            return self._helper_call(("shl" if isinstance(op, ast.LShift) else "shr") + "_int32_t", a, b, t)
        if isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            prec = _BIT_PREC[type(op)]
            return _Expr(f"{_wrap(left, a, 15)} {symbol} {_wrap(right, b, 15)}", t, prec=prec)
        if t == "i32" and isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
            return self._helper_call({ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}[type(op)] + "_i32", a, b, t)
        prec = _PREC_MUL if isinstance(op, (ast.Mult, ast.Div)) else _PREC_ADD
        return _Expr(f"{_wrap(left, a, prec)} {symbol} {_wrap(right, b, prec, right=True)}", t, prec=prec)

    def _helper_call(self, helper: str, a: str, b: str, t: str) -> _Expr:
        self.helpers.add(helper)
        return _Expr(f"cignal_{helper}({a}, {b})", t)

    def _pow(self, left: _Expr, right: _Expr, t: str, node: ast.AST) -> _Expr:
        if t == "i32":
            exponent = right.value if right.is_const else None
            if not (isinstance(exponent, int) and not isinstance(exponent, bool) and exponent >= 0):
                self.errors.add(
                    RuleId.UNSUPPORTED_SYNTAX,
                    "i32 `**` needs a non-negative constant exponent",
                    node,
                    "Python returns a float for a negative exponent; cast the base with f32(...)",
                )
                raise _Abort
            return self._helper_call("pow_int32_t", self._coerce(left, t, node), self._coerce(right, t, node), t)
        # CPython computes float ** float with libm pow, so even a constant integer exponent calls pow.
        name = "powf" if self.target.ctype(t) == "float" else "pow"
        self.program.math_symbols.add(name)
        return _Expr(f"{name}({self._coerce(left, t, node)}, {self._coerce(right, t, node)})", t)

    def _unary(self, node: ast.UnaryOp) -> _Expr:
        operand = self._expr(node.operand)
        if isinstance(operand.type, tuple) or operand.type == NONE:
            self.errors.add(RuleId.TYPE_MISMATCH, f"unary operator on {_type_text(operand.type)}", node)
            raise _Abort
        if isinstance(node.op, ast.Not):
            if operand.is_const and isinstance(operand.value, bool):
                return _Expr("", None, not operand.value)
            if operand.type != "bool":
                self.errors.add(RuleId.TYPE_MISMATCH, "`not` needs a bool operand", node)
                raise _Abort
            return _Expr(f"!{_wrap(operand, operand.code, _PREC_UNARY, right=True)}", "bool", prec=_PREC_UNARY)
        if operand.type == "bool" or (operand.is_const and isinstance(operand.value, bool)):
            self.errors.add(RuleId.TYPE_MISMATCH, "arithmetic on bool", node)
            raise _Abort
        if isinstance(node.op, ast.Invert):
            if not (operand.type == "i32" or (operand.is_const and isinstance(operand.value, int))):
                self.errors.add(RuleId.TYPE_MISMATCH, "`~` needs an i32 operand", node)
                raise _Abort
            if operand.is_const:
                assert isinstance(operand.value, int)
                return _Expr("", None, ~operand.value)
            return _Expr(f"~{_wrap(operand, operand.code, 15)}", "i32", prec=_PREC_UNARY)
        if operand.is_const:
            assert operand.value is not None
            return _Expr("", None, -operand.value if isinstance(node.op, ast.USub) else operand.value)
        if isinstance(node.op, ast.UAdd):
            return operand
        if operand.type == "i32":
            self.helpers.add("neg_i32")
            return _Expr(f"cignal_neg_i32({operand.code})", "i32")
        # Unary operands are wrapped at equal precedence too, so `-(-x)` never prints as `--x`.
        return _Expr(f"-{_wrap(operand, operand.code, _PREC_UNARY, right=True)}", operand.type, prec=_PREC_UNARY)

    def _compare(self, node: ast.Compare) -> _Expr:
        if len(node.ops) > 1:
            # `a < b < c` is `a < b and b < c` with b evaluated once; C evaluates it twice, so no calls.
            middle = node.comparators[:-1]
            if any(isinstance(n, ast.Call) for m in middle for n in ast.walk(m)):
                self._unsupported(node, "a call in the middle of a chained comparison")
            pairs = [ast.Compare(left=lhs, ops=[op], comparators=[rhs]) for lhs, op, rhs in
                     zip([node.left, *middle], node.ops, node.comparators, strict=True)]  # fmt: skip
            for pair in pairs:
                ast.copy_location(pair, node)
            parts = [self._compare(pair) for pair in pairs]
            if all(p.is_const for p in parts):
                return _Expr("", None, all(p.value for p in parts))
            # Constant parts stay in: `&&` then short-circuits in Python's order, keeping calls at the ends.
            return _Expr(" && ".join(self._coerce(p, "bool", node) for p in parts), "bool", prec=_PREC_AND)
        if type(node.ops[0]) not in _COMPARES:
            self._unsupported(node, f"comparison `{_op_text(node.ops[0])}`")
        symbol, fn = _COMPARES[type(node.ops[0])]
        left, right = self._expr(node.left), self._expr(node.comparators[0])
        t = self._unify(left, right, node=node)
        if isinstance(t, tuple) or t == NONE:
            self.errors.add(RuleId.TYPE_MISMATCH, f"comparison of {_type_text(t)}", node)
            raise _Abort
        if t is None:
            assert left.value is not None and right.value is not None
            return _Expr("", None, fn(left.value, right.value))
        a, b = self._coerce(left, t, node), self._coerce(right, t, node)
        # Operands that are themselves comparisons keep parentheses: GCC's -Wparentheses flags `a < b == c`.
        return _Expr(f"{_wrap(left, a, _PREC_REL + 1)} {symbol} {_wrap(right, b, _PREC_REL + 1)}", "bool",
                     prec=_PREC_REL if symbol in ("<", "<=", ">", ">=") else _PREC_EQ)  # fmt: skip

    def _boolop(self, node: ast.BoolOp) -> _Expr:
        symbol = "&&" if isinstance(node.op, ast.And) else "||"
        parts = []
        for value in node.values:
            expr = self._expr(value)
            if expr.type != "bool" and not (expr.is_const and isinstance(expr.value, bool)):
                self.errors.add(RuleId.TYPE_MISMATCH, f"`{ast.unparse(node.op)}` needs bool operands", value)
                raise _Abort
            # `&&` inside `||` keeps parentheses: -Wparentheses flags it otherwise.
            parts.append(_wrap(expr, self._coerce(expr, "bool", value), _PREC_AND + 1))
        return _Expr(f" {symbol} ".join(parts), "bool", prec=_PREC_AND if symbol == "&&" else _PREC_OR)

    def _ifexp(self, node: ast.IfExp) -> _Expr:
        test = self._expr(node.test)
        if test.type != "bool" and not (test.is_const and isinstance(test.value, bool)):
            self.errors.add(RuleId.TYPE_MISMATCH, "a conditional expression's test must be bool", node.test)
            raise _Abort
        body, orelse = self._expr(node.body), self._expr(node.orelse)
        t = self._unify(body, orelse, node=node)
        if isinstance(t, tuple) or t == NONE:
            self.errors.add(RuleId.TYPE_MISMATCH, f"conditional expression of {_type_text(t)}", node)
            raise _Abort
        if t is None:
            if test.is_const:
                return body if test.value else orelse
            floats = any(isinstance(e.value, float) for e in (body, orelse))
            t = "f32" if floats else self._const_type(body)
        parts = [_wrap(e, self._coerce(e, ty, n), _PREC_COND, right=True)
                 for e, ty, n in ((test, "bool", node.test), (body, t, node), (orelse, t, node))]  # fmt: skip
        return _Expr(f"{parts[0]} ? {parts[1]} : {parts[2]}", t, prec=_PREC_COND)

    def _call(self, node: ast.Call) -> _Expr:
        if node.keywords:
            self._unsupported(node, "keyword arguments")
        func = node.func
        if isinstance(func, ast.Name) and func.id in _CASTS and self.kernel.is_cignal_alias(func.id):
            if len(node.args) != 1:
                self.errors.add(RuleId.UNSUPPORTED_CALL, f"{func.id}() takes one argument", node)
                raise _Abort
            return self._cast(func.id, self._expr(node.args[0]), node)
        if isinstance(func, ast.Name) and isinstance(getattr(self.kernel.module, func.id, None), Table):
            return self._table_call(func.id, node)
        if isinstance(func, ast.Name) and func.id in self.program.free_nodes:
            return self._free_call(func.id, node)
        if isinstance(func, ast.Name) and func.id in _BUILTINS and self._is_builtin(func.id):
            return self._builtin(func.id, node)
        if isinstance(func, ast.Attribute):
            if self.kernel.is_module(func.value, math) and func.attr in _MATH_FUNCTIONS:
                return self._math(func.attr, node)
            if isinstance(func.value, ast.Attribute) and func.attr in _DELAY_METHODS:
                code, field_ = self._member(func.value)
                if field_.delay is not None and not field_.count:
                    return self._delay_call(code, field_.delay, func.attr, node)
            if (isinstance(func.value, ast.Subscript) and isinstance(func.value.value, ast.Attribute)
                    and func.attr in _DELAY_METHODS):  # fmt: skip
                found = self._element(func.value)
                if found is not None and found[1].delay is not None:
                    return self._delay_call(found[0], found[1].delay, func.attr, node)
            receiver = self._receiver(func.value)
            if receiver is not None and func.attr == "target":
                return self._target_call(receiver[2], node)
            if receiver is not None:
                return self._method_call(receiver, func.attr, node)
        self.errors.add(RuleId.UNSUPPORTED_CALL, f"call to `{ast.unparse(func)}`", node,
                        "supported: f32()/f64()/i32(), abs()/min()/max(), math." + "/".join(_MATH_FUNCTIONS) +
                        ", and methods of this kernel or its sub-kernels")  # fmt: skip
        raise _Abort

    def _method_call(self, receiver: tuple[str, str, "_KernelModel"], name: str, node: ast.Call) -> _Expr:
        _, pointer, model = receiver
        signature = model.signatures.get(name)
        if signature is None or name == "init":
            why = "`init` runs automatically" if name == "init" else f"`{model.name}` has no method `{name}`"
            self.errors.add(RuleId.UNSUPPORTED_CALL, why, node)
            raise _Abort
        if len(node.args) != len(signature.params):
            self.errors.add(RuleId.BAD_SIGNATURE, f"`{name}` takes {len(signature.params)} argument(s)", node)
            raise _Abort
        args = [self._coerce(self._expr(a), t, a) for a, t in zip(node.args, signature.params.values(), strict=True)]
        self.program.calls.setdefault(self.key, set()).add((model.name, name))
        if isinstance(signature.returns, tuple):
            self.program.tuples.add(signature.returns)
        call = f"{model.cname}_{name}({', '.join([pointer, *args])})"
        return self._leaf(_Expr(call, signature.returns if signature.returns is not None else NONE))

    def _target_call(self, model: "_KernelModel", node: ast.Call) -> _Expr:
        """`self.target("cutoff")`: the value a parameter glides towards, which changes only between blocks."""
        arg = node.args[0] if node.args else None
        name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
        field_ = model.fields.get(name) if name is not None and model is self.kernel else None
        if field_ is None or field_.param is None or not model.exported or \
                len(node.args) != (2 if field_.array else 1):  # fmt: skip
            self.errors.add(RuleId.BAD_PARAM, "`target` takes the name of a parameter of the exported kernel, "
                            "as a string literal, and an index for an `Array` parameter", node,
                            'e.g. `self.target("cutoff")` or `self.target("cutoffs", i)`')  # fmt: skip
            raise _Abort
        assert name is not None
        code = f"self->cignal_target_{name}" if field_.smooth > 0 else f"self->{c_name(name)}"
        if field_.array is None:
            return self._leaf(_Expr(code, field_.type))
        index = self._proven_index(node.args[1], self._expr(node.args[1]), field_.array[1])
        return self._leaf(_Expr(f"{code}[{index}]", field_.array[0]))

    def _free_call(self, name: str, node: ast.Call) -> _Expr:
        signature = self.program.free_function(name)
        if len(node.args) != len(signature.params):
            self.errors.add(RuleId.BAD_SIGNATURE, f"`{name}` takes {len(signature.params)} argument(s)", node)
            raise _Abort
        args = [self._coerce(self._expr(a), t, a) for a, t in zip(node.args, signature.params.values(), strict=True)]
        self.program.calls.setdefault(self.key, set()).add(("", name))
        if isinstance(signature.returns, tuple):
            self.program.tuples.add(signature.returns)
        call = f"cignal_fn_{name}({', '.join(args)})"
        return _Expr(call, signature.returns if signature.returns is not None else NONE)

    def _table_call(self, name: str, node: ast.Call) -> _Expr:
        t = self.program.table_type(name)
        if len(node.args) != 1:
            self.errors.add(RuleId.BAD_SIGNATURE, f"table `{name}` takes one argument", node)
            raise _Abort
        arg = self._coerce(self._expr(node.args[0]), t, node.args[0])
        self.helpers.add(f"table_{name}")
        return _Expr(f"cignal_lookup_{name}({arg})", t)

    def _delay_call(self, code: str, delay: tuple[str, int], method: str, node: ast.Call) -> _Expr:
        element, size = delay
        if len(node.args) != _DELAY_METHODS[method]:
            self.errors.add(RuleId.BAD_SIGNATURE, f"`{method}` takes {_DELAY_METHODS[method]} argument(s)", node)
            raise _Abort
        ctype = self.target.ctype(element)
        self.helpers.add(f"delay_{method}_{ctype}")
        arg_type = {"write": element, "read": "i32", "read_frac": "f64"}.get(method)
        args = []
        if arg_type:
            arg = self._expr(node.args[0])
            if method == "read_frac" and arg.type == "f32":
                arg = _Expr(f"(double){_wrap(arg, arg.code, _PREC_UNARY, right=True)}", "f64", prec=_PREC_UNARY)
            args.append(self._coerce(arg, arg_type, node.args[0]))
        buffer = f"{code}.buf, {size}"
        if method in ("write", "clear"):
            return _Expr(f"cignal_delay_{method}_{ctype}({', '.join([buffer, f'&{code}.pos', *args])})", NONE)
        return self._leaf(_Expr(f"cignal_delay_{method}_{ctype}({', '.join([buffer, f'{code}.pos', *args])})", element))

    def _cast(self, t: str, operand: _Expr, node: ast.AST) -> _Expr:
        if operand.type in ("bool", NONE) or isinstance(operand.type, tuple) or isinstance(operand.value, bool):
            self.errors.add(RuleId.TYPE_MISMATCH, f"{t}() of {_type_text(operand.type)}", node)
            raise _Abort
        if operand.is_const:
            assert operand.value is not None
            try:
                value: Number = _CASTS[t](operand.value)
            except (OverflowError, ValueError):
                self.errors.add(RuleId.CONSTANT_RANGE, f"{t}({operand.value!r}) is not finite", node)
                raise _Abort from None
            negative = not isinstance(value, bool) and value < 0
            return _Expr(_literal(value, t, self.target, self.errors, node), t, prec=_PREC_UNARY if negative else 16)
        if operand.type == t:
            return operand
        if t == "i32":
            self.helpers.add("to_i32")
            return _Expr(f"cignal_to_i32((double){_wrap(operand, operand.code, _PREC_UNARY, right=True)})", t)
        cast = f"({self.target.ctype(t)}){_wrap(operand, operand.code, _PREC_UNARY, right=True)}"
        return _Expr(cast, t, prec=_PREC_UNARY)

    def _is_builtin(self, name: str) -> bool:
        """`name` is Python's built-in: the kernel's module does not rebind it."""
        return getattr(self.kernel.module, name, getattr(builtins, name)) is getattr(builtins, name)

    def _builtin(self, name: str, node: ast.Call) -> _Expr:
        """`abs`, and `min`/`max` of two or more arguments, folded left as CPython evaluates them."""
        if (name == "abs") != (len(node.args) == 1) or not node.args:
            arity = "one argument" if name == "abs" else "two or more arguments"
            self.errors.add(RuleId.UNSUPPORTED_CALL, f"{name}() takes {arity}", node)
            raise _Abort
        args = [self._expr(arg) for arg in node.args]
        t = self._unify(*args, node=node)
        if t not in (None, "i32", *FLOAT_TYPES) or any(isinstance(a.value, bool) for a in args):
            self.errors.add(RuleId.TYPE_MISMATCH, f"{name}() of {_type_text(t or 'bool')}", node)
            raise _Abort
        if t is None:
            return _Expr("", None, getattr(builtins, name)(*(a.value for a in args)))
        assert isinstance(t, str)
        ctype = self.target.ctype(t)
        if name == "abs" and t == "i32":
            self.helpers.add("abs_int32_t")
            return _Expr(f"cignal_abs_int32_t({self._coerce(args[0], t, node)})", t)
        if name == "abs":
            symbol = "fabsf" if ctype == "float" else "fabs"
            self.program.math_symbols.add(symbol)
            return _Expr(f"{symbol}({self._coerce(args[0], t, node)})", t)
        helper = f"{name}_{ctype}"
        self.helpers.add(helper)
        code = self._coerce(args[0], t, node)
        for arg in args[1:]:
            code = f"cignal_{helper}({code}, {self._coerce(arg, t, node)})"
        return _Expr(code, t)

    def _math(self, name: str, node: ast.Call) -> _Expr:
        if len(node.args) != _MATH_FUNCTIONS[name]:
            self.errors.add(RuleId.UNSUPPORTED_CALL, f"math.{name} takes {_MATH_FUNCTIONS[name]} argument(s)", node)
            raise _Abort
        args = [self._expr(arg) for arg in node.args]
        t = self._unify(*args, node=node)
        if t is None:
            try:
                return _Expr("", None, getattr(math, name)(*(a.value for a in args)))
            except (ValueError, OverflowError):
                self.errors.add(RuleId.CONSTANT_RANGE, f"math.{name} of these constants is not finite", node)
                raise _Abort from None
        if t not in FLOAT_TYPES:
            self.errors.add(RuleId.MIXED_NUMERIC, f"math.{name} of {_type_text(t)}", node,
                            "cast the argument with f32(...)")  # fmt: skip
            raise _Abort
        assert isinstance(t, str)
        symbol = name + ("f" if self.target.ctype(t) == "float" else "")
        self.program.math_symbols.add(symbol)
        return _Expr(f"{symbol}(" + ", ".join(self._coerce(a, t, node) for a in args) + ")", t)

    # -- evaluation order -------------------------------------------------
    # Python evaluates operands and arguments left to right; C leaves both orders unspecified (C99 6.5p3,
    # 6.5.2.2p10). Where a statement's order is observable, every read of kernel state and every method
    # call is hoisted into a temporary, in Python's order.

    def _order_sensitive(self, nodes: list[ast.expr]) -> bool:
        """Whether `nodes` call a method that may write state and touch state anywhere else.

        `and`, `or`, `if`-`else` and chained comparisons are sequence points in C, so only their operands count.
        """
        if len(nodes) == 1 and _is_conditional(nodes[0]):
            return any(self._order_sensitive([n]) for n in _operands(nodes[0]))
        walked = [n for node in nodes for n in ast.walk(node)]
        selfs = sum(isinstance(n, ast.Name) and n.id == "self" for n in walked)
        return selfs >= 2 and any(self._is_effect_call(n) for n in walked)

    def _is_effect_call(self, node: ast.AST) -> bool:
        """A call of a method of `self` or a sub-kernel, which may write state; delay reads and `target` do not."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            return False
        root = node.func.value
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if not (isinstance(root, ast.Name) and root.id == "self" and self.has_self):
            return False
        if node.func.attr == "target" and isinstance(node.func.value, ast.Name):
            return False
        if node.func.attr in ("read", "read_frac"):
            field_ = self._static_field(node.func.value)
            return field_ is None or field_.delay is None
        return True

    def _static_field(self, node: ast.expr) -> Field | None:
        """The field `self.a.b` or `self.a[i]` names, resolved without lowering anything."""
        if isinstance(node, ast.Subscript):
            node = node.value
        if not isinstance(node, ast.Attribute):
            return None
        if isinstance(node.value, ast.Name):
            model = self.kernel if node.value.id == "self" else None
        else:
            base = self._static_field(node.value)
            model = self.program.models.get(base.kernel) if base is not None and base.kernel else None
        return model.fields.get(node.attr) if model is not None else None

    def _leaf(self, expr: _Expr) -> _Expr:
        """In an order-sensitive statement, evaluate `expr` now, into a temporary."""
        if not self.sequenced or expr.type is None or expr.type == NONE:
            return expr
        temp = f"cignal_t{next(self.temps)}"
        self._emit(self.depth, f"const {self.target.ctype(expr.type)} {temp} = {expr.code};")
        return _Expr(temp, expr.type)

    def _conditional_leaf(self, node: ast.expr) -> _Expr:
        """`and`, `or`, `if`-`else` or a chained comparison, evaluated whole into a temporary.

        Its operands stay inline, since hoisting would evaluate them unconditionally. C sequences them, so
        only an operand whose own order is observable is an error.
        """
        for operand in _operands(node):
            if self._order_sensitive([operand]):
                self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"`{ast.unparse(operand)}` calls a method and touches "
                                "other state inside `and`, `or` or `if`-`else`, where C would not keep Python's "
                                "evaluation order", operand, "compute it into a local first")  # fmt: skip
                raise _Abort
        self.sequenced = False
        try:
            expr = self._expr(node)
        finally:
            self.sequenced = True
        return self._leaf(expr)

    # -- typing -----------------------------------------------------------

    def _unify(self, *exprs: _Expr, node: ast.AST) -> Ty | None:
        """Return the shared type of `exprs`; constants take the type of their neighbours."""
        types = {e.type for e in exprs if e.type is not None}
        if len(types) > 1:
            self.errors.add(
                RuleId.MIXED_NUMERIC,
                "operands have different types: " + " and ".join(sorted(map(_type_text, types))),
                node,
                "cast explicitly with f32(), f64() or i32()",
            )
            raise _Abort
        return types.pop() if types else None

    def _coerce(self, expr: _Expr, t: str, node: ast.AST) -> str:
        if expr.is_const:
            assert expr.value is not None
            return _literal(expr.value, t, self.target, self.errors, node)
        if expr.type != t:
            numeric = expr.type in _CASTS and t in _CASTS
            rule = RuleId.MIXED_NUMERIC if numeric else RuleId.TYPE_MISMATCH
            self.errors.add(rule, f"expected {t}, got {_type_text(expr.type)}", node,
                            f"cast with {t}(...)" if numeric else None)  # fmt: skip
            raise _Abort
        return expr.code

    @staticmethod
    def _const_type(expr: _Expr) -> str:
        if isinstance(expr.value, bool):
            return "bool"
        return "i32" if isinstance(expr.value, int) else "f32"

    def _unsupported(self, node: ast.AST, what: str) -> NoReturn:
        self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"{what} is not supported", node)
        raise _Abort


def _is_conditional(node: ast.expr) -> bool:
    return isinstance(node, (ast.BoolOp, ast.IfExp)) or (isinstance(node, ast.Compare) and len(node.ops) > 1)


def _operands(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp):
        return node.values
    if isinstance(node, ast.IfExp):
        return [node.test, node.body, node.orelse]
    assert isinstance(node, ast.Compare)
    return [node.left, *node.comparators]


def _mentions_self(node: ast.expr) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "self" for n in ast.walk(node))


def _statement_exprs(stmt: ast.stmt) -> list[ast.expr]:
    """The expressions a statement evaluates, in Python's order, for `_order_sensitive`."""
    if isinstance(stmt, ast.Assign):
        return [stmt.value, *stmt.targets]
    if isinstance(stmt, ast.AugAssign):
        return [stmt.target, stmt.value]
    if isinstance(stmt, (ast.AnnAssign, ast.Return)):
        return [stmt.value] if stmt.value is not None else []
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, ast.For):
        return [stmt.iter]
    if isinstance(stmt, ast.Expr):
        return [stmt.value]
    return []


def _all_int(*exprs: _Expr) -> bool:
    return all(isinstance(e.value, int) and not isinstance(e.value, bool) for e in exprs)


def _op_text(op: ast.AST) -> str:
    return {
        ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>", ast.BitOr: "|",
        ast.BitXor: "^", ast.BitAnd: "&", ast.MatMult: "@", ast.Invert: "~", ast.Is: "is", ast.IsNot: "is not",
        ast.In: "in", ast.NotIn: "not in",
    }.get(type(op), type(op).__name__)  # fmt: skip


class _KernelModel:
    """Fields, method signatures and build-time values of one kernel class."""

    def __init__(self, program: "_Program", name: str, exported: bool) -> None:
        self.program = program
        self.name = name
        self.exported = exported
        self.module = program.module
        self.node = program.classes[name]
        self.cls: type = getattr(program.module, name)
        self.errors = program.errors
        self.fields: dict[str, Field] = {"sample_rate": Field("sample_rate", "f32", 0.0)}
        self.methods: dict[str, ast.FunctionDef] = {}
        self.signatures: dict[str, Signature] = {}
        self._collect()

    def resolve_scalar(self, annotation: ast.expr) -> str | None:
        """Resolve a scalar type annotation, reporting anything else."""
        name = annotation.attr if isinstance(annotation, ast.Attribute) else getattr(annotation, "id", None)
        if name in ("float", "int"):
            suggestion = "f32 or f64" if name == "float" else "i32"
            self.errors.add(
                RuleId.IMPLICIT_PRECISION, f"`{name}` does not name a precision", annotation, f"use {suggestion}"
            )
            return None
        if name == "bool" or (name in _CASTS and self.is_cignal_alias(name, annotation)):
            return name
        self.errors.add(RuleId.UNKNOWN_TYPE, f"unsupported type `{ast.unparse(annotation)}`", annotation,
                        "types: f32, f64, i32, bool, or a kernel class defined in this file")  # fmt: skip
        return None

    def resolve_returns(self, annotation: ast.expr | None) -> Ty | None:
        if annotation is None or (isinstance(annotation, ast.Constant) and annotation.value is None):
            return None
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
            if annotation.value.id == "tuple":
                elts = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
                types = [self.resolve_scalar(e) for e in elts]
                if any(t is None for t in types):
                    raise _Abort
                return tuple(t for t in types if t is not None)
        t = self.resolve_scalar(annotation)
        if t is None:
            raise _Abort
        return t

    def is_cignal_alias(self, name: str, node: ast.expr | None = None) -> bool:
        """Whether `name` (or `cignal.name`) refers to cignal's alias in the kernel module."""
        if isinstance(node, ast.Attribute):
            return (
                isinstance(node.value, ast.Name) and getattr(self.module, node.value.id, None) is sys.modules["cignal"]
            )
        return getattr(self.module, name, None) is _CASTS[name]

    def is_module(self, node: ast.expr, module: ModuleType) -> bool:
        return isinstance(node, ast.Name) and getattr(self.module, node.id, None) is module

    def constant(self, name: str) -> Number | None:
        value = getattr(self.module, name, None)
        return value if isinstance(value, (bool, int, float)) else None

    @property
    def cname(self) -> str:
        """The C identifier: `--prefix` plus the class name."""
        return self.program.prefix + self.name

    @property
    def params(self) -> list[Field]:
        return [f for f in self.fields.values() if f.param is not None] if self.exported else []

    @property
    def slots(self) -> list[tuple[str, Field, int | None]]:
        """Host parameters: one per scalar parameter, one per element of an `Array` parameter, named `x_k`."""
        return [(f.name if k is None else f"{f.name}_{k}", f, k)
                for f in self.params for k in (range(f.array[1]) if f.array else [None])]  # fmt: skip

    def _collect(self) -> None:
        for index, stmt in enumerate(self.node.body):
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self._field(stmt, stmt.target.id)
            elif isinstance(stmt, ast.FunctionDef):
                self.methods[stmt.name] = stmt
            elif isinstance(stmt, ast.Pass) or (
                index == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            ):
                continue
            else:
                self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"`{type(stmt).__name__}` in a kernel class body", stmt,
                                "declare fields as `name: f32 = default`")  # fmt: skip
        if self.cls.__mro__[1] is not Kernel:
            self.errors.add(RuleId.UNSUPPORTED_SYNTAX, "kernel inheritance", self.node, "derive directly from Kernel")
        for f in self.fields.values():
            if f.param is not None and f.array is not None:
                for k in range(f.array[1]):
                    if f"{f.name}_{k}" in self.fields:
                        self.errors.add(RuleId.BAD_PARAM, f"field `{f.name}_{k}` has the host name of element {k} "
                                        f"of parameter `{f.name}`", self.node)  # fmt: skip
        for name, method in self.methods.items():
            if name in _RESERVED_METHODS:
                self.errors.add(RuleId.BAD_SIGNATURE, f"`{name}` is reserved by the Kernel runtime", method)
                continue
            try:
                self.signatures[name] = _signature(self, method)
            except _Abort:
                pass

    def _field(self, stmt: ast.AnnAssign, name: str) -> None:
        if name == "sample_rate":
            self.errors.add(RuleId.UNSUPPORTED_SYNTAX, "`sample_rate` is predefined", stmt)
            return
        annotation = stmt.annotation
        if isinstance(annotation, ast.Name) and annotation.id in self.program.classes:
            if stmt.value is not None:
                self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"a default for sub-kernel `{name}`", stmt,
                                "sub-kernels are constructed automatically")  # fmt: skip
                return
            self.fields[name] = Field(name, None, kernel=annotation.id)
            return
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
            kind = getattr(self.module, annotation.value.id, None)
            if kind is Array or kind is Delay:
                self._buffer(stmt, name, kind)
                return
        t = self.resolve_scalar(annotation)
        if t is None:
            return
        default = self.cls.__dict__.get(name)
        if isinstance(default, Param):
            self._param(stmt, name, t, default)
            return
        if not isinstance(default, (bool, int, float)) or stmt.value is None:
            self.errors.add(RuleId.MISSING_DEFAULT, f"field `{name}` needs a numeric default", stmt,
                            f"`{name}: {t} = 0`")  # fmt: skip
            return
        self.fields[name] = Field(name, t, default)

    def _buffer(self, stmt: ast.AnnAssign, name: str, kind: type) -> None:
        """An `Array[T, N]` or `Delay[T, N]` field. T comes from the source, N from the evaluated annotation."""
        label = kind.__name__
        annotation = stmt.annotation
        assert isinstance(annotation, ast.Subscript)
        elts = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else []
        default = self.cls.__dict__.get(name)
        is_param = kind is Array and isinstance(default, Param)
        if len(elts) != 2 or (stmt.value is not None and not is_param):
            self.errors.add(RuleId.UNSUPPORTED_SYNTAX, f"`{label}` fields are written `{name}: {label}[f32, N]` "
                            "with no default", stmt)  # fmt: skip
            return
        spec = inspect.get_annotations(self.cls, eval_str=True).get(name)
        assert isinstance(spec, Spec)
        size = self._buffer_size(kind, spec, elts[1])
        element = elts[0]
        if is_param:
            assert isinstance(default, Param)
            t = self.resolve_scalar(element)
            if t is not None and size is not None:
                self._param(stmt, name, t, default, size)
            return
        if kind is Array and isinstance(element, ast.Name) and element.id in self.program.classes:
            if size is not None:
                self.fields[name] = Field(name, None, kernel=element.id, count=size)
            return
        if kind is Array and isinstance(element, ast.Subscript) and isinstance(element.value, ast.Name):
            if getattr(self.module, element.value.id, None) is Delay and isinstance(spec.element, Spec):
                inner = element.slice.elts if isinstance(element.slice, ast.Tuple) else []
                delay = self._delay(spec.element, inner) if len(inner) == 2 else None
                if delay is not None and size is not None:
                    self.fields[name] = Field(name, None, delay=delay, count=size)
                return
        if kind is Delay:
            delay = self._delay(spec, elts)
            if delay is not None:
                self.fields[name] = Field(name, None, delay=delay)
            return
        t = self.resolve_scalar(element)
        if t is not None and size is not None:
            self.fields[name] = Field(name, None, array=(t, size))

    def _buffer_size(self, kind: type, spec: Spec, node: ast.expr) -> int | None:
        size, label, minimum = spec.size, kind.__name__, 2 if kind is Delay else 1
        if not isinstance(size, int) or isinstance(size, bool) or not minimum <= size <= MAX_BUFFER:
            self.errors.add(RuleId.CONSTANT_RANGE, f"`{label}` length must be an int in [{minimum}, {MAX_BUFFER}], "
                            f"got {size!r}", node)  # fmt: skip
            return None
        return size

    def _delay(self, spec: Spec, elts: list[ast.expr]) -> tuple[str, int] | None:
        """`Delay[T, N]`, alone or as an `Array` element: (element type, length)."""
        t = self.resolve_scalar(elts[0])
        size = self._buffer_size(Delay, spec, elts[1])
        if t is None or size is None:
            return None
        if t not in FLOAT_TYPES:
            self.errors.add(RuleId.UNKNOWN_TYPE, f"`Delay` elements must be f32 or f64, not {t}", elts[0])
            return None
        return t, size

    def _param(self, stmt: ast.AnnAssign, name: str, t: str, p: Param, size: int = 0) -> None:
        """A scalar parameter, or an `Array` of `size` parameters sharing one range, default and smoothing."""
        values = (p.default, p.lo, p.hi)
        if t not in FLOAT_TYPES:
            self.errors.add(RuleId.BAD_PARAM, f"parameter `{name}` must be f32 or f64, not {t}", stmt)
        elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            self.errors.add(RuleId.BAD_PARAM, f"parameter `{name}` needs finite numeric default, lo and hi", stmt)
        elif not p.lo <= p.default <= p.hi:
            self.errors.add(RuleId.BAD_PARAM, f"parameter `{name}`: need lo <= default <= hi, got "
                            f"{p.lo} <= {p.default} <= {p.hi}", stmt)  # fmt: skip
        elif not (isinstance(p.smooth, (int, float)) and math.isfinite(p.smooth) and p.smooth >= 0):
            self.errors.add(RuleId.BAD_PARAM, f"parameter `{name}`: smooth must be a finite time >= 0 in seconds, "
                            f"got {p.smooth!r}", stmt)  # fmt: skip
        else:
            self.fields[name] = Field(name, None if size else t, float(p.default), param=(float(p.lo), float(p.hi)),
                                      smooth=float(p.smooth), array=(t, size) if size else None)  # fmt: skip


def _signature(model: _KernelModel, method: ast.FunctionDef, free: bool = False) -> Signature:
    """Check a method signature, or a module-level function's (`free`, no `self`).

    Lifecycle methods have fixed shapes; others take scalars.
    """
    args = method.args
    errors = model.errors
    name = method.name
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults:
        errors.add(RuleId.BAD_SIGNATURE, f"`{name}` may take only plain positional parameters", method)
    marked = [d for d in method.decorator_list if not free and _is_event_decorator(model, d)]
    for d in method.decorator_list:
        if d not in marked:
            errors.add(RuleId.UNSUPPORTED_SYNTAX, "decorated methods, other than `@event`", d)
    if not free and (not args.args or args.args[0].arg != "self"):
        errors.add(RuleId.BAD_SIGNATURE, f"`{name}` must take `self` first", method)
        raise _Abort
    params: dict[str, str] = {}
    for arg in args.args if free else args.args[1:]:
        if arg.annotation is None:
            errors.add(RuleId.BAD_SIGNATURE, f"parameter `{arg.arg}` needs an annotation", arg)
            raise _Abort
        t = model.resolve_scalar(arg.annotation)
        if t is None:
            raise _Abort
        params[arg.arg] = t
    returns = model.resolve_returns(method.returns)
    if marked or name.startswith("on_"):
        if not marked:
            errors.add(RuleId.BAD_SIGNATURE, f"`{name}` is named as an event: mark it `@event`", method,
                       "or rename it without the `on_` prefix")  # fmt: skip
        elif not name.startswith("on_") or name == "on_":
            errors.add(RuleId.BAD_SIGNATURE, f"event method `{name}` must be named `on_<event>`", method)
        elif name[3:] in PD_SELECTORS:
            errors.add(RuleId.BAD_SIGNATURE, f"event `{name[3:]}` is reserved by a host", method,
                       f"reserved: {', '.join(sorted(PD_SELECTORS))}")  # fmt: skip
        if returns is not None:
            errors.add(RuleId.BAD_SIGNATURE, f"event `{name}` must return None", method)
        return Signature(params, returns, method, event=bool(marked))
    if free:
        if returns is not None and (not method.body or not isinstance(method.body[-1], ast.Return)):
            errors.add(RuleId.BAD_SIGNATURE, f"`{name}` must end with a `return`", method)
        return Signature(params, returns, method)
    if name in ("init", "control") and (params or returns is not None):
        errors.add(RuleId.BAD_SIGNATURE, f"`{name}` takes only `self` and returns None", method)
    if name == "tick":
        outputs = returns if isinstance(returns, tuple) else (returns,)
        if any(t != "f32" for t in params.values()) or not outputs or any(t != "f32" for t in outputs):
            errors.add(RuleId.BAD_SIGNATURE, "`tick` takes f32 samples and returns f32 or tuple[f32, ...]", method)
        if not method.body or not isinstance(method.body[-1], ast.Return):
            errors.add(RuleId.BAD_SIGNATURE, "`tick` must end with a `return`", method)
    elif returns is not None and (not method.body or not isinstance(method.body[-1], ast.Return)):
        errors.add(RuleId.BAD_SIGNATURE, f"`{name}` must end with a `return`", method)
    return Signature(params, returns, method)


def _is_event_decorator(model: _KernelModel, node: ast.expr) -> bool:
    """`@event` or `@cignal.event`, resolved in the kernel module."""
    if isinstance(node, ast.Name):
        return getattr(model.module, node.id, None) is event
    return isinstance(node, ast.Attribute) and node.attr == "event" and model.is_cignal_alias("event", node)


class _Program:
    """State shared by all kernels compiled from one file."""

    def __init__(self, module: ModuleType, tree: ast.Module, target: Target, errors: _Errors) -> None:
        self.module = module
        self.target = target
        self.errors = errors
        self.tree = tree
        self.prefix = ""
        self.classes = dict(_kernel_classes(module, tree))
        # Module-level functions kernel code may call. They are lowered only when called, so build-time
        # helpers (filter design with SciPy, say) never need to be valid kernel code.
        self.free_nodes = {
            n.name: n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and not n.decorator_list
            and inspect.isfunction(getattr(module, n.name, None))
        }
        self.free_signatures: dict[str, Signature] = {}
        self.lower_free: Callable[[str, Signature], None] = lambda name, signature: None
        self.table_types: dict[str, str | None] = {}
        self.const_arrays: dict[str, tuple[str, list[str]]] = {}  # tuple name -> (C type, literals)
        self.models: dict[str, _KernelModel] = {}
        self.exported: str | None = None
        self.math_symbols: set[str] = set()
        self.helpers: set[str] = set()
        self.tuples: set[tuple[str, ...]] = set()
        self.calls: dict[tuple[str, str], set[tuple[str, str]]] = {}

    def free_function(self, name: str) -> Signature:
        """The signature of a module-level function, lowering it on first use. Reports problems once."""
        if name not in self.free_signatures:
            node = self.free_nodes[name]
            scope = self.model(self.exported or "")
            try:
                signature = _signature(scope, node, free=True)
            except _Abort:
                signature = Signature({}, None, node, failed=True)
            self.free_signatures[name] = signature  # before lowering, so recursion finds it
            if not signature.failed:
                self.lower_free(name, signature)
        signature = self.free_signatures[name]
        if signature.failed:
            raise _Abort
        return signature

    def const_tuple(self, name: str) -> tuple[str, tuple[Number, ...]] | None:
        """A module-level tuple of numbers or of bools: (element type, values). `i32` if every number is an int."""
        value = getattr(self.module, name, None)
        if not isinstance(value, tuple) or not value:
            return None
        if all(isinstance(v, bool) for v in value):
            return "bool", value
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            return None
        return ("i32" if all(isinstance(v, int) for v in value) else "f32"), value

    def const_literals(self, name: str, t: str, values: tuple[Number, ...], node: ast.AST) -> None:
        """Render a tuple's elements once, reporting an element outside `t` at the first runtime index."""
        if name not in self.const_arrays:
            self.const_arrays[name] = (self.target.ctype(t), [_literal(v, t, self.target, self.errors, node)
                                                               for v in values])  # fmt: skip

    def const_code(self, name: str) -> str:
        ctype, values = self.const_arrays[name]
        rows = ",\n".join("    " + ", ".join(values[k : k + 8]) for k in range(0, len(values), 8))
        return f"static const {ctype} cignal_const_{name}[{len(values)}] = {{\n{rows}\n}};"

    def table_type(self, name: str) -> str:
        """Check a `@table` function and return its element type. Reports problems once per table."""
        if name not in self.table_types:
            self.table_types[name] = self._check_table(name)
        t = self.table_types[name]
        if t is None:
            raise _Abort
        return t

    def _check_table(self, name: str) -> str | None:
        table_: Table = getattr(self.module, name)
        node = next((n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
        where: ast.AST | None = node
        if node is None or len(node.args.args) != 1 or node.args.args[0].annotation is None:
            self.errors.add(RuleId.BAD_TABLE, f"table `{name}` must be a module-level `def {name}(x: f32) -> f32`",
                            where)  # fmt: skip
            return None
        scope = self.model(self.exported or "")
        annotation = node.args.args[0].annotation
        assert annotation is not None
        arg = scope.resolve_scalar(annotation)
        ret = scope.resolve_scalar(node.returns) if node.returns is not None else None
        if arg is None or arg != ret or arg not in FLOAT_TYPES:
            self.errors.add(RuleId.BAD_TABLE, f"table `{name}` must map f32 to f32 or f64 to f64", node)
            return None
        size, lo, hi = table_.size, table_.lo, table_.hi
        if not (isinstance(size, int) and not isinstance(size, bool) and 1 <= size <= MAX_BUFFER):
            self.errors.add(RuleId.BAD_TABLE, f"table `{name}` size must be an int in [1, {MAX_BUFFER}]", node)
            return None
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            self.errors.add(RuleId.BAD_TABLE, f"table `{name}` needs finite lo < hi, got {lo!r}, {hi!r}", node)
            return None
        if not all(math.isfinite(v) for v in table_.values):
            self.errors.add(RuleId.BAD_TABLE, f"table `{name}` has non-finite values on [{lo}, {hi}]", node)
            return None
        return arg

    def table_code(self, name: str) -> str:
        """The sampled values and the lookup, following `kernel.Table` operation for operation."""
        table_: Table = getattr(self.module, name)
        t = self.table_types[name]
        assert t is not None
        ctype = self.target.ctype(t)
        node = self.classes[self.exported or ""]

        def lit(v: float) -> str:
            return _literal(v, t, self.target, self.errors, node)

        values = [lit(v) for v in table_.values]
        rows = ",\n".join("    " + ", ".join(values[k : k + 4]) for k in range(0, len(values), 4))
        size, lo, hi, scale = table_.size, lit(table_.lo), lit(table_.hi), lit(table_.scale)
        # `x - (-c)` and `x + c` are the same IEEE operation, signed zeros included.
        offset = f"x + {lit(-table_.lo)}" if table_.lo < 0 else f"x - {lo}"
        return f"""static const {ctype} cignal_table_{name}[{size + 1}] = {{
{rows}
}};

static inline {ctype} cignal_lookup_{name}({ctype} x) {{
    if (!(x >= {lo})) x = {lo};
    else if (x > {hi}) x = {hi};
    {ctype} p = ({offset}) * {scale};
    int32_t i = (int32_t)p;
    if (i >= {size}) i = {size - 1};
    {ctype} frac = p - ({ctype})i;
    {ctype} a = cignal_table_{name}[i];
    return a + frac * (cignal_table_{name}[i + 1] - a);
}}"""

    def model(self, name: str) -> _KernelModel:
        if name not in self.models:
            self.models[name] = _KernelModel(self, name, exported=name == self.exported)
        return self.models[name]

    def held_kernels(self) -> set[str]:
        """Kernels that some other kernel in the file holds as a field."""
        held = set()
        for name, node in self.classes.items():
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign):  # `sub: K` or `subs: Array[K, N]`
                    names = {n.id for n in ast.walk(stmt.annotation) if isinstance(n, ast.Name)}
                    held |= (names & set(self.classes)) - {name}
        return held

    def ordered(self, root: str) -> list[_KernelModel]:
        """`root` and the kernels it holds, dependencies first."""
        order: list[_KernelModel] = []
        state: dict[str, str] = {}

        def visit(name: str, node: ast.AST) -> None:
            if state.get(name) == "done":
                return
            if state.get(name) == "active":
                self.errors.add(RuleId.RECURSION, f"kernel `{name}` contains itself", node)
                return
            state[name] = "active"
            model = self.model(name)
            for f in model.fields.values():
                if f.kernel is not None:
                    visit(f.kernel, model.node)
            state[name] = "done"
            order.append(model)

        visit(root, self.classes[root])
        return order

    def check_recursion(self) -> None:
        state: dict[tuple[str, str], str] = {}

        def visit(key: tuple[str, str]) -> None:
            state[key] = "active"
            for callee in sorted(self.calls.get(key, ())):
                if state.get(callee) == "active":
                    node = self.free_nodes[callee[1]] if not callee[0] else self.model(callee[0]).methods[callee[1]]
                    self.errors.add(RuleId.RECURSION, f"`{callee[0]}.{callee[1]}` calls itself, directly or "
                                    "through other methods", node)  # fmt: skip
                elif callee not in state:
                    visit(callee)
            state[key] = "done"

        for key in sorted(self.calls):
            if key not in state:
                visit(key)


def _load_module(path: Path, errors: _Errors) -> tuple[ModuleType | None, ast.Module | None]:
    source = path.read_text()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.items.append(
            Diagnostic(RuleId.SYNTAX_ERROR, DiagnosticSeverity.ERROR, exc.msg, SourceSpan(exc.lineno or 1))
        )
        return None, None
    name = f"_cignal_kernel_{next(_module_ids)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered so string annotations (`from __future__ import annotations`) evaluate in the module's
    # globals; `compile_kernel` removes it again.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any failure in user code is reported, not raised
        del sys.modules[name]
        errors.add(RuleId.IMPORT_ERROR, f"importing the module failed: {type(exc).__name__}: {exc}")
        return None, tree
    return module, tree


_module_ids = itertools.count()


def _kernel_classes(module: ModuleType, tree: ast.Module) -> Iterator[tuple[str, ast.ClassDef]]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = getattr(module, node.name, None)
            if isinstance(cls, type) and issubclass(cls, Kernel) and cls is not Kernel:
                yield node.name, node


def compile_kernel(
    path: str | Path, kernel: str | None = None, target: Target | None = None, prefix: str = ""
) -> CKernel:
    """Compile kernel `kernel` from `path`. Raises `CompileError` with every diagnostic found.

    `prefix` starts every C identifier derived from a kernel name, so two generated kernels that hold
    sub-kernels of the same name can share a translation unit.
    """
    if prefix and not (prefix.isidentifier() and prefix.isascii()):
        raise ValueError(f"prefix {prefix!r} must be an ASCII C identifier")
    path = Path(path)
    errors = _Errors()
    module, tree = _load_module(path, errors)
    if module is None or tree is None or errors.failed:
        raise CompileError(errors.items, str(path))
    try:
        return _compile(module, tree, path, kernel, target or Target(), errors, prefix)
    finally:
        sys.modules.pop(module.__name__, None)


def _compile(
    module: ModuleType, tree: ast.Module, path: Path, kernel: str | None, target: Target, errors: _Errors, prefix: str
) -> CKernel:
    program = _Program(module, tree, target, errors)
    program.prefix = prefix
    candidates = [name for name in program.classes if name not in program.held_kernels()]
    if kernel is None and len(candidates) == 1:
        kernel = candidates[0]
    if kernel not in program.classes:
        found = ", ".join(program.classes) or "none"
        wanted = f"kernel `{kernel}` not found" if kernel else "several top-level kernels; choose one"
        errors.add(RuleId.KERNEL_NOT_FOUND, f"{wanted} (kernels in file: {found})", None,
                   "pass --kernel NAME" if program.classes else "derive a class from cignal.Kernel")  # fmt: skip
        raise CompileError(errors.items, str(path))
    assert kernel is not None
    program.exported = kernel
    result = _Generator(program, path).generate(kernel)
    if errors.failed:
        raise CompileError(errors.items, str(path))
    result.warnings = list(errors.items)
    return result


@dataclass
class _Generator:
    program: _Program
    path: Path
    # Each function's C lines, keyed by (kernel, method); None for init and reset, which are always emitted.
    functions: list[tuple[tuple[str, str] | None, list[str]]] = field(default_factory=list)
    helpers: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    # (kernel, method) -> (definition order, signature), for emitting prototypes only where needed.
    defined: dict[tuple[str, str], tuple[int, str]] = field(default_factory=dict)
    reachable: set[tuple[str, str]] = field(default_factory=set)
    smoothed_reads: dict[tuple[str, str], list[ast.AST]] = field(default_factory=dict)

    def generate(self, name: str) -> CKernel:
        program = self.program
        target = program.target
        sample = target.ctype("f32")
        models = program.ordered(name)
        root = program.model(name)
        tick = root.signatures.get("tick")
        if "tick" not in root.methods:
            program.errors.add(RuleId.MISSING_TICK, f"kernel `{name}` has no `tick` method", root.node)
        n_inputs = len(tick.params) if tick else 0
        outputs = tick.returns if tick is not None and isinstance(tick.returns, tuple) else ("f32",)
        program.lower_free = self._free_function
        for model in models:
            self._kernel_functions(model)
        program.check_recursion()
        events = [(m[3:], s) for m, s in root.signatures.items() if s.event]
        for e, s in events:
            if e in {n for n, _, _ in root.slots}:
                program.errors.add(RuleId.BAD_SIGNATURE, f"event `{e}` has the host name of a parameter", s.node,
                                   "Pd sends both as `<name> args...`; rename one")  # fmt: skip
        roots = {(name, "tick"), (name, "control"), *((m.name, "init") for m in models)}
        self.reachable = self._reachable(roots | {(name, f"on_{e}") for e, _ in events})
        for caller in sorted(self._reachable({(name, "control")})):
            for node in self.smoothed_reads.get(caller, []):
                program.errors.add(RuleId.BAD_PARAM, f"`{ast.unparse(node)}` is smoothed per sample, so `control` "
                                   "may not read it, directly or through a method", node,
                                   "read it in `tick`, or drop `smooth=`")  # fmt: skip

        structs = []
        for model in models:
            members = "\n".join(self._member_decl(f) for f in model.fields.values())
            structs.append(f"typedef struct {model.cname} {{\n{members}\n}} {model.cname};\n")
        cname = root.cname
        params = root.params
        slots = root.slots
        setters = "".join(
            f"void {cname}_set_{p.name}({cname} *self, {'int element, ' if p.array else ''}{sample} value);\n"
            for p in params
        )
        event_decls = "".join(
            f"void {cname}_on_{e}({cname} *self"
            + "".join(f", {target.ctype(t)} {c_name(a)}" for a, t in s.params.items())
            + ");\n"
            for e, s in events
        )
        process_sig = f"void {cname}_process({cname} *self, const {sample} *const *in, {sample} *const *out, int n)"
        init_sig = f"void {cname}_init({cname} *self, {sample} sample_rate)"
        guard = f"CIGNAL_{(self.program.prefix + self.path.stem).upper()}_H"
        header = f"""/* Generated by cignal from {self.path.name}. Do not edit. */
#ifndef {guard}
#define {guard}

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#ifndef CIGNAL_INFO_TYPES
#define CIGNAL_INFO_TYPES
typedef struct {{ const char *name; double lo, hi, def; }} cignal_param_info;
typedef struct {{ const char *name; int n_args; }} cignal_event_info;
typedef struct {{
    const char *name;
    int n_inputs, n_outputs, n_params;
    const cignal_param_info *params;
    size_t size;
    int n_events;
    const cignal_event_info *events;
}} cignal_kernel_info;
#endif

{chr(10).join(structs)}
enum {{ {cname}_N_INPUTS = {n_inputs}, {cname}_N_OUTPUTS = {len(outputs)}, {cname}_N_PARAMS = {len(slots)},
       {cname}_N_EVENTS = {len(events)} }};

extern const cignal_kernel_info {cname}_info;

{init_sig};
void {cname}_reset({cname} *self);
{process_sig};
void {cname}_set_param({cname} *self, int index, {sample} value);
{setters}void {cname}_event({cname} *self, int index, const {sample} *args);
{event_decls}
#ifdef __cplusplus
}}
#endif

#endif
"""
        tuples = [
            f"typedef struct {{ {' '.join(f'{target.ctype(e)} v{i};' for i, e in enumerate(t))} }} {target.ctype(t)};"
            for t in sorted(program.tuples | ({outputs} if len(outputs) > 1 else set()))
        ]
        if any(t == "i32" for _, s in events for t in s.params.values()):
            program.helpers.add("to_i32")
        used = set(program.helpers).union(*(self.helpers[k] for k in self.reachable if k in self.helpers))
        used |= {dep for h in used for dep in _HELPER_DEPS.get(h, [])}
        ordered_helpers = sorted(used, key=lambda h: (h not in {d for v in _HELPER_DEPS.values() for d in v}, h))
        codes = [
            program.table_code(h[6:]) if h.startswith("table_")
            else program.const_code(h[6:]) if h.startswith("const_")
            else _HELPERS[h]
            for h in ordered_helpers
        ]  # fmt: skip
        helpers = ["\n\n".join(codes)] if codes else []
        for h in used:
            program.math_symbols |= _HELPER_SYMBOLS.get(h, set())
        protos = self.prototypes()
        source = [f"/* Generated by cignal from {self.path.name}. Do not edit. */", f'#include "{name_h(self.path)}"',
                  "", "#include <math.h>", ""]  # fmt: skip
        for section in (tuples, helpers, protos):
            source += [*section, ""] if section else []
        # Methods nothing calls are checked but not emitted: -Wunused-function would reject them.
        for key, lines in self.functions:
            if key is None or key in self.reachable:
                source += lines
        source += self._public(root, n_inputs, outputs, init_sig, process_sig, events)
        info = [(n, *(p.param or (0.0, 0.0)), float(p.default)) for n, p, _ in slots]
        return CKernel(name, name_h(self.path), header, "\n".join(source), n_inputs, len(outputs),
                       [(n, lo, hi, d) for n, lo, hi, d in info], sample,
                       frozenset(program.math_symbols), c_name=root.cname,
                       events=[(e, tuple(s.params.values())) for e, s in events])  # fmt: skip

    def _kernel_functions(self, model: _KernelModel) -> None:
        target = self.program.target
        for method, signature in model.signatures.items():
            lowerer = _FunctionLowerer(model, method, signature)
            body = lowerer.lower_body(signature.node.body)
            args = "".join(f", {target.ctype(t)} {c_name(p)}" for p, t in signature.params.items())
            returns = target.ctype(signature.returns) if signature.returns is not None else "void"
            if isinstance(signature.returns, tuple):
                self.program.tuples.add(signature.returns)
            fname = f"{model.cname}_{method}_body" if method == "init" else f"{model.cname}_{method}"
            storage = "" if signature.event and model.exported else "static inline "
            sig = f"{storage}{returns} {fname}({model.cname} *self{args})"
            self.defined[(model.name, method)] = (len(self.defined), sig)
            self.helpers[(model.name, method)] = lowerer.helpers
            self.smoothed_reads[(model.name, method)] = lowerer.smoothed_reads
            self.functions.append(((model.name, method), [sig + " {", *body, "}", ""]))
        # Init: field defaults and the sample rate, then sub-kernels in field order, then the kernel's `init`.
        # Reset: the same without the sample rate, and keeping the exported kernel's parameters.
        # Sub-kernel init and reset are defined before every kernel that holds them: no prototypes needed.
        storage = "" if model.exported else "static "
        subs = [f for f in model.fields.values() if f.kernel is not None]
        body = [f"    {model.cname}_init_body(self);"] if "init" in model.signatures else []
        sample = target.ctype("f32")
        init = [*self._state(model, keep_params=False), "    self->sample_rate = sample_rate;"]
        for p in model.params:
            if p.smooth > 0:
                self.program.math_symbols.add("exp")
                ctype = target.ctype(p.array[0] if p.array else p.type or "")
                loop, k = (f"for (int32_t k = 0; k < {p.array[1]}; k++) ", "[k]") if p.array else ("", "")
                init += [f"    {loop}self->cignal_target_{p.name}{k} = self->{c_name(p.name)}{k};",
                         f"    self->cignal_coef_{p.name} = ({ctype})(1.0 - exp(-1.0 / ({p.smooth!r} * "
                         "(double)sample_rate)));"]  # fmt: skip
        init += [self._each(f, f"{self._cname(f)}_init(&self->{c_name(f.name)}{{k}}, sample_rate);") for f in subs]
        reset = self._state(model, keep_params=model.exported)
        reset += [self._each(f, f"{self._cname(f)}_reset(&self->{c_name(f.name)}{{k}});") for f in subs]
        self.functions.append((None, [f"{storage}void {model.cname}_init({model.cname} *self, {sample} sample_rate) {{",
                                      *init, *body, "}", ""]))  # fmt: skip
        self.functions.append((None, [f"{storage}void {model.cname}_reset({model.cname} *self) {{",
                                      *(reset + body or ["    (void)self;"]), "}", ""]))  # fmt: skip

    def _free_function(self, name: str, signature: Signature) -> None:
        """Lower a module-level function called from kernel code, as `static inline cignal_fn_<name>`."""
        target = self.program.target
        scope = self.program.model(self.program.exported or "")
        lowerer = _FunctionLowerer(scope, name, signature, free=True)
        body = [line for line in lowerer.lower_body(signature.node.body) if line != "    (void)self;"]
        args = ", ".join(f"{target.ctype(t)} {c_name(p)}" for p, t in signature.params.items()) or "void"
        returns = target.ctype(signature.returns) if signature.returns is not None else "void"
        if isinstance(signature.returns, tuple):
            self.program.tuples.add(signature.returns)
        sig = f"static inline {returns} cignal_fn_{name}({args})"
        self.defined[("", name)] = (len(self.defined), sig)
        self.helpers[("", name)] = lowerer.helpers
        self.functions.append((("", name), [sig + " {", *body, "}", ""]))

    @staticmethod
    def _each(f: Field, statement: str) -> str:
        """`statement` for a sub-kernel field, looped over the elements of an `Array` of them."""
        if f.count:
            return f"    for (int32_t k = 0; k < {f.count}; k++) " + statement.format(k="[k]")
        return "    " + statement.format(k="")

    def _cname(self, f: Field) -> str:
        assert f.kernel is not None
        return self.program.model(f.kernel).cname

    def _member_decl(self, f: Field) -> str:
        target = self.program.target
        if f.kernel is not None:
            return f"    {self._cname(f)} {c_name(f.name)}{f.c_suffix};"
        if f.delay is not None:
            ctype, size = target.ctype(f.delay[0]), f.delay[1]
            return f"    struct {{ {ctype} buf[{size}]; int32_t pos; }} {c_name(f.name)}{f.c_suffix};"
        ctype = target.ctype(f.array[0] if f.array else f.type or "")
        suffix = f"[{f.array[1]}]" if f.array else ""
        member = f"    {ctype} {c_name(f.name)}{suffix};"
        if f.smooth > 0 and self.program.model(self.program.exported or "").fields.get(f.name) is f:
            member += f"\n    {ctype} cignal_target_{f.name}{suffix};\n    {ctype} cignal_coef_{f.name};"
        return member

    def _state(self, model: _KernelModel, keep_params: bool) -> list[str]:
        """Assignments restoring every field except the sample rate and sub-kernels to its default."""
        target = self.program.target
        lines = []
        for f in model.fields.values():
            name = f"self->{c_name(f.name)}"
            if f.array is not None:
                if keep_params and f.param is not None:
                    continue
                fill = f.default if f.param is not None else False if f.array[0] == "bool" else 0
                lines.append(f"    for (int32_t k = 0; k < {f.array[1]}; k++) {name}[k] = "
                             f"{self._literal(fill, f.array[0])};")  # fmt: skip
            elif f.delay is not None:
                ctype = target.ctype(f.delay[0])
                self.program.helpers.add(f"delay_clear_{ctype}")
                if f.count:
                    clear = f"cignal_delay_clear_{ctype}({name}[k].buf, {f.delay[1]}, &{name}[k].pos);"
                    lines.append(f"    for (int32_t k = 0; k < {f.count}; k++) {clear}")
                else:
                    lines.append(f"    cignal_delay_clear_{ctype}({name}.buf, {f.delay[1]}, &{name}.pos);")
            elif f.is_scalar and f.name != "sample_rate" and not (keep_params and f.param is not None):
                lines.append(f"    {name} = {self._literal(f.default, f.type or '')};")
        return lines

    def _reachable(self, roots: set[tuple[str, str]]) -> set[tuple[str, str]]:
        """Methods reachable from `roots` through the call graph."""
        seen, stack = set(), [r for r in roots if r in self.defined]
        while stack:
            key = stack.pop()
            if key not in seen:
                seen.add(key)
                stack += [c for c in self.program.calls.get(key, ()) if c in self.defined]
        return seen

    def prototypes(self) -> list[str]:
        """Prototypes for methods called before their definition. Recursion is rejected, so no cycles."""
        needed = {
            callee
            for caller, callees in self.program.calls.items()
            for callee in callees
            if caller in self.reachable and callee in self.defined and self.defined[callee][0] > self.defined[caller][0]
        }
        return [self.defined[key][1] + ";" for key in sorted(needed, key=lambda k: self.defined[k][0])]

    def _public(
        self,
        root: _KernelModel,
        n_inputs: int,
        outputs: tuple[str, ...],
        init_sig: str,
        process_sig: str,
        events: list[tuple[str, Signature]],
    ) -> list[str]:
        name = root.cname
        sample = self.program.target.ctype("f32")
        # `restrict` on the definition only: the header stays valid C++, and top-level parameter qualifiers do
        # not affect compatibility. It lets the compiler keep state in registers across the sample loop;
        # without it, every write to `out` might alias a field (measured: biquad 1.7x slower).
        lines = [process_sig.replace(f"({name} *self", f"({name} *restrict self") + " {"]
        lines += ["    (void)in;"] if n_inputs == 0 else []
        lines += [f"    {name}_control(self);"] if "control" in root.signatures else []
        args = "".join(f", in{k}" for k in range(n_inputs))
        lines += ["    for (int i = 0; i < n; i++) {"]
        for p in root.params:
            if p.smooth > 0:  # kernel.smooth_step
                t = p.array[0] if p.array else p.type or ""
                ctype, snap = self.program.target.ctype(t), self._literal(SMOOTH_SNAP, t)
                k = "[k]" if p.array else ""
                v, target, coef = (
                    f"self->{c_name(p.name)}{k}",
                    f"self->cignal_target_{p.name}{k}",
                    f"self->cignal_coef_{p.name}",
                )
                # When value == target the full step yields target too (signed zeros included), so the
                # branch skips the work once a glide has finished.
                step = [f"if ({v} == {target}) {{",
                        f"    {v} = {target};",
                        "} else {",
                        f"    const {ctype} d = {target} - {v};",
                        f"    const {ctype} stepped = {v} + {coef} * d;",
                        f"    {v} = stepped == {v} || (d > -{snap} && d < {snap}) ? {target} : stepped;",
                        "}"]  # fmt: skip
                if p.array:
                    step = [f"for (int32_t k = 0; k < {p.array[1]}; k++) {{", *("    " + s for s in step), "}"]
                lines += ["        " + s for s in step]
        lines += [f"        const {sample} in{k} = in[{k}][i];" for k in range(n_inputs)]
        if len(outputs) == 1:
            lines.append(f"        out[0][i] = {name}_tick(self{args});")
        else:
            lines.append(f"        const {self.program.target.ctype(outputs)} r = {name}_tick(self{args});")
            lines += [f"        out[{k}][i] = r.v{k};" for k in range(len(outputs))]
        lines += ["    }", "}", ""]
        slots = root.slots
        lines += [f"void {name}_set_param({name} *self, int index, {sample} value) {{"]
        if slots:
            lines.append("    switch (index) {")
            for index, (_, p, element) in enumerate(slots):
                assert p.param is not None
                t = p.array[0] if p.array else p.type or ""
                lo, hi = (self._literal(v, t) for v in p.param)
                v = "value" if t == "f32" else f"({self.program.target.ctype(t)})value"
                clamp = f"!({v} >= {lo}) ? {lo} : ({v} > {hi} ? {hi} : {v})"  # NaN to lo
                dest = f"cignal_target_{p.name}" if p.smooth > 0 else c_name(p.name)
                dest += "" if element is None else f"[{element}]"
                lines.append(f"    case {index}: self->{dest} = {clamp}; break;")
            lines += ["    default: break;", "    }"]
        else:
            lines += ["    (void)self;", "    (void)index;", "    (void)value;"]
        lines += ["}", ""]
        for index, (_, p, element) in enumerate(slots):
            if element is None:
                lines += [f"void {name}_set_{p.name}({name} *self, {sample} value) {{",
                          f"    {name}_set_param(self, {index}, value);", "}", ""]  # fmt: skip
            elif element == 0:
                assert p.array is not None
                lines += [f"void {name}_set_{p.name}({name} *self, int element, {sample} value) {{",
                          f"    if (element >= 0 && element < {p.array[1]}) {name}_set_param(self, {index} + element, "
                          "value);", "}", ""]  # fmt: skip
        table = "NULL"
        if slots:
            entries = ", ".join(f'{{"{n}", {p.param[0]!r}, {p.param[1]!r}, {float(p.default)!r}}}'
                                for n, p, _ in slots if p.param is not None)  # fmt: skip
            lines += [f"static const cignal_param_info {name}_params[] = {{{entries}}};", ""]
            table = f"{name}_params"
        lines += [f"void {name}_event({name} *self, int index, const {sample} *args) {{"]
        if events:
            lines.append("    switch (index) {")
            for index, (e, s) in enumerate(events):
                args = "".join(f", {self._event_arg(t, f'args[{k}]')}" for k, t in enumerate(s.params.values()))
                lines.append(f"    case {index}: {name}_on_{e}(self{args}); break;")
            lines += ["    default: break;", "    }"]
            lines += [] if any(s.params for _, s in events) else ["    (void)args;"]
        else:
            lines += ["    (void)self;", "    (void)index;", "    (void)args;"]
        lines += ["}", ""]
        event_table = "NULL"
        if events:
            entries = ", ".join(f'{{"{e}", {len(s.params)}}}' for e, s in events)
            lines += [f"static const cignal_event_info {name}_events[] = {{{entries}}};", ""]
            event_table = f"{name}_events"
        fields = (f'"{root.name}", {n_inputs}, {len(outputs)}, {len(slots)}, {table}, sizeof({name}), '
                  f"{len(events)}, {event_table}")  # fmt: skip
        lines += [f"const cignal_kernel_info {name}_info = {{{fields}}};", ""]
        return lines

    def _event_arg(self, t: str, value: str) -> str:
        """A host's `sample` value as an event argument of type `t` (kernel.Kernel.event)."""
        if t == "i32":
            return f"cignal_to_i32((double){value})"
        if t == "bool":
            return f"{value} != 0"
        return value if t == "f32" else f"({self.program.target.ctype(t)}){value}"

    def _literal(self, value: Number, t: str) -> str:
        try:
            return _literal(value, t, self.program.target, self.program.errors, self.program.classes[
                self.program.exported or ""])  # fmt: skip
        except _Abort:
            return "0"


def name_h(path: Path) -> str:
    return f"{path.stem}.h"


__all__ = ["PD_SELECTORS", "CKernel", "Field", "Target", "c_name", "compile_kernel"]
