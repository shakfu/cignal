"""The kernel fields each method reads and writes, to decide whether `control` may be skipped.

`control` runs at the start of every `K_process` call. Skipping it while nothing it depends on has
changed gives the same output when it is idempotent and reads nothing `tick` writes (spec 3.3).
The compiler then runs it only after init, reset, a parameter change or an event.

A path names a field from `self`: `(("sweeps", "i"), ("svf", None), ("k", None))` for
`self.sweeps[i].svf.k`. The second element of each step is the index: None without one, a constant's
text, a loop variable's name, `*` when unknown, or `all` after a `for i in range(N)` loop wrote
element `i` of an N-element field on every iteration. A read is covered when an earlier write in the
same method certainly wrote the same path; loop-variable indices match only within the loop body.
"""

import ast
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

Step = tuple[str, str | None]
Path = tuple[Step, ...]

UNKNOWN = "*"
ALL = "all"
_DELAY_READS = ("read", "read_frac")
_DELAY_WRITES = ("write", "clear")


class _Field(Protocol):
    @property
    def kernel(self) -> str | None: ...
    @property
    def delay(self) -> tuple[str, int] | None: ...
    @property
    def array(self) -> tuple[str, int] | None: ...
    @property
    def count(self) -> int: ...


class _Model(Protocol):
    @property
    def fields(self) -> Mapping[str, _Field]: ...
    @property
    def methods(self) -> Mapping[str, ast.FunctionDef]: ...
    def constant(self, name: str) -> bool | int | float | None: ...


@dataclass
class Effects:
    """What one method does to fields: reads not covered by its own writes, all writes, certain writes.

    `reads` and `writes` map each path to the first node in the method that reads or writes it.
    """

    reads: dict[Path, ast.AST] = field(default_factory=dict)
    writes: dict[Path, ast.AST] = field(default_factory=dict)
    defs: set[Path] = field(default_factory=set)  # written on every path to the end; constant indices only
    unknown: bool = False  # `self` used in a way this analysis does not follow


def _fields(path: Path) -> tuple[str, ...]:
    return tuple(name for name, _ in path)


def covered(path: Path, defs: set[Path]) -> bool:
    """Whether `defs` certainly wrote `path`."""
    return any(
        len(d) == len(path) and all(dn == n and di in (i, ALL) for (dn, di), (n, i) in zip(d, path, strict=True))
        for d in defs
    )


def _meets(x: Path, others: Iterable[Path]) -> bool:
    """Whether `x` and some path in `others` can name the same storage (indices ignored)."""
    fx = _fields(x)
    return any(fx[: len(fy)] == fy or fy[: len(fx)] == fx for fy in map(_fields, others))


def overlap(a: Iterable[Path], b: Iterable[Path]) -> bool:
    """Whether some path in `a` and some path in `b` can name the same storage (indices ignored)."""
    b = list(b)
    return any(_meets(x, b) for x in a)


def dotted(path: Path) -> str:
    """`self.sweeps[i].svf.k` for a path, with `[]` for any index."""
    return "self" + "".join(f".{name}" + ("[]" if index is not None else "") for name, index in path)


@dataclass
class ControlReport:
    skippable: bool  # running `control` again with nothing changed leaves the state as it was
    hazards: list[tuple[ast.AST, str]]  # where `control` reads or writes what `tick` writes


def analyse_control(model_of: Callable[[str], _Model], root: str) -> ControlReport:
    """Whether `root`'s `control` may be skipped, and where its result depends on block boundaries."""
    analysis = _Analysis(model_of)
    control, tick = analysis.method(root, "control"), analysis.method(root, "tick")
    if control.unknown or tick.unknown:
        return ControlReport(False, [])
    hazards = [
        (node, f"`control` reads `{dotted(path)}`, which `tick` writes")
        for path, node in control.reads.items()
        if _meets(path, tick.writes)
    ]
    hazards += [
        (node, f"`control` writes `{dotted(path)}`, which `tick` also writes")
        for path, node in control.writes.items()
        if _meets(path, tick.writes)
    ]
    idempotent = not overlap(control.reads, control.writes)
    return ControlReport(idempotent and not hazards, hazards)


class _Analysis:
    def __init__(self, model_of: Callable[[str], _Model]) -> None:
        self.model_of = model_of
        self.memo: dict[tuple[str, str], Effects] = {}

    def method(self, kernel: str, name: str) -> Effects:
        key = (kernel, name)
        if key not in self.memo:
            self.memo[key] = Effects(unknown=True)  # recursion is rejected elsewhere; stay conservative
            fn = self.model_of(kernel).methods.get(name)
            self.memo[key] = Effects() if fn is None else _Walker(self, kernel, fn).run()
        return self.memo[key]


class _Walker:
    def __init__(self, analysis: _Analysis, kernel: str, fn: ast.FunctionDef) -> None:
        self.analysis = analysis
        self.kernel = kernel
        self.fn = fn
        self.effects = Effects()
        self.loop_vars: list[str] = []

    def run(self) -> Effects:
        body = self.fn.body
        returns = [n for s in body for n in ast.walk(s) if isinstance(n, ast.Return)]
        exits_early = any(r is not body[-1] for r in returns)
        defs = self._block(body, set())
        # Loop-variable indices mean nothing to a caller; an early return leaves no write certain.
        self.effects.defs = set() if exits_early else {p for p in defs if self._constant(p)}
        reads: dict[Path, ast.AST] = {}
        for path, node in self.effects.reads.items():
            reads.setdefault(self._lift(path), node)
        self.effects.reads = reads
        return self.effects

    def _constant(self, path: Path) -> bool:
        return all(index is None or index == ALL or index.lstrip("-").isdigit() for _, index in path)

    def _lift(self, path: Path) -> Path:
        return tuple((n, i if self._constant(((n, i),)) else UNKNOWN) for n, i in path)

    # Statements: each returns the paths certainly written once it has run.

    def _block(self, body: list[ast.stmt], defs: set[Path]) -> set[Path]:
        defs = set(defs)
        for stmt in body:
            defs = self._stmt(stmt, defs)
        return defs

    def _stmt(self, stmt: ast.stmt, defs: set[Path]) -> set[Path]:
        if isinstance(stmt, ast.Assign):
            self._expr(stmt.value, defs, True)
            for target in stmt.targets:
                for t in target.elts if isinstance(target, ast.Tuple) else [target]:
                    self._store(t, defs)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._expr(stmt.value, defs, True)
            self._store(stmt.target, defs)
        elif isinstance(stmt, ast.AugAssign):
            self._expr(stmt.target, defs, True)
            self._expr(stmt.value, defs, True)
            self._store(stmt.target, defs)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._expr(stmt.value, defs, True)
        elif isinstance(stmt, ast.If):
            self._expr(stmt.test, defs, True)
            body = self._block(stmt.body, defs)
            orelse = self._block(stmt.orelse, defs)
            return defs | (body & orelse)
        elif isinstance(stmt, ast.For):
            self._expr(stmt.iter, defs, True)
            assert isinstance(stmt.target, ast.Name)  # the lowerer rejects other loops
            var = stmt.target.id
            self.loop_vars.append(var)
            written = self._block(stmt.body, defs) - defs  # the body may not run
            self.loop_vars.pop()
            jumps = any(isinstance(n, (ast.Break, ast.Continue)) for s in stmt.body for n in ast.walk(s))
            trips = self._trips(stmt.iter)
            for path in written if trips and not jumps else []:
                steps = [j for j, (_, index) in enumerate(path) if index == var]
                if len(steps) == 1 and self._length(path[: steps[0] + 1]) == trips:
                    j = steps[0]
                    defs.add(path[:j] + ((path[j][0], ALL),) + path[j + 1 :])
        elif isinstance(stmt, ast.While):
            self._expr(stmt.test, defs, True)
            self._block(stmt.body, defs)
        elif isinstance(stmt, ast.Expr):
            self._expr(stmt.value, defs, True)
        return defs

    def _store(self, target: ast.expr, defs: set[Path]) -> None:
        if isinstance(target, ast.Name):
            return
        found = self._chain(target, defs)
        if found is None:
            self.effects.unknown = True
            return
        path, _ = found
        self.effects.writes.setdefault(path, target)
        if all(index != UNKNOWN for _, index in path):
            defs.add(path)

    # Expressions. `certain` is False inside a branch of `a if c else b`, `and` and `or`.

    def _expr(self, node: ast.expr, defs: set[Path], certain: bool) -> None:
        if isinstance(node, ast.Call):
            self._call(node, defs, certain)
        elif isinstance(node, (ast.Attribute, ast.Subscript)) and self._rooted(node):
            found = self._chain(node, defs)
            if found is None:
                self.effects.unknown = True
            elif not covered(found[0], defs):
                self.effects.reads.setdefault(found[0], node)
        elif isinstance(node, ast.Name):
            if node.id == "self":
                self.effects.unknown = True
        elif isinstance(node, ast.IfExp):
            self._expr(node.test, defs, certain)
            self._expr(node.body, set(defs), False)
            self._expr(node.orelse, set(defs), False)
        elif isinstance(node, ast.BoolOp):
            self._expr(node.values[0], defs, certain)
            for value in node.values[1:]:
                self._expr(value, set(defs), False)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._expr(child, defs, certain)

    def _call(self, node: ast.Call, defs: set[Path], certain: bool) -> None:
        func = node.func
        for arg in node.args:
            self._expr(arg, defs, certain)
        if not (isinstance(func, ast.Attribute) and self._rooted(func.value)):
            if not isinstance(func, ast.Name):
                self._expr(func, defs, certain)
            return
        found = self._chain(func.value, defs)
        if found is None:
            self.effects.unknown = True
            return
        path, model = found
        if model is None:  # a delay line
            if func.attr in _DELAY_READS:
                self.effects.reads.setdefault(path, node)
            elif func.attr in _DELAY_WRITES:
                self.effects.writes.setdefault(path, node)
            else:
                self.effects.unknown = True
            return
        if func.attr == "target" and not path:
            return  # the last value set, which changes only through a setter
        callee = self.analysis.method(model, func.attr)
        if callee.unknown:
            self.effects.unknown = True
        for r in callee.reads:
            full = path + r
            if not covered(full, defs):
                self.effects.reads.setdefault(full, node)
        for w in callee.writes:
            self.effects.writes.setdefault(path + w, node)
        if certain and all(index != UNKNOWN for _, index in path):
            defs |= {path + d for d in callee.defs}

    def _rooted(self, node: ast.expr) -> bool:
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return isinstance(node, ast.Name) and node.id == "self"

    def _chain(self, node: ast.expr, defs: set[Path]) -> tuple[Path, str | None] | None:
        """`self.a[i].b`: its path, and the kernel it names (None for a delay line or a scalar)."""
        if isinstance(node, ast.Name) and node.id == "self":
            return (), self.kernel
        index: str | None = None
        if isinstance(node, ast.Subscript):
            self._expr(node.slice, defs, True)
            index = self._index(node.slice)
            node = node.value
        if not isinstance(node, ast.Attribute):
            return None
        base = self._chain(node.value, defs)
        if base is None:
            return None
        path, kernel = base
        if kernel is None:
            return None
        f = self.analysis.model_of(kernel).fields.get(node.attr)
        if f is None:
            return None
        return path + ((node.attr, index),), f.kernel

    def _trips(self, call: ast.expr) -> int | None:
        """N for `range(N)` or `range(0, N)` with N a literal or a module constant, else None."""
        if not isinstance(call, ast.Call) or not 1 <= len(call.args) <= 2:
            return None
        values = [self._int(a) for a in call.args]
        if len(values) == 2 and values[0] != 0:
            return None
        return values[-1]

    def _int(self, node: ast.expr) -> int | None:
        value = node.value if isinstance(node, ast.Constant) else None
        if isinstance(node, ast.Name):
            value = self.analysis.model_of(self.kernel).constant(node.id)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _length(self, path: Path) -> int | None:
        """The element count of the field `path` ends at."""
        kernel: str | None = self.kernel
        f = None
        for name, _ in path:
            if kernel is None:
                return None
            f = self.analysis.model_of(kernel).fields.get(name)
            if f is None:
                return None
            kernel = f.kernel
        if f is None:
            return None
        return f.count or (f.array[1] if f.array else None)

    def _index(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return str(node.value)
        if isinstance(node, ast.Name) and node.id in self.loop_vars:
            return node.id
        return UNKNOWN
