"""One minimal failing kernel per rule id (spec 9.4)."""

import json
import textwrap
from pathlib import Path

import pytest
from support import FIXTURES

from cignal.build import BuildError, render_shim, sc_class_file
from cignal.cli import main
from cignal.compiler import compile_kernel
from cignal.diagnostics import CompileError, RuleId

HEADER = "import math\nfrom cignal import Array, Delay, Kernel, event, f32, f64, i32, param, table\n\n"


def kernel(fields: str = "", tick: str = "return x", tick_sig: str = "(self, x: f32) -> f32") -> str:
    body = textwrap.indent(textwrap.dedent(tick).strip(), " " * 8)
    fields = textwrap.indent(textwrap.dedent(fields).strip(), " " * 4) + "\n" if fields.strip() else ""
    return HEADER + f"class K(Kernel):\n{fields}    def tick{tick_sig}:\n{body}\n"


TABLE = (
    HEADER
    + """@table(size={size}, lo={lo}, hi={hi})
def f(x: f32) -> {ret}:
    return {body}
class K(Kernel):
    def tick(self, x: f32) -> f32:
        return f(x)
"""
)
SMOOTH = "g: f32 = param(1.0, lo=0.0, hi=2.0, smooth=0.01)\n"
ARRAY_PARAM = "g: Array[f32, 2] = param(1.0, lo=0.0, hi=2.0)\n"

FREE_CALLER = "class K(Kernel):\n    def tick(self, x: f32) -> f32:\n        return f(x)\n"

# (rule id, source, line of the first diagnostic)
CASES = [
    (RuleId.SYNTAX_ERROR, HEADER + "class K(Kernel:\n", 4),
    (RuleId.IMPORT_ERROR, HEADER + "raise RuntimeError('boom')\n", None),
    (RuleId.KERNEL_NOT_FOUND, HEADER + "X = 1\n", None),
    (RuleId.UNSUPPORTED_SYNTAX, kernel(tick="assert x > 0.0\nreturn x"), 6),
    (RuleId.UNSUPPORTED_CALL, kernel(tick="print(x)\nreturn x"), 6),
    (RuleId.IMPLICIT_PRECISION, kernel("g: float = 1.0"), 5),
    (RuleId.UNKNOWN_TYPE, kernel("g: list[f32] = 1.0"), 5),
    (RuleId.UNKNOWN_NAME, kernel(tick="return x * missing"), 6),
    (RuleId.UNDECLARED_STATE, kernel(tick="self.q = x\nreturn x"), 6),
    (RuleId.MISSING_DEFAULT, kernel("g: f32"), 5),
    (RuleId.MISSING_TICK, HEADER + "class K(Kernel):\n    g: f32 = 0.0\n", 4),
    (RuleId.BAD_SIGNATURE, kernel(tick="pass", tick_sig="(self, x: f32) -> None"), 5),
    (RuleId.BAD_SIGNATURE, kernel(tick_sig="(self, x: f64) -> f64"), 5),  # only a sub-kernel's may
    (RuleId.MIXED_NUMERIC, kernel("n: i32 = 0", tick="return x + self.n"), 7),
    (RuleId.TYPE_MISMATCH, kernel(tick="if x:\n    return x\nreturn x"), 6),
    (RuleId.CONSTANT_RANGE, kernel(tick="return x * 1e39"), 6),
    (RuleId.CONSTANT_RANGE, kernel(tick="return x * 10**400"), 6),  # beyond double range
    (RuleId.BAD_PARAM, kernel("g: f32 = param(1.0, lo=2.0, hi=3.0)"), 5),
    (RuleId.RECURSION, kernel("def f(self, a: f32) -> f32:\n    return self.f(a)", tick="return self.f(x)"), 5),
    # Regression cases for restrictions that share a rule id with the cases above.
    (
        RuleId.RECURSION,
        # Python 3.12 and 3.13 evaluate `k: K` while K is still undefined, unless annotations are deferred.
        "from __future__ import annotations\n"
        + HEADER
        + "class K(Kernel):\n    k: K\n    def tick(self, x: f32) -> f32:\n"
        "        return x\n",
        5,
    ),
    (RuleId.UNSUPPORTED_SYNTAX, kernel("g: f32 = param(1.0, lo=0.0, hi=2.0)", tick="self.g = x\nreturn x"), 7),
    (RuleId.UNSUPPORTED_CALL, kernel(tick="self.init()\nreturn x"), 6),
    (
        RuleId.TYPE_MISMATCH,
        kernel("def f(self) -> tuple[f32, f32]:\n    return 1.0, 2.0", tick="a, b, c = self.f()\nreturn x"),
        8,
    ),
    (RuleId.BAD_SIGNATURE, kernel(tick_sig="(self, x: f32) -> tuple[f32, i32]"), 5),
    (RuleId.UNPROVEN_INDEX, kernel("a: Array[f32, 4]\nn: i32 = 0", tick="return self.a[self.n]"), 8),
    (RuleId.UNPROVEN_INDEX, kernel("a: Array[f32, 4]", tick="for i in range(5):\n    x += self.a[i]\nreturn x"), 8),
    (RuleId.UNPROVEN_INDEX, kernel("a: Array[f32, 4]", tick="return self.a[i32(x) % 4]"), 7),
    (RuleId.UNSUPPORTED_SYNTAX, kernel(tick="for i in range(4):\n    i = 2\nreturn x"), 7),
    # A local proves an index only if every assignment does; each of these has one that does not.
    (
        RuleId.UNPROVEN_INDEX,
        kernel("a: Array[f32, 4]", tick="k = 0\nfor i in range(4):\n    k = k + 1\nreturn self.a[k]"),
        10,
    ),
    (RuleId.UNPROVEN_INDEX, kernel("a: Array[f32, 4]\nn: i32 = 0", tick="k = self.n\nreturn self.a[k]"), 9),
    (RuleId.UNPROVEN_INDEX, kernel("a: Array[f32, 4]", tick="k = 0\nk += 1\nreturn self.a[k]"), 9),
    (RuleId.UNPROVEN_INDEX, kernel("a: Array[f32, 4]", tick="k = 0\nif x > 0.0:\n    k = 4\nreturn self.a[k]"), 10),
    (
        RuleId.UNPROVEN_INDEX,
        # The local `N` shadows the module constant in the loop bound.
        HEADER
        + "N = 2\n"
        + kernel("a: Array[f32, 4]", tick="N = i32(x)\nfor i in range(N):\n    k = i\nreturn self.a[k]")[len(HEADER) :],
        11,
    ),
    (RuleId.CONSTANT_RANGE, kernel("d: Delay[f32, 1]"), 5),
    (RuleId.UNKNOWN_TYPE, kernel("d: Delay[i32, 8]"), 5),
    (RuleId.TYPE_MISMATCH, kernel("d: Delay[f32, 8]", tick="return self.d"), 7),
    (RuleId.BAD_TABLE, TABLE.format(size=8, lo=1.0, hi=0.0, body="x", ret="f32"), 5),
    (RuleId.BAD_TABLE, TABLE.format(size=8, lo=0.0, hi=1.0, body="x", ret="f64"), 5),
    (RuleId.BAD_TABLE, TABLE.format(size=8, lo=0.0, hi=1.0, body="x * 1e308 * 10.0", ret="f32"), 5),
    (RuleId.BAD_PARAM, kernel(SMOOTH + "k: f32 = 0.0\ndef control(self) -> None:\n    self.k = self.g"), 8),
    (
        RuleId.BAD_PARAM,
        kernel(
            SMOOTH + "k: f32 = 0.0\ndef read(self) -> f32:\n    return self.g\ndef control(self) -> None:\n"
            "    self.k = self.read()"
        ),
        8,
    ),
    (RuleId.BAD_PARAM, kernel("g: f32 = param(1.0, lo=0.0, hi=2.0, smooth=-1.0)"), 5),
    (RuleId.BAD_PARAM, kernel("k: f32 = 0.0", tick='return self.target("k")'), 7),
    (RuleId.BAD_PARAM, kernel(SMOOTH, tick='return self.target("g" if x > 0.0 else "g")'), 7),
    (
        RuleId.BAD_PARAM,
        # A sub-kernel's parameters are plain fields, so nothing glides.
        HEADER + "class S(Kernel):\n    " + SMOOTH + "    def tick(self, x: f32) -> f32:\n"
        "        return self.target('g')\nclass K(Kernel):\n    s: S\n    def tick(self, x: f32) -> f32:\n"
        "        return self.s.tick(x)\n",
        7,
    ),
    (RuleId.BAD_SIGNATURE, kernel("def target(self) -> f32:\n    return 0.0"), 5),
    (RuleId.BAD_SIGNATURE, kernel("def set_param_immediate(self) -> f32:\n    return 0.0"), 5),
    (RuleId.UNSUPPORTED_SYNTAX, kernel(ARRAY_PARAM, tick="self.g[0] = x\nreturn x"), 7),
    (
        RuleId.BAD_PARAM,
        kernel(
            ARRAY_PARAM.replace(")", ", smooth=0.01)") + "k: f32 = 0.0\ndef control(self) -> None:\n"
            "    self.k = self.g[0]"
        ),
        8,
    ),
    (RuleId.BAD_PARAM, kernel(ARRAY_PARAM, tick='return self.target("g")'), 7),
    (RuleId.BAD_PARAM, kernel(ARRAY_PARAM + "g_1: f32 = 0.0"), 4),
    (RuleId.BAD_PARAM, kernel(ARRAY_PARAM.replace("f32", "i32")), 5),
    (RuleId.BAD_PARAM, kernel("g: bool = param(1.0, lo=0.0, hi=2.0)"), 5),
    (RuleId.BAD_SIGNATURE, kernel("def on_x(self) -> None:\n    pass"), 5),  # an event needs `@event`
    (RuleId.BAD_SIGNATURE, kernel("@event\ndef x(self) -> None:\n    pass"), 6),  # and the `on_` prefix
    (RuleId.BAD_SIGNATURE, kernel("@event\ndef on_reset(self) -> None:\n    pass"), 6),
    (RuleId.BAD_SIGNATURE, kernel("@event\ndef on_x(self) -> f32:\n    return 1.0"), 6),
    (RuleId.BAD_SIGNATURE, kernel("def event(self) -> None:\n    pass"), 5),
    (RuleId.BAD_SIGNATURE, kernel("g: f32 = param(1.0, lo=0.0, hi=2.0)\n@event\ndef on_g(self) -> None:\n    pass"), 7),
    (RuleId.UNSUPPORTED_SYNTAX, kernel("@staticmethod\ndef f(self) -> None:\n    pass"), 5),
    (RuleId.UNSUPPORTED_SYNTAX, kernel("n: i32 = 2", tick="return x + f32(self.n ** self.n)"), 7),
    (RuleId.UNSUPPORTED_SYNTAX, kernel("n: i32 = 2", tick="return x + f32(self.n ** -1)"), 7),
    (RuleId.TYPE_MISMATCH, kernel(tick="return f32(i32(x) & 3) + x * 2.0 % 1.0 + f32(x & 1.0)"), 6),
    (
        RuleId.UNSUPPORTED_SYNTAX,
        kernel("def f(self) -> f32:\n    return 1.0", tick="return x if 0.0 < self.f() < 1.0 else x"),
        8,
    ),
    (
        # Inside a conditional operand the order cannot be fixed by hoisting.
        RuleId.UNSUPPORTED_SYNTAX,
        kernel(
            "n: i32 = 0\ndef bump(self) -> i32:\n    self.n += 1\n    return self.n",
            tick="return x if self.bump() - self.n > 0 else x",
        ),
        10,
    ),
    (
        RuleId.RECURSION,
        HEADER + "def f(a: f32) -> f32:\n    return f(a)\n" + FREE_CALLER,
        4,
    ),
    (
        RuleId.UNKNOWN_NAME,
        HEADER + "def f(a: f32) -> f32:\n    return self.g\n" + FREE_CALLER,
        5,
    ),
    (RuleId.UNSUPPORTED_CALL, kernel(tick="return min(x)"), 6),
    (RuleId.UNSUPPORTED_CALL, kernel(tick="return abs(x, x)"), 6),
    (RuleId.TYPE_MISMATCH, kernel(tick="return x if min(x > 0.0, x < 1.0) else x"), 6),
    (RuleId.MIXED_NUMERIC, kernel("n: i32 = 0", tick="return max(x, self.n)"), 7),
    # Module-level tuples: indices are proven like Array indices; elements must fit the element type.
    (RuleId.UNPROVEN_INDEX, HEADER + "T = (1.0, 2.0)\n" + FREE_CALLER.replace("f(x)", "T[i32(x)]"), 7),
    (RuleId.CONSTANT_RANGE, HEADER + "T = (1.0, 2.0)\n" + FREE_CALLER.replace("f(x)", "T[2]"), 7),
    (
        RuleId.CONSTANT_RANGE,
        HEADER
        + "T = (2**40, 1)\n"
        + FREE_CALLER.replace("return f(x)", "for i in range(2):\n            x += f32(T[i])\n        return x"),
        8,
    ),
    (
        RuleId.UNSUPPORTED_SYNTAX,
        HEADER + "T = (1.0, 2.0)\n" + FREE_CALLER.replace("return f(x)", "T[0] = x\n        return x"),
        7,
    ),
    # A module that rebinds a built-in's name gets no built-in.
    (RuleId.UNSUPPORTED_CALL, HEADER + "abs = math.fabs\n" + FREE_CALLER.replace("f(x)", "abs(x)"), 7),
]

# Warnings do not stop the build; they are returned with the generated code.
WARNING_CASES = [
    (RuleId.UNBOUNDED_LOOP, kernel(tick="while x > 1.0:\n    x = x - 1.0\nreturn x"), 6),
    (
        RuleId.BAD_PARAM,
        HEADER + "class S(Kernel):\n    g: f32 = param(1.0, lo=0.0, hi=2.0)\n    def tick(self, x: f32) -> f32:\n"
        "        return x * self.g\nclass K(Kernel):\n    s: S\n    def tick(self, x: f32) -> f32:\n"
        "        return self.s.tick(x)\n",
        5,
    ),
    (
        RuleId.BLOCK_DEPENDENT,
        HEADER + "class K(Kernel):\n    s: f32 = 0.0\n    g: f32 = 0.0\n    def control(self) -> None:\n"
        "        self.g = self.s\n    def tick(self, x: f32) -> f32:\n        self.s += x\n        return self.g\n",
        8,
    ),
]


def test_every_rule_id_has_a_case() -> None:
    rule_ids = {v for k, v in vars(RuleId).items() if not k.startswith("_")}
    assert rule_ids == {rule for rule, _, _ in CASES + WARNING_CASES}


@pytest.mark.parametrize(("rule", "source", "line"), WARNING_CASES, ids=[c[0] for c in WARNING_CASES])
def test_warning(tmp_path: Path, rule: str, source: str, line: int) -> None:
    path = tmp_path / "k.py"
    path.write_text(source)
    [warning] = compile_kernel(path).warnings
    assert warning.rule_id == rule and not warning.is_error
    assert warning.span is not None and warning.span.line == line


@pytest.mark.parametrize(("rule", "source", "line"), CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_rule(tmp_path: Path, rule: str, source: str, line: int | None) -> None:
    path = tmp_path / "k.py"
    path.write_text(source)
    with pytest.raises(CompileError) as info:
        compile_kernel(path)
    first = next(d for d in info.value.diagnostics if d.is_error)  # warnings may come first
    assert first.rule_id == rule, info.value
    if line is not None:
        assert first.span is not None and first.span.line == line, info.value


def test_all_errors_are_reported_together(tmp_path: Path) -> None:
    path = tmp_path / "k.py"
    path.write_text(kernel("g: float = 1.0", tick="self.q = x\nreturn x * missing"))
    with pytest.raises(CompileError) as info:
        compile_kernel(path)
    rules = [d.rule_id for d in info.value.diagnostics]
    assert rules == [RuleId.IMPLICIT_PRECISION, RuleId.UNDECLARED_STATE, RuleId.UNKNOWN_NAME]


def test_check_cli_reports_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "k.py"
    path.write_text(kernel(tick="return x * missing"))
    assert main(["check", str(path), "--format", "json"]) == 1
    [diagnostic] = json.loads(capsys.readouterr().out)
    assert diagnostic["rule_id"] == RuleId.UNKNOWN_NAME
    assert diagnostic["span"]["line"] == 6


def test_check_cli_accepts_fixture() -> None:
    assert main(["check", str(FIXTURES / "biquad.py")]) == 0


def test_build_cli_writes_header_and_source(tmp_path: Path) -> None:
    assert main(["build", str(FIXTURES / "biquad.py"), "-o", str(tmp_path)]) == 0
    assert (tmp_path / "biquad.h").is_file() and (tmp_path / "biquad.c").is_file()


@pytest.mark.parametrize("flag", ["--pd-name=x~", "--pd-include=.", "--sc-name=X", "--sc-include=."])
def test_build_cli_rejects_a_host_flag_without_its_shim(tmp_path: Path, flag: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["build", str(FIXTURES / "biquad.py"), "-o", str(tmp_path), "--shim", "driver", flag])
    assert exc.value.code == 2


def test_build_cli_layout_reaches_the_code_and_the_manifest(tmp_path: Path) -> None:
    assert main(["build", str(FIXTURES / "nest.py"), "-o", str(tmp_path), "--layout", "aos", "--manifest"]) == 0
    assert json.loads((tmp_path / "nest.cignal.json").read_text())["layout"] == "aos"
    assert "    Lane lanes[3];" in (tmp_path / "nest.h").read_text()


def test_build_cli_writes_manifest(tmp_path: Path) -> None:
    """The manifest names the files written beside it and the C symbols the header declares."""
    assert main(["build", str(FIXTURES / "synth.py"), "-o", str(tmp_path), "--prefix", "fx_", "--manifest"]) == 0
    ck = compile_kernel(FIXTURES / "synth.py", prefix="fx_")
    data = json.loads((tmp_path / "synth.cignal.json").read_text())
    assert data["format"] == 1
    assert (data["name"], data["c_name"], data["sample_type"]) == ("Synth", "fx_Synth", "float")
    assert data["layout"] == "soa"
    assert (tmp_path / data["header"]).is_file() and (tmp_path / data["source"]).is_file()
    assert (data["n_inputs"], data["n_outputs"]) == (ck.n_inputs, ck.n_outputs)
    assert [(p["name"], p["lo"], p["hi"], p["default"]) for p in data["params"]] == [tuple(p) for p in ck.params]
    assert [(e["name"], tuple(e["args"])) for e in data["events"]] == ck.events
    header = (tmp_path / data["header"]).read_text()
    assert f"void {data['c_name']}_process(" in header and f"{data['c_name']}_info;" in header


@pytest.mark.parametrize("name", ["2pole~", "lowpass", "l\u00e9~", "a-b~"])
def test_pd_name_must_be_an_ascii_identifier_and_tilde(name: str) -> None:
    ck = compile_kernel(FIXTURES / "biquad.py")
    with pytest.raises(BuildError, match="Pd object name"):
        render_shim("pd", ck, pd_name=name)


@pytest.mark.parametrize("name", ["in0", "Gain", "var", "thisProcess"])
def test_sc_class_file_rejects_invalid_parameter_names(tmp_path: Path, name: str) -> None:
    path = tmp_path / "k.py"
    path.write_text(kernel(f"{name}: f32 = param(1.0, lo=0.0, hi=2.0)", tick=f"return x * self.{name}"))
    ck = compile_kernel(path)
    with pytest.raises(BuildError, match=repr(name)):
        sc_class_file(ck, "CignalK")
    assert "cutoff = " in sc_class_file(compile_kernel(FIXTURES / "lowpass.py"), "CignalLowpass")
