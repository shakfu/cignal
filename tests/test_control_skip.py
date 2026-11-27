"""`control` runs only after a change when that cannot alter output (effects.py). Fixtures cover the output."""

import ast
import subprocess
import textwrap
from pathlib import Path

import pytest

from cignal.build import STRICT_FLAGS, cc, compile_object, write_kernel
from cignal.compiler import compile_kernel
from cignal.diagnostics import CompileError

FIXTURES = Path(__file__).parent / "fixtures"

HEAD = """
from cignal import Array, Delay, Kernel, event, f32, i32, param

N = 4

class Sub(Kernel):
    a: f32 = 0.0
    b: f32 = 0.0
    state: f32 = 0.0

    def set(self, v: f32) -> None:
        self.a = v
        self.b = self.a * 2.0

    def peek(self) -> f32:
        return self.state

    def tick(self, x: f32) -> f32:
        self.state = self.state * 0.5 + x * self.b
        return self.state
"""

SKIPPABLE = {
    "params only": "self.g = self.p * 2.0",
    "reads its own earlier write": "self.g = self.p\n        self.h = self.g + 1.0",
    "write on both branches": "if self.p > 1.0:\n            self.g = 1.0\n        else:\n            self.g = 2.0\n"
    "        self.h = self.g",
    "every element written by a full loop": "for i in range(N):\n            self.xs[i] = self.p\n"
    "        self.h = self.xs[3]",
    "element read in the iteration that wrote it": "for i in range(N):\n            self.xs[i] = self.p\n"
    "            self.ys[i] = self.xs[i]",
    "a sub-kernel method's own writes": "self.sub.set(self.p)",
}

NOT_SKIPPABLE = {
    "reads a field tick writes": "self.g = self.t",
    "accumulates": "self.g += 1.0",
    "reads before writing": "self.h = self.g\n        self.g = self.p",
    "write on one branch only": "if self.p > 1.0:\n            self.g = 1.0\n        self.h = self.g",
    "partial loop": "for i in range(3):\n            self.xs[i] = self.p\n        self.h = self.xs[3]",
    "loop with break": "for i in range(N):\n            if self.p > 2.0:\n                break\n"
    "            self.xs[i] = self.p\n        self.h = self.xs[3]",
    "unknown index": "k = i32(self.p)\n        self.xs[k % N] = 1.0\n        self.h = self.xs[0]",
    "writes a field tick writes": "self.t = 0.0",
    "reads sub-kernel state tick writes": "self.g = self.sub.peek()",
    "delay line that tick writes": "self.g = self.line.read(2)",
}


def _kernel(tmp_path: Path, control: str) -> str:
    body = HEAD + textwrap.dedent("""
    class K(Kernel):
        p: f32 = param(1.0, lo=0.0, hi=4.0)
        g: f32 = 0.0
        h: f32 = 0.0
        t: f32 = 0.0
        xs: Array[f32, N]
        ys: Array[f32, N]
        sub: Sub
        line: Delay[f32, 8]

        def control(self) -> None:
            CONTROL

        def tick(self, x: f32) -> f32:
            self.t += x
            self.line.write(x)
            return self.sub.tick(x) + self.g + self.h + self.t + self.xs[0] + self.ys[0]
    """).replace("CONTROL", control)
    path = tmp_path / "k.py"
    path.write_text(body)
    return compile_kernel(path, "K").source


@pytest.mark.parametrize("control", SKIPPABLE.values(), ids=SKIPPABLE.keys())
def test_skipped_when_rerunning_changes_nothing(tmp_path: Path, control: str) -> None:
    source = _kernel(tmp_path, control)
    assert "    if (self->cignal_stale) {\n        self->cignal_stale = false;\n        K_control(self);" in source


@pytest.mark.parametrize("control", NOT_SKIPPABLE.values(), ids=NOT_SKIPPABLE.keys())
def test_runs_every_block_otherwise(tmp_path: Path, control: str) -> None:
    source = _kernel(tmp_path, control)
    assert "cignal_stale" not in source
    assert "\n    K_control(self);\n" in source


TRIGGERS = """
from cignal import Kernel, event, f32, param

class K(Kernel):
    level: f32 = param(1.0, lo=0.0, hi=10.0)
    scale: f32 = 1.0
    g: f32 = 0.0

    @event
    def on_scale(self, s: f32) -> None:
        self.scale = s

    def control(self) -> None:
        self.g = self.level * self.scale

    def tick(self) -> f32:
        return self.g
"""

DRIVER = r"""
#include <stdio.h>
#include "k.h"
static K k;
static void show(void) {
    float y = 0.0f;
    float *out[1] = {&y};
    K_process(&k, NULL, out, 1);
    printf("%g ", (double)y);
}
int main(void) {
    const float three = 3.0f;
    K_init(&k, 48000.0f);
    show(); show();
    K_event(&k, 0, &three); show();
    K_set_param(&k, 0, 2.0f); show();
    K_set_param_immediate(&k, 0, 4.0f); show();
    K_on_scale(&k, 0.5f); show();
    K_set_level(&k, 6.0f); show();
    K_reset(&k); show();
    return 0;
}
"""


def test_every_change_reruns_control(tmp_path: Path) -> None:
    path = tmp_path / "k.py"
    path.write_text(TRIGGERS)
    kernel = compile_kernel(path, "K")
    assert "cignal_stale" in kernel.source
    _, source = write_kernel(kernel, tmp_path)
    main = tmp_path / "main.c"
    main.write_text(DRIVER)
    exe = tmp_path / "m"
    command = [cc(), *STRICT_FLAGS, "-I", str(tmp_path), str(main), str(compile_object(source)), "-o", str(exe)]
    subprocess.run([*command, "-lm"], check=True)
    run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    # reset restores `scale` and keeps the parameter, as K_reset does without the skip
    assert run.stdout.split() == ["1", "1", "3", "6", "12", "2", "3", "6"]


BLOCK_DEPENDENT = {
    "reads a field tick writes": "self.t",
    "writes a field tick writes": "self.t",
    "reads sub-kernel state tick writes": "self.sub.state",
    "delay line that tick writes": "self.line",
}


@pytest.mark.parametrize("case", NOT_SKIPPABLE.keys())
def test_block_dependence_is_a_warning_at_the_access(tmp_path: Path, case: str) -> None:
    path = tmp_path / "k.py"
    _kernel(tmp_path, NOT_SKIPPABLE[case])
    kernel = compile_kernel(path, "K")  # a warning, not an error
    found = [w for w in kernel.warnings if w.rule_id == "CIGNAL.BLOCK_DEPENDENT"]
    if case not in BLOCK_DEPENDENT:  # only unproven, not shown to depend on blocks
        assert found == []
        return
    assert len(found) == 1
    assert f"`{BLOCK_DEPENDENT[case]}`" in found[0].message
    assert found[0].severity.value == "warning"
    assert found[0].span is not None
    line = path.read_text().splitlines()[found[0].span.line - 1]
    assert line.strip() == NOT_SKIPPABLE[case]


def test_no_fixture_depends_on_blocks() -> None:
    for fixture in sorted(FIXTURES.glob("*.py")):
        tree = ast.parse(fixture.read_text())
        for name in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]:
            try:
                kernel = compile_kernel(fixture, name)
            except CompileError:
                continue  # a sub-kernel the file does not export
            assert not [w for w in kernel.warnings if w.rule_id == "CIGNAL.BLOCK_DEPENDENT"], (fixture, name)
