"""The emitted C text: parentheses, declarations, prototypes. Behaviour is covered by test_differential."""

import subprocess
import textwrap
from pathlib import Path

from cignal.build import STRICT_FLAGS, cc, compile_object, write_kernel
from cignal.compiler import compile_kernel

FIXTURES = Path(__file__).parent / "fixtures"

SOURCE = """
from cignal import Kernel, f32

class K(Kernel):
    g: f32 = 0.0

    def tick(self, a: f32, b: f32, c: f32) -> f32:
        left = a - b - c
        right = a - (b - c)
        product = (a + b) * c
        plain = a * b + c
        divided = a / (b * c)
        neg = -2.0 * a
        if a > b:
            branch = 1.0
        else:
            branch = 2.0
        return self.later(self.pure(left + right + product + plain + divided + neg + branch))

    def later(self, v: f32) -> f32:
        return v * self.g

    def pure(self, v: f32) -> f32:
        return v

    def unused(self, v: f32) -> f32:
        return v
"""


def _source(tmp_path: Path) -> str:
    path = tmp_path / "k.py"
    path.write_text(textwrap.dedent(SOURCE))
    return compile_kernel(path).source


def test_parentheses_follow_c_precedence_and_keep_evaluation_order(tmp_path: Path) -> None:
    c = _source(tmp_path)
    assert "float left = a - b - c;" in c
    assert "float right = a - (b - c);" in c
    assert "float product = (a + b) * c;" in c
    assert "float plain = a * b + c;" in c
    assert "float divided = a / (b * c);" in c
    assert "float neg = -2.0f * a;" in c


def test_top_level_locals_are_declared_inline_and_branch_locals_hoisted(tmp_path: Path) -> None:
    c = _source(tmp_path)
    tick = c[c.index("K_tick(K *self") :]
    assert tick.index("float branch = 0.0f;") < tick.index("float left =")
    assert "        branch = 1.0f;" in tick


def test_prototype_only_for_a_method_called_before_its_definition(tmp_path: Path) -> None:
    c = _source(tmp_path)
    assert "static inline float K_later(K *self, float v);" in c
    assert "static inline float K_tick(K *self, float a, float b, float c);" not in c


def test_void_self_only_where_self_is_unused(tmp_path: Path) -> None:
    c = _source(tmp_path)
    pure = c[c.index("K_pure(K *self, float v) {") :]
    assert pure.split("\n")[1].strip() == "(void)self;"
    methods = c[: c.index("void K_init(")]  # user methods; the generated stubs follow
    assert methods.count("(void)self;") == 1


def test_uncalled_method_is_checked_but_not_emitted(tmp_path: Path) -> None:
    """An unused static function would fail -Wunused-function."""
    assert "K_unused" not in _source(tmp_path)


def test_deferred_annotations_generate_the_same_c(tmp_path: Path) -> None:
    """On Python 3.12 and 3.13, `from __future__ import annotations` is how a kernel names a later class."""
    source = (Path(__file__).parent / "fixtures" / "echo.py").read_text()
    plain, deferred = tmp_path / "echo.py", tmp_path / "deferred" / "echo.py"
    deferred.parent.mkdir()
    plain.write_text(source)
    deferred.write_text("from __future__ import annotations\n" + source)
    assert compile_kernel(plain).source == compile_kernel(deferred).source


def _two_kernels_in_one_program(tmp_path: Path, prefixes: tuple[str, str]) -> subprocess.CompletedProcess[str]:
    """Two builds of the lowpass, which holds a `Section` sub-kernel, included and linked together."""
    objects = []
    for directory, prefix in zip("ab", prefixes, strict=True):
        ck = compile_kernel(FIXTURES / "lowpass.py", prefix=prefix)
        _, source = write_kernel(ck, tmp_path / directory)
        objects.append(compile_object(source))
    a, b = (prefix + "Lowpass" for prefix in prefixes)
    main = tmp_path / "main.c"
    main.write_text(f"""#include "a/lowpass.h"
#include "b/lowpass.h"
int main(void) {{
    static {a} x;
    static {b} y;
    float s = 1.0f, t = 0.0f;
    const float *in[1] = {{&s}};
    float *out[1] = {{&t}};
    {a}_init(&x, 48000.0f);
    {b}_init(&y, 48000.0f);
    {a}_process(&x, in, out, 1);
    {b}_process(&y, in, out, 1);
    return t > 0.0f ? 0 : 1;
}}
""")
    command = [
        cc(),
        *STRICT_FLAGS,
        "-I",
        str(tmp_path),
        str(main),
        *map(str, objects),
        "-o",
        str(tmp_path / "m"),
        "-lm",
    ]
    built = subprocess.run(command, capture_output=True, text=True)
    return built if built.returncode else subprocess.run([str(tmp_path / "m")], capture_output=True, text=True)


def test_prefix_lets_two_kernels_share_a_program(tmp_path: Path) -> None:
    assert _two_kernels_in_one_program(tmp_path, ("a_", "b_")).returncode == 0


def test_without_prefix_the_shared_sub_kernel_collides(tmp_path: Path) -> None:
    """The failure `--prefix` exists for. Both headers share an include guard, so the second is skipped and
    the linker finds both objects' `Lowpass_*` functions.
    """
    built = _two_kernels_in_one_program(tmp_path, ("", ""))
    assert built.returncode != 0
    assert "duplicate symbol" in built.stderr or "multiple definition" in built.stderr  # Apple ld, GNU ld


def test_builtin_min_yields_to_a_module_function_and_folds_constants(tmp_path: Path) -> None:
    path = tmp_path / "k.py"
    path.write_text(
        textwrap.dedent("""
        from cignal import Kernel, f32

        def min(a: f32, b: f32) -> f32:
            return a

        class K(Kernel):
            def tick(self, x: f32) -> f32:
                return min(x, 2.0) + max(1.0, 3.0, 2.0) * abs(-x)
    """)
    )
    c = compile_kernel(path).source
    assert "cignal_fn_min(x, 2.0f)" in c
    assert "3.0f * fabsf(-x)" in c
    assert "cignal_min_" not in c and "cignal_max_" not in c


def test_only_tuples_indexed_at_run_time_are_emitted(tmp_path: Path) -> None:
    path = tmp_path / "k.py"
    path.write_text(
        textwrap.dedent("""
        from cignal import Kernel, f32, i32

        USED = (1, 2, 3)
        FOLDED = (0.5, 0.25)
        UNUSED = (9.0, 8.0)
        FLAGS = (True, False)

        class K(Kernel):
            def tick(self, x: f32) -> f32:
                for n in range(3):
                    x += f32(USED[n]) * FOLDED[-1] if FLAGS[n % 2] else x
                return x
    """)
    )
    c = compile_kernel(path).source
    assert "static const int32_t cignal_const_USED[3] = {\n    1, 2, 3\n};" in c
    assert "static const bool cignal_const_FLAGS[2]" in c
    assert "0.25f" in c and "cignal_const_FOLDED" not in c and "UNUSED" not in c


def test_only_short_constant_loops_are_unrolled(tmp_path: Path) -> None:
    path = tmp_path / "k.py"
    path.write_text(
        textwrap.dedent("""
        from cignal import Kernel, f32, i32

        class K(Kernel):
            n: i32 = 3

            def tick(self, x: f32) -> f32:
                for a in range(8, 0, -2):
                    x += 1.0
                for b in range(1):
                    x += 1.0
                for c in range(17):
                    x += 1.0
                for d in range(self.n):
                    x += 1.0
                return x
    """)
    )
    c = compile_kernel(path).source
    assert [line for line in c.splitlines() if "CIGNAL_UNROLL(" in line and "for (" in line] == [
        "    CIGNAL_UNROLL(4) for (int32_t a = 8; a > 0; a -= 2) {"
    ]


def test_target_reads_the_setter_value_of_a_smoothed_param_and_the_field_otherwise(tmp_path: Path) -> None:
    path = tmp_path / "k.py"
    path.write_text(
        textwrap.dedent("""
        from cignal import Kernel, f32, param

        class K(Kernel):
            a: f32 = param(1.0, lo=0.0, hi=2.0, smooth=0.01)
            b: f32 = param(1.0, lo=0.0, hi=2.0)
            held: f32 = 0.0

            def control(self) -> None:
                self.held = self.target("a") + self.target("b")

            def tick(self, x: f32) -> f32:
                return x * self.held
    """)
    )
    assert "self->held = self->cignal_target_a + self->b;" in compile_kernel(path).source
