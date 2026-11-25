"""Generated kernels compile cleanly under strict flags and reference only libm (spec 9.3)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from support import CASES, FIXTURES, Case

from cignal.build import BuildError, build_driver, compile_object, write_kernel
from cignal.compiler import compile_kernel

# Calls compilers substitute for source code. Apple clang fuses sinf/cosf pairs into libm's sincos,
# rewrites pow(2.0, x) as exp2(x), and turns fill loops into bzero/memset, or memset_pattern16 for a
# repeated non-zero value. None of these allocate, lock or do I/O.
COMPILER_LIBM = {"__sincosf_stret", "__sincos_stret", "sincosf", "sincos", "exp2", "exp2f"}
COMPILER_MEMORY = {"bzero", "__bzero", "memset", "memset_pattern16", "memcpy", "memmove"}


@pytest.mark.parametrize("case", CASES, ids=str)
def test_kernel_object_references_only_libm(tmp_path: Path, case: Case) -> None:
    ck = compile_kernel(FIXTURES / case.fixture, case.kernel)
    _, source = write_kernel(ck, tmp_path)
    obj = compile_object(source)  # STRICT_FLAGS include -Werror -Wconversion -Wdouble-promotion
    nm = subprocess.run(["nm", "-u", str(obj)], capture_output=True, text=True, check=True)
    # macOS prints `_sincosf`; GNU nm prints `U sincosf`.
    undefined = {line.split()[-1].removeprefix("_") for line in nm.stdout.splitlines() if line.strip()}
    allowed = ck.math_symbols | COMPILER_LIBM | COMPILER_MEMORY
    assert undefined <= allowed, undefined - allowed


CAST_KERNEL = """
from cignal import Kernel, f32, i32

class Cast(Kernel):
    def tick(self, x: f32) -> f32:
        return f32(i32(x))
"""


def test_float_to_i32_cast_is_defined_for_every_input(tmp_path: Path) -> None:
    """A C cast is undefined for NaN and out-of-range values (C99 6.3.1.4); `i32()` saturates, NaN to 0.

    On arm64 the raw cast happens to saturate too, so the sanitizer, not the output, catches a regression.
    """
    (tmp_path / "cast.py").write_text(CAST_KERNEL)
    sanitize = ("-fsanitize=float-cast-overflow", "-fno-sanitize-recover=all")
    try:
        exe = build_driver(tmp_path / "cast.py", tmp_path, extra_flags=sanitize)
    except BuildError as exc:
        pytest.skip(f"no float-cast-overflow sanitizer: {exc}")
    x = np.array([np.nan, np.inf, -np.inf, 3e9, -3e9, 2.7, -2.7], dtype=np.float32)
    proc = subprocess.run([str(exe), "48000", "64"], input=x.tobytes(), capture_output=True, check=True)
    assert proc.stderr == b""
    expected = [0.0, 2.0**31, -(2.0**31), 2.0**31, -(2.0**31), 2.0, -2.0]
    assert np.frombuffer(proc.stdout, dtype=np.float32).tolist() == expected
