"""Generated kernels compile cleanly under strict flags and reference only libm (spec 9.3)."""

import subprocess
from pathlib import Path

import pytest
from support import CASES, FIXTURES, Case

from cignal.build import compile_object, write_kernel
from cignal.compiler import compile_kernel

# Calls compilers substitute for source code. Apple clang fuses sinf/cosf pairs into libm's sincos, and
# turns fill loops into bzero/memset, or memset_pattern16 for a repeated non-zero value. None of these
# allocate, lock or do I/O.
COMPILER_LIBM = {"__sincosf_stret", "__sincos_stret", "sincosf", "sincos"}
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
