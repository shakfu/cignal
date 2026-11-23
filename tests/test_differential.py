"""Compiled kernels against the CPython reference (spec 9.1) and across block patterns (spec 9.2).

Each case applies its parameter changes mid-stream, so these also test `set_param` and `control`.
"""

import numpy as np
import pytest
from conftest import Driver
from support import CASES, Case, reference, run_driver, signals, snr_db

SIGNALS = signals()
BLOCK_PATTERNS = ["1", "7", "64", "128", "3,61,64,1", "4800"]


def _skip_duplicate_runs(case: Case, signal: str) -> None:
    if case.info.n_inputs == 0 and signal != "impulse":
        pytest.skip("a kernel without inputs gives the same output for every signal")


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("case", CASES, ids=str)
def test_double_build_matches_cpython(driver: Driver, case: Case, signal: str) -> None:
    """Bit-identical: same IEEE operations in the same order, same libm, no FMA contraction."""
    _skip_duplicate_runs(case, signal)
    x = SIGNALS[signal]
    expected = reference(case, x)
    actual = run_driver(driver(case, "double"), case, x, np.float64)
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), float(np.max(np.abs(actual - expected)))


@pytest.mark.parametrize("signal", SIGNALS)
@pytest.mark.parametrize("case", CASES, ids=str)
def test_float_build_snr(driver: Driver, case: Case, signal: str) -> None:
    _skip_duplicate_runs(case, signal)
    x = SIGNALS[signal]
    expected = run_driver(driver(case, "double"), case, x, np.float64)
    actual = run_driver(driver(case, "float"), case, x, np.float32)
    assert snr_db(expected, actual) >= case.min_snr_db


@pytest.mark.parametrize("case", CASES, ids=str)
def test_output_independent_of_block_pattern(driver: Driver, case: Case) -> None:
    """State must persist across `process` calls: any split of the input gives identical output."""
    x = SIGNALS["noise"]
    exe = driver(case, "float")
    outputs = {blocks: run_driver(exe, case, x, np.float32, blocks) for blocks in BLOCK_PATTERNS}
    for blocks, y in outputs.items():
        assert np.array_equal(y, outputs["4800"]), f"blocks={blocks}"


@pytest.mark.parametrize("case", [c for c in CASES if any(n == "reset" for _, n, _ in c.changes)], ids=str)
def test_reset_changes_output(case: Case) -> None:
    """Without this, a reset that did nothing in both Python and C would still match."""
    x = SIGNALS["noise"]
    without = tuple(change for change in case.changes if change[1] != "reset")
    assert not np.array_equal(reference(case, x), reference(case, x, without))


def test_float_smoothing_reaches_its_target_without_denormals(driver: Driver) -> None:
    """A float one-pole stalls short of its target once `coef * d` is below half an ulp; the snap prevents it.

    Near 440 the stall would leave the float build about 7e-4 short. Gliding to 0 must snap (at |d| < 1e-20,
    about 58 time constants from 440) before the value becomes denormal.
    """
    case = next(c for c in CASES if c.kernel == "Glide")
    y = run_driver(driver(case, "float"), case, SIGNALS["impulse"], np.float32)[0]
    assert y[1999] == 440.0
    assert y[-1] == 0.0
    denormal = (y != 0) & (np.abs(y) < np.finfo(np.float32).tiny)
    assert not denormal.any()
