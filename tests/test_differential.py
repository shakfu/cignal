"""Compiled kernels against the CPython reference (spec 9.1) and across block patterns (spec 9.2).

Each case applies its parameter changes mid-stream, so these also test `set_param` and `control`.
"""

import numpy as np
import pytest
from conftest import Driver
from support import CASES, SAMPLE_RATE, Case, Change, reference, run_driver, signals, snr_db

SIGNALS = signals()
BLOCK_PATTERNS = ["1", "7", "64", "128", "3,61,64,1", "4800"]


# A kernel without inputs gives the same output for every signal, so it runs on one.
CASE_SIGNALS = [
    pytest.param(case, signal, id=f"{case}-{signal}")
    for case in CASES
    for signal in (SIGNALS if case.info.n_inputs else ["impulse"])
]


@pytest.mark.parametrize(("case", "signal"), CASE_SIGNALS)
def test_double_build_matches_cpython(driver: Driver, case: Case, signal: str) -> None:
    """Bit-identical: same IEEE operations in the same order, same libm, no FMA contraction."""
    x = SIGNALS[signal]
    expected = reference(case, x)
    actual = run_driver(driver(case, "double"), case, x, np.float64)
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected), float(np.max(np.abs(actual - expected)))


@pytest.mark.parametrize(("case", "signal"), CASE_SIGNALS)
def test_float_build_snr(driver: Driver, case: Case, signal: str) -> None:
    x = SIGNALS[signal]
    expected = run_driver(driver(case, "double"), case, x, np.float64)
    actual = run_driver(driver(case, "float"), case, x, np.float32)
    assert snr_db(expected, actual) >= case.min_snr_db


TAIL_CASES = [c for c in CASES if c.tail_s > 0]


def tail_run(case: Case) -> tuple[np.ndarray, tuple[Change, ...]]:
    """A 0.5 s burst then silence, `case.tail_s` long, and every parameter set to its float32-rounded
    default at frame 0. Unrounded, a default like Echo's `time = 0.05` differs between builds by input."""
    rng = np.random.default_rng(1234)
    x = np.zeros(int(case.tail_s * SAMPLE_RATE))
    x[: SAMPLE_RATE // 2] = 0.5 * rng.uniform(-1, 1, SAMPLE_RATE // 2)
    defaults = tuple((0, name, float(np.float32(default))) for name, _, _, default in case.info.params)
    return x.astype(np.float32).astype(np.float64), defaults


@pytest.mark.parametrize("case", TAIL_CASES, ids=str)
def test_double_build_matches_cpython_long_tail(driver: Driver, case: Case) -> None:
    """Past FRAMES, so delay lines longer than FRAMES wrap; the float-vs-double test alone would pass a
    wrap bug both builds share."""
    x, changes = tail_run(case)
    expected = reference(case, x, changes)
    actual = run_driver(driver(case, "double"), case, x, np.float64, changes=changes)
    assert np.array_equal(actual, expected), float(np.max(np.abs(actual - expected)))


@pytest.mark.parametrize("case", TAIL_CASES, ids=str)
def test_float_build_snr_long_tail(driver: Driver, case: Case) -> None:
    """Every 1 s window meets the case threshold. Coefficient error compounds per pass through feedback:
    an f32 decay rate held Gigaverb above 100 dB over FRAMES but at 37.5 dB in its 10th second."""
    x, changes = tail_run(case)
    expected = run_driver(driver(case, "double"), case, x, np.float64, changes=changes)
    actual = run_driver(driver(case, "float"), case, x, np.float32, changes=changes)
    windows = range(0, len(x), SAMPLE_RATE)
    snrs = [snr_db(expected[:, i : i + SAMPLE_RATE], actual[:, i : i + SAMPLE_RATE]) for i in windows]
    assert min(snrs) >= case.min_snr_db, [round(s, 1) for s in snrs]


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


@pytest.mark.parametrize("kernel", ["Lowpass", "Drive"])  # Drive's parameter is smoothed
def test_nan_parameter_takes_lo(driver: Driver, kernel: str) -> None:
    """NaN fails both clamp comparisons; stored, it would stay in recursive state until `reset`."""
    case = next(c for c in CASES if c.kernel == kernel)
    x = SIGNALS["noise"]
    changes: tuple[Change, ...] = tuple((1000 + 500 * k, name, float("nan")) for k, (name, *_) in
                                        enumerate(case.info.params))  # fmt: skip
    expected = reference(case, x, changes)
    assert np.all(np.isfinite(expected))
    actual = run_driver(driver(case, "double"), case, x, np.float64, changes=changes)
    assert np.array_equal(actual, expected)
