"""Compiled kernels against the CPython reference (spec 9.1) and across block patterns (spec 9.2).

Each case applies its parameter changes mid-stream, so these also test `set_param` and `control`.
"""

import re
from pathlib import Path

import numpy as np
import pytest
from conftest import Driver
from support import CASES, FIXTURES, SAMPLE_RATE, Case, Change, _module, reference, run_driver, signals, snr_db

from cignal import Array, Kernel, f32, i32
from cignal.build import REFERENCE_FLAGS, build_driver
from cignal.compiler import Target, compile_kernel

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


def test_a_parameter_set_before_the_first_block_starts_there(driver: Driver) -> None:
    """Shims set parameters after `K_init`; gliding from the default would sweep every new instance."""
    case = next(c for c in CASES if c.kernel == "Glide")  # sets 440 at frame 0, then 0 at frame 2000
    for build, dtype in [("double", np.float64), ("float", np.float32)]:
        level = run_driver(driver(case, build), case, SIGNALS["impulse"], dtype)[0]
        assert level[0] == 440.0, build
        assert 0.0 < level[2000] < 440.0, build  # later changes still glide


def test_set_param_immediate_skips_the_glide(driver: Driver) -> None:
    case = next(c for c in CASES if c.kernel == "Glide")
    changes: tuple[Change, ...] = ((0, "level", 440.0), (1000, "now:level", -200.0), (2000, "level", 0.0))
    x = SIGNALS["impulse"]
    expected = reference(case, x, changes)
    assert expected[0][1000] == -200.0 and expected[1][1000] == -200.0  # value and target
    assert 0.0 > expected[0][2000] > -200.0
    assert np.array_equal(run_driver(driver(case, "double"), case, x, np.float64, changes=changes), expected)
    assert run_driver(driver(case, "float"), case, x, np.float32, changes=changes)[0][1000] == -200.0


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


def test_reference_names_a_complex_power() -> None:
    """CPython's `(-1.0) ** 0.5` is complex where C gives NaN; `float()` alone would not say why it failed."""

    class Root(Kernel):
        def tick(self, x: f32) -> f32:
            return x**0.5

    with pytest.raises(ValueError, match="fractional power is complex"):
        Root(SAMPLE_RATE).process([[-1.0]], 1)


class Counter(Kernel):
    n: i32 = 2**31 - 2
    steps: Array[i32, 2]

    def tick(self, x: f32) -> f32:
        self.n += 1
        return x


class Outer(Kernel):
    inner: Array[Counter, 2]

    def tick(self, x: f32) -> f32:
        return self.inner[1].tick(x)


@pytest.mark.parametrize(("kernel", "path"), [(Counter, "Counter.n"), (Outer, "Outer.inner[1].n")])
def test_reference_rejects_a_stored_i32_that_c_would_wrap(kernel: type[Kernel], path: str) -> None:
    """Spec 5.3: the first tick stores 2**31 - 1, which fits; the second stores 2**31."""
    k = kernel(SAMPLE_RATE)
    k.process([[0.0]], 1)
    with pytest.raises(OverflowError, match=rf"{re.escape(path)} = 2147483648 after `tick`"):
        k.process([[0.0]], 1)


@pytest.mark.parametrize(("kernel", "n_outputs"), [("Lowpass", 1), ("Glide", 2)])
def test_reference_returns_one_list_per_output_for_zero_frames(kernel: str, n_outputs: int) -> None:
    """C writes zero frames to each output channel; `[]` would break a caller that indexes the result."""
    case = next(c for c in CASES if c.kernel == kernel)
    k = getattr(_module(case.fixture), kernel)(SAMPLE_RATE)
    assert k.process([[] for _ in range(case.info.n_inputs)], 0) == [[] for _ in range(n_outputs)]


ARRAY_CASES = [c for c in CASES if re.search(r"\btypedef struct \w+_x\d+ \{", c.info.header)]


@pytest.mark.parametrize("case", ARRAY_CASES, ids=str)
def test_aos_layout_matches_cpython(tmp_path: Path, case: Case) -> None:
    """`--layout aos` stores `Array[K, N]` as N structs; it must compute what the default layout does."""
    compiled = compile_kernel(FIXTURES / case.fixture, case.kernel, Target(f32_ctype="double"), layout="aos")
    assert "_x" not in compiled.header.split("typedef struct")[-1]  # the exported struct holds plain arrays
    exe = build_driver(FIXTURES / case.fixture, tmp_path, case.kernel, extra_flags=REFERENCE_FLAGS, compiled=compiled)
    x = SIGNALS["noise"] if case.info.n_inputs else SIGNALS["impulse"]
    assert np.array_equal(run_driver(exe, case, x, np.float64), reference(case, x))
