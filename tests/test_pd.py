"""The Pd external, run headless, matches the standalone driver bit for bit."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import Driver
from support import (
    CASES,
    FIXTURES,
    FRAMES,
    SAMPLE_RATE,
    Case,
    inputs,
    pd_patch,
    pd_written,
    read_float_wav,
    run_driver,
    signals,
    write_float_wav,
)

from cignal.build import build_pd_external, find_pd_binary, find_pd_include

PD = find_pd_binary()
pytestmark = pytest.mark.skipif(PD is None or find_pd_include() is None, reason="Pd or m_pd.h not found")


@pytest.mark.parametrize("case", CASES, ids=str)
def test_pd_external_matches_driver(tmp_path: Path, driver: Driver, case: Case) -> None:
    """Parameters reach the kernel through creation arguments and through float inlets."""
    assert PD is not None
    x = signals()["noise"]
    info = case.info
    # A `cignal_` prefix avoids Pd built-ins such as `biquad~`, which would shadow the external.
    name = f"cignal_{case.kernel.lower()}~"
    build_pd_external(FIXTURES / case.fixture, tmp_path, case.kernel, pd_name=name)
    # The last value each case sets for every parameter, applied from the start.
    values = {param: value for _, param, value in case.changes if param != "reset"}
    settings = [(p[0], values.get(p[0], p[3])) for p in info.params]
    # Parameters before the last go in as creation arguments; the last goes to its float inlet.
    # A kernel without inputs takes its first parameter on the left inlet, so that one is sent there.
    args: list[float] = []
    inlet_values: dict[int, float] = {}
    if info.n_inputs == 0 and len(settings) == 1:
        inlet_values = {0: settings[0][1]}
    elif settings:
        args = [value for _, value in settings[:-1]]
        last = len(settings) - 1
        inlet_values = {info.n_inputs + last: settings[-1][1]}
    obj = " ".join([name, *map(repr, args)])
    signal = inputs(case, x) if info.n_inputs else np.zeros((1, FRAMES))
    write_float_wav(tmp_path / "in.wav", signal)
    (tmp_path / "run.pd").write_text(pd_patch(obj, info.n_inputs, info.n_outputs, FRAMES, inlet_values))
    proc = subprocess.run(
        [str(PD), "-nogui", "-batch", "-r", str(SAMPLE_RATE), "-stderr", "-open", "run.pd"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert "couldn't create" not in proc.stderr and "error" not in proc.stderr.lower(), proc.stderr
    actual = read_float_wav(tmp_path / "out.wav")
    changes = tuple((0, param, value) for param, value in settings)
    expected = pd_written(run_driver(driver(case, "float"), case, x, np.float32, "64", changes))
    assert np.array_equal(actual, expected)


def test_pd_flushes_denormals(tmp_path: Path) -> None:
    """Denormals cost up to 100x per operation on some CPUs. The shim sets flush-to-zero around the kernel."""
    assert PD is not None
    name = "cignal_denormal~"
    build_pd_external(FIXTURES / "denormal.py", tmp_path, "Denormal", pd_name=name)
    write_float_wav(tmp_path / "in.wav", np.full((1, FRAMES), 0.5))
    (tmp_path / "run.pd").write_text(pd_patch(name, 1, 1, FRAMES, {}))
    proc = subprocess.run([str(PD), "-nogui", "-batch", "-r", str(SAMPLE_RATE), "-stderr", "-open", "run.pd"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert "couldn't create" not in proc.stderr, proc.stderr
    y = read_float_wav(tmp_path / "out.wav")
    assert np.count_nonzero(y) == 0, f"denormals were not flushed: output {y[0, :3]}"


@pytest.mark.parametrize("case", [c for c in CASES if c.changes], ids=str)
def test_pd_changes_mid_stream_match_driver(tmp_path: Path, driver: Driver, case: Case) -> None:
    """Parameter changes and resets sent while DSP runs, moved to the start of their 64-sample block."""
    assert PD is not None
    x = signals()["noise"]
    info = case.info
    name = f"cignal_{case.kernel.lower()}~"
    build_pd_external(FIXTURES / case.fixture, tmp_path, case.kernel, pd_name=name)
    changes = tuple((frame - frame % 64, param, value) for frame, param, value in case.changes)
    events = tuple(
        (frame, 0, "reset") if param == "reset" else (frame, case.param_index(param) + info.n_inputs, repr(value))
        for frame, param, value in changes
    )
    signal = inputs(case, x) if info.n_inputs else np.zeros((1, FRAMES))
    write_float_wav(tmp_path / "in.wav", signal)
    (tmp_path / "run.pd").write_text(pd_patch(name, info.n_inputs, info.n_outputs, FRAMES, {}, events))
    proc = subprocess.run([str(PD), "-nogui", "-batch", "-r", str(SAMPLE_RATE), "-stderr", "-open", "run.pd"],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert "couldn't create" not in proc.stderr and "error" not in proc.stderr.lower(), proc.stderr
    actual = read_float_wav(tmp_path / "out.wav")
    expected = pd_written(run_driver(driver(case, "float"), case, x, np.float32, "64", changes))
    assert np.array_equal(actual, expected)
