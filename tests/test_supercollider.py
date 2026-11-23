"""The SuperCollider UGen, rendered by scsynth in non-real-time mode, matches the driver bit for bit.

Each case's parameter changes and resets are moved to the start of their 64-sample block, since
SuperCollider reads parameters once per block, and the driver gets the same moved changes.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import Driver
from sc_support import NODE, osc_message, render, synthdef
from support import (
    CASES,
    FIXTURES,
    FRAMES,
    SAMPLE_RATE,
    Case,
    inputs,
    read_float_wav,
    run_driver,
    signals,
    write_float_wav,
)

from cignal.build import build_sc_plugin, find_sc_include, find_scsynth, sc_ugen_name

SCSYNTH = find_scsynth()
BLOCK = 64
pytestmark = pytest.mark.skipif(SCSYNTH is None or find_sc_include() is None,
                                reason="scsynth or the SuperCollider headers not found")  # fmt: skip


@pytest.mark.parametrize("case", CASES, ids=str)
def test_sc_ugen_matches_driver(tmp_path: Path, driver: Driver, case: Case) -> None:
    assert SCSYNTH is not None
    info = case.info
    ugen = sc_ugen_name(info)
    plugin = build_sc_plugin(FIXTURES / case.fixture, tmp_path / "plugin", case.kernel)
    x = signals()["noise"]
    if info.n_inputs:
        write_float_wav(tmp_path / "in.wav", inputs(case, x))
    changes = tuple((frame - frame % BLOCK, name, value) for frame, name, value in case.changes)
    params = [(p[0], p[3]) for p in info.params]
    kernel_index = synthdef(ugen, info.n_inputs, info.n_outputs, params)[1]
    events = [
        (frame, osc_message("/u_cmd", NODE, kernel_index, "reset") if name == "reset"
         else osc_message("/n_set", NODE, name, value))
        for frame, name, value in changes
    ]  # fmt: skip
    proc = render(SCSYNTH, plugin.parent, tmp_path, ugen, info.n_inputs, info.n_outputs, params, events, FRAMES,
                  SAMPLE_RATE, tmp_path / "in.wav" if info.n_inputs else None)  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    actual = read_float_wav(tmp_path / "out.wav")[:, :FRAMES]  # scsynth renders the block holding the end time too
    expected = run_driver(driver(case, "float"), case, x, np.float32, str(BLOCK), changes)
    assert np.array_equal(actual, expected)


_BUNDLED_SCLANG = SCSYNTH.parent.parent / "MacOS/sclang" if SCSYNTH else None  # same app as scsynth
SCLANG = shutil.which("sclang") or (str(_BUNDLED_SCLANG) if _BUNDLED_SCLANG and _BUNDLED_SCLANG.is_file() else None)


@pytest.fixture(scope="module")
def sclang_defs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, bytes]]:
    """Every case's plugin and generated class, and a SynthDef that sclang builds from that class.

    One sclang run for all cases: compiling the class library takes seconds.
    """
    if SCLANG is None:
        pytest.skip("sclang not found")
    root = tmp_path_factory.mktemp("sclang")
    lines, plugins = [], {}
    for case in CASES:
        info = case.info
        ugen = sc_ugen_name(info)
        plugins[case.kernel] = build_sc_plugin(FIXTURES / case.fixture, root / "ext" / case.kernel, case.kernel)
        args = ", ".join(f"{name} = {default!r}" for name, _, _, default in info.params)
        ins = [f"In.ar({info.n_outputs + k}, 1)" for k in range(info.n_inputs)]
        call = f"{ugen}.ar({', '.join([*ins, *(p[0] for p in info.params)])})"
        lines.append(f'SynthDef(\\cignal_{case.kernel}, {{ {f"|{args}| " if args else ""}Out.ar(0, {call}) }})'
                     f'.writeDefFile("{root / "defs"}");')  # fmt: skip
    (root / "defs").mkdir()
    (root / "conf.yaml").write_text(f"includePaths:\n  - {root / 'ext'}\n")
    (root / "defs.scd").write_text("\n".join([*lines, '"CIGNAL_DONE".postln;', "0.exit;", ""]))
    proc = subprocess.run([SCLANG, "-l", str(root / "conf.yaml"), str(root / "defs.scd")], capture_output=True,
                          text=True, timeout=180)  # fmt: skip
    assert "CIGNAL_DONE" in proc.stdout and "ERROR" not in proc.stdout, proc.stdout[-3000:]
    return {k: (plugin, (root / "defs" / f"cignal_{k}.scsyndef").read_bytes()) for k, plugin in plugins.items()}


@pytest.mark.parametrize("case", CASES, ids=str)
def test_generated_sc_class_builds_a_working_synthdef(
    tmp_path: Path, driver: Driver, sclang_defs: dict[str, tuple[Path, bytes]], case: Case
) -> None:
    """The .sc class file, compiled by sclang, drives the plugin as the hand-written SynthDef does."""
    assert SCSYNTH is not None
    info = case.info
    plugin, definition = sclang_defs[case.kernel]
    x = signals()["noise"]
    if info.n_inputs:
        write_float_wav(tmp_path / "in.wav", inputs(case, x))
    # Resets need the UGen's index for /u_cmd, which sclang chooses; test_sc_ugen_matches_driver covers them.
    changes = tuple((f - f % BLOCK, n, v) for f, n, v in case.changes if n != "reset")
    events = [(frame, osc_message("/n_set", NODE, name, value)) for frame, name, value in changes]
    proc = render(SCSYNTH, plugin.parent, tmp_path, sc_ugen_name(info), info.n_inputs, info.n_outputs, [], events,
                  FRAMES, SAMPLE_RATE, tmp_path / "in.wav" if info.n_inputs else None,
                  definition=(f"cignal_{case.kernel}", definition))  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    actual = read_float_wav(tmp_path / "out.wav")[:, :FRAMES]
    expected = run_driver(driver(case, "float"), case, x, np.float32, str(BLOCK), changes)
    assert np.array_equal(actual, expected)


def test_sc_ugen_flushes_denormals(tmp_path: Path) -> None:
    """The shim sets flush-to-zero around the kernel; see the fixture for how the probe works."""
    assert SCSYNTH is not None
    plugin = build_sc_plugin(FIXTURES / "denormal.py", tmp_path / "plugin", "Denormal")
    write_float_wav(tmp_path / "in.wav", np.full((1, FRAMES), 0.5))
    proc = render(SCSYNTH, plugin.parent, tmp_path, "CignalDenormal", 1, 1, [], [], FRAMES, SAMPLE_RATE,
                  tmp_path / "in.wav")  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    y = read_float_wav(tmp_path / "out.wav")[:, :FRAMES]
    assert np.count_nonzero(y) == 0, f"denormals were not flushed: output {y[0, :3]}"
