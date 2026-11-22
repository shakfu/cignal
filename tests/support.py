"""Shared helpers: test cases, signals, the CPython reference, driver runs, float WAV I/O, Pd patches."""

import functools
import importlib.util
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from cignal.compiler import CKernel, compile_kernel

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_RATE = 48000
FRAMES = 4800  # a multiple of Pd's 64-sample block

Change = tuple[int, str, float]  # (frame, parameter, value); the parameter "reset" resets the kernel


@dataclass(frozen=True)
class Case:
    fixture: str
    kernel: str
    # Parameter changes, at frames that do not fall on block boundaries. Some values are out of range,
    # to test that the setter clamps. Values are rounded to float32, like the signals, so every build
    # sees the same parameter.
    changes: tuple[Change, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "changes", tuple((f, n, float(np.float32(v))) for f, n, v in self.changes))

    def __str__(self) -> str:
        return self.kernel

    @property
    def info(self) -> CKernel:
        return _compiled(self.fixture, self.kernel)

    def param_index(self, name: str) -> int:
        return [p[0] for p in self.info.params].index(name)


CASES = [
    Case("biquad.py", "Biquad"),
    Case("onepole.py", "OnePole"),
    Case("mix.py", "Mix"),
    Case("lowpass.py", "Lowpass", ((1000, "cutoff", 3000.0), (3001, "q", 4.0), (4000, "cutoff", 1e6))),
    Case("saw.py", "Saw", ((1200, "freq", 1000.0), (2500, "freq", -5.0), (3100, "freq", 220.5))),
    Case("stereo.py", "Stereo", ((700, "balance", -0.5), (2000, "gain", 2.5), (3333, "balance", 7.0))),
    Case(
        "echo.py", "Echo", ((900, "time", 0.0333), (2000, "feedback", 0.9), (2600, "reset", 0.0), (3500, "time", 1.0))
    ),
    Case(
        "comb.py", "Comb", ((1100, "delay", 337.6), (2222, "damp", 0.6), (3000, "reset", 0.0), (3001, "feedback", -0.8))
    ),
    Case("svf.py", "SVF", ((800, "cutoff", 5000.0), (1601, "res", 0.9), (2400, "reset", 0.0), (3200, "cutoff", 1e6))),
    Case("fir.py", "Fir", ((1500, "cutoff", 12000.0), (2900, "reset", 0.0), (3300, "cutoff", 1.0))),
    # No reset: the only state is the smoothed parameter, and reset keeps parameters.
    Case("drive.py", "Drive", ((1000, "drive", 3.0), (2500, "drive", 0.5), (3600, "drive", 10.0))),
    Case("ops.py", "Ops"),
    Case("glide.py", "Glide", ((0, "level", 440.0), (2000, "level", 0.0))),
    Case("wavetable.py", "Wavetable", ((1700, "freq", 1234.5), (3100, "reset", 0.0), (3900, "freq", 99.0))),
]


@functools.cache
def _compiled(fixture: str, kernel: str) -> CKernel:
    return compile_kernel(FIXTURES / fixture, kernel)


def signals() -> dict[str, np.ndarray]:
    """Test inputs, rounded to float32 so every build sees identical samples."""
    rng = np.random.default_rng(1234)
    t = np.arange(FRAMES)
    impulse = np.zeros(FRAMES)
    impulse[0] = 1.0
    burst = np.concatenate([np.zeros(FRAMES // 2), 0.5 * rng.uniform(-1, 1, FRAMES - FRAMES // 2)])
    raw = {
        "impulse": impulse,
        "step": np.ones(FRAMES),
        "sine": np.sin(2 * np.pi * 440.0 * t / SAMPLE_RATE),
        "noise": rng.uniform(-1, 1, FRAMES),
        "silence_burst": burst,
    }
    return {name: x.astype(np.float32).astype(np.float64) for name, x in raw.items()}


def inputs(case: Case, x: np.ndarray) -> np.ndarray:
    """Shape (n_inputs, frames). Channel k is `x` reversed k times, so stereo channels differ."""
    channels = [x if k % 2 == 0 else x[::-1] for k in range(case.info.n_inputs)]
    return np.stack(channels) if channels else np.zeros((0, len(x)))


def _module(fixture: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ref_{Path(fixture).stem}", FIXTURES / fixture)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # string annotations evaluate in the module's globals
    spec.loader.exec_module(module)
    return module


def reference(case: Case, x: np.ndarray, changes: tuple[Change, ...] | None = None) -> np.ndarray:
    """Run the kernel under CPython, splitting `process` calls at the parameter changes."""
    changes = case.changes if changes is None else changes
    kernel = getattr(_module(case.fixture), case.kernel)(SAMPLE_RATE)
    data = inputs(case, x)
    frames = data.shape[1]
    out: list[list[float]] = []
    pos = 0
    for frame, name, value in [*changes, (frames, "", 0.0)]:
        if frame > pos:
            block = kernel.process([channel[pos:frame].tolist() for channel in data], frame - pos)
            out = block if not out else [a + b for a, b in zip(out, block, strict=True)]
            pos = frame
        if name == "reset":
            kernel.reset()
        elif name:
            kernel.set_param(name, value)
    return np.array(out)


def run_driver(
    exe: Path, case: Case, x: np.ndarray, dtype: type, blocks: str = "64", changes: tuple[Change, ...] | None = None
) -> np.ndarray:
    changes = case.changes if changes is None else changes
    data = inputs(case, x)
    args = [str(exe), str(SAMPLE_RATE), blocks, "--frames", str(len(x))]
    for frame, name, value in changes:
        args += ["--reset", str(frame)] if name == "reset" else ["--set", f"{frame}:{case.param_index(name)}:{value!r}"]
    proc = subprocess.run(args, input=data.T.astype(dtype).tobytes(), capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=dtype).astype(np.float64).reshape(-1, case.info.n_outputs).T


def snr_db(reference: np.ndarray, actual: np.ndarray) -> float:
    noise = float(np.sum((actual - reference) ** 2))
    return float("inf") if noise == 0 else 10 * np.log10(float(np.sum(reference**2)) / noise)


def write_float_wav(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write `data` of shape (channels, frames) as 32-bit float WAV."""
    channels = data.shape[0]
    payload = data.T.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, channels, sample_rate, sample_rate * 4 * channels, 4 * channels, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def read_float_wav(path: Path) -> np.ndarray:
    """Read a 32-bit float WAV as shape (channels, frames)."""
    raw = path.read_bytes()
    pos, channels = 12, 1
    while pos < len(raw):
        chunk, size = raw[pos : pos + 4], struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        if chunk == b"fmt ":
            channels = struct.unpack("<H", raw[pos + 10 : pos + 12])[0]
        if chunk == b"data":
            samples = np.frombuffer(raw[pos + 8 : pos + 8 + size], dtype="<f4").astype(np.float64)
            return samples.reshape(-1, channels).T
        pos += 8 + size + (size & 1)
    raise ValueError(f"no data chunk in {path}")


def pd_written(x: np.ndarray) -> np.ndarray:
    """What `tabwrite~` stores: Pd's PD_BIGORSMALL check zeroes |x| < 2**-63 and |x| >= 2**65.

    Measured on the glide fixture: every sample below 2**-63 came back as 0, every other sample exact.
    """
    magnitude = np.abs(x)
    return np.where((magnitude < 2.0**-63) | (magnitude >= 2.0**65), 0.0, x)


def pd_patch(
    obj: str,
    n_inputs: int,
    n_outputs: int,
    frames: int,
    inlet_values: dict[int, float],
    events: tuple[tuple[int, int, str], ...] = (),
) -> str:
    """A batch patch: play in.wav through `obj`, write out.wav, quit.

    `inlet_values` maps an inlet index to a float sent before DSP starts. `events` are (frame, inlet,
    message) triples sent mid-stream, each through `[delay frame 1 samp]`. A kernel without inputs still
    gets one input array, played but unconnected, to time the run.
    """
    lines: list[str] = ["#N canvas 0 50 800 600 12;"]
    connects: list[tuple[int, int, int, int]] = []

    def add(text: str) -> int:
        lines.append(f"#X {text};")
        return len(lines) - 2

    n_arrays = max(n_inputs, 1)
    ins = " ".join(f"in_{k}" for k in range(n_arrays))
    outs = " ".join(f"out_{k}" for k in range(n_outputs))
    loadbang = add("obj 10 10 loadbang")
    trigger = add("obj 10 40 t b b b b")
    read = add(f"msg 300 70 read -resize in.wav {ins}")
    reader = add("obj 300 100 soundfiler")
    dsp = add("msg 200 130 \\; pd dsp 1")
    kernel = add(f"obj 10 220 {obj}")
    connects += [(loadbang, 0, trigger, 0), (trigger, 3, read, 0), (read, 0, reader, 0), (trigger, 1, dsp, 0)]
    for inlet, value in inlet_values.items():
        message = add(f"msg {400 + 60 * inlet} 160 {value!r}")
        connects += [(trigger, 2, message, 0), (message, 0, kernel, inlet)]
    for k, (frame, inlet, text) in enumerate(events):
        timer = add(f"obj {500 + 90 * k} 120 delay {frame} 1 samp")
        message = add(f"msg {500 + 90 * k} 150 {text}")
        connects += [(trigger, 0, timer, 0), (timer, 0, message, 0), (message, 0, kernel, inlet)]
    players = []
    for k in range(n_arrays):
        add(f"obj 500 {300 + 30 * k} table in_{k}")
        player = add(f"obj {10 + 120 * k} 180 tabplay~ in_{k}")
        players.append(player)
        connects.append((trigger, 0, player, 0))
        if k < n_inputs:
            connects.append((player, 0, kernel, k))
    done = add("obj 300 260 t b b")
    connects.append((players[0], 1, done, 0))
    for k in range(n_outputs):
        add(f"obj 650 {300 + 30 * k} table out_{k} {frames}")
        writer = add(f"obj {10 + 120 * k} 260 tabwrite~ out_{k}")
        connects += [(trigger, 0, writer, 0), (kernel, k, writer, 0)]
    write = add(f"msg 300 290 write -bytes 4 out.wav {outs}")
    writer_obj = add("obj 300 320 soundfiler")
    quit_msg = add("msg 300 350 \\; pd quit")
    connects += [(done, 1, write, 0), (write, 0, writer_obj, 0), (done, 0, quit_msg, 0)]
    lines += [f"#X connect {a} {b} {c} {d};" for a, b, c, d in connects]
    return "\n".join(lines) + "\n"
