"""SuperCollider non-real-time rendering without sclang: a SynthDef (SCgf v2) and an OSC score, by hand.

The graph is In.ar -> the cignal UGen -> Out.ar, with a Control UGen supplying the parameters so that
/n_set can change them mid-stream.
"""

import math
import struct
import subprocess
from pathlib import Path

SYNTH = "cignal_test"
NODE = 1000
SCALAR, CONTROL, AUDIO = 0, 1, 2


def _pstring(text: str) -> bytes:
    data = text.encode()
    return struct.pack(">B", len(data)) + data


def synthdef(ugen: str, n_inputs: int, n_outputs: int, params: list[tuple[str, float]]) -> tuple[bytes, int]:
    """The SynthDef bytes, and the index of the cignal UGen within it (for /u_cmd)."""
    constants = [float(n_outputs), 0.0]  # In's bus (inputs start after the outputs), Out's bus
    ugens: list[bytes] = []

    def add(name: str, rate: int, inputs: list[tuple[int, int]], outputs: list[int], special: int = 0) -> int:
        spec = _pstring(name) + struct.pack(">biih", rate, len(inputs), len(outputs), special)
        spec += b"".join(struct.pack(">ii", u, o) for u, o in inputs)
        spec += b"".join(struct.pack(">b", r) for r in outputs)
        ugens.append(spec)
        return len(ugens) - 1

    control = add("Control", CONTROL, [], [CONTROL] * len(params)) if params else -1
    source = add("In", AUDIO, [(-1, 0)], [AUDIO] * n_inputs) if n_inputs else -1
    inputs = [(source, k) for k in range(n_inputs)] + [(control, p) for p in range(len(params))]
    kernel = add(ugen, AUDIO, inputs, [AUDIO] * n_outputs)
    add("Out", AUDIO, [(-1, 1)] + [(kernel, k) for k in range(n_outputs)], [])

    body = _pstring(SYNTH)
    body += struct.pack(">i", len(constants)) + b"".join(struct.pack(">f", c) for c in constants)
    body += struct.pack(">i", len(params)) + b"".join(struct.pack(">f", v) for _, v in params)
    body += struct.pack(">i", len(params)) + b"".join(_pstring(n) + struct.pack(">i", i) for i, (n, _) in
                                                      enumerate(params))  # fmt: skip
    body += struct.pack(">i", len(ugens)) + b"".join(ugens) + struct.pack(">h", 0)
    return b"SCgf" + struct.pack(">ih", 2, 1) + body, kernel


def _osc_pad(data: bytes) -> bytes:
    return data + b"\0" * (4 - len(data) % 4)


def osc_message(address: str, *args: int | float | str | bytes) -> bytes:
    tags, payload = ",", b""
    for arg in args:
        if isinstance(arg, bytes):
            tags += "b"
            payload += struct.pack(">i", len(arg)) + arg + b"\0" * (-len(arg) % 4)
        elif isinstance(arg, str):
            tags += "s"
            payload += _osc_pad(arg.encode())
        elif isinstance(arg, int):
            tags += "i"
            payload += struct.pack(">i", arg)
        else:
            tags += "f"
            payload += struct.pack(">f", arg)
    return _osc_pad(address.encode()) + _osc_pad(tags.encode()) + payload


def _bundle(seconds: float, messages: list[bytes]) -> bytes:
    whole = int(seconds)
    # Rounded up: a timetag a fraction of a sample before a block start ran the command a block early
    # (measured: /u_cmd at frame 64, 128 and 448).
    timetag = struct.pack(">II", whole, math.ceil((seconds - whole) * 2**32))
    body = b"#bundle\0" + timetag + b"".join(struct.pack(">i", len(m)) + m for m in messages)
    return struct.pack(">i", len(body)) + body


def render(
    scsynth: Path,
    plugin_dir: Path,
    work: Path,
    ugen: str,
    n_inputs: int,
    n_outputs: int,
    params: list[tuple[str, float]],
    events: list[tuple[int, bytes]],
    frames: int,
    sample_rate: int,
    input_wav: Path | None,
    definition: tuple[str, bytes] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Render `frames` frames to work/out.wav. `events` are (frame, OSC message) pairs.

    `definition` is (SynthDef name, bytes) to use instead of the hand-written one.
    """
    name, data = definition or (SYNTH, synthdef(ugen, n_inputs, n_outputs, params)[0])
    score = [(0, osc_message("/d_recv", data)), (0, osc_message("/s_new", name, NODE, 0, 0)), *events]
    score.append((frames, osc_message("/c_set", 0, 0)))  # the last command's time ends the render
    (work / "score.osc").write_bytes(b"".join(_bundle(f / sample_rate, [m]) for f, m in score))
    bundled = scsynth.parent / "plugins"  # macOS app bundle; Linux packages use /usr/lib
    core = bundled if bundled.is_dir() else Path("/usr/lib/SuperCollider/plugins")
    command = [str(scsynth), "-N", "score.osc", str(input_wav) if input_wav else "_", "out.wav", str(sample_rate),
               "WAV", "float", "-i", str(max(n_inputs, 1)), "-o", str(n_outputs),
               "-U", f"{core}:{plugin_dir}"]  # fmt: skip
    return subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=120)
