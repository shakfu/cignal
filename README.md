# cignal

cignal compiles DSP kernels written in a static subset of Python to C99, for real-time audio hosts. The same source runs unchanged under CPython, as the reference model the C is tested against.

- One C99 kernel ABI: a state struct, `init`, `process`, `set_param`, `event`, `reset`. No allocation, locks or I/O in the processing path.

- Host shims for [Pd](https://puredata.info/) externals and [SuperCollider](https://supercollider.github.io/) UGens, plus a standalone test driver.

- Compiled with `double` in place of `f32`, the C matches CPython bit for bit. The test suite checks this for every fixture on macOS (clang) and Linux (GCC, glibc).

The design follows [Vult](https://github.com/vult-dsp/vult). The full specification, with the reasons for each choice, is in [`docs/dev/spec.md`](docs/dev/spec.md).

Status: 0.x, on [PyPI](https://pypi.org/project/cignal/). The language and the ABI may change before 1.0.

## Requirements

- Python 3.12 or later, and [uv](https://docs.astral.sh/uv/).

- A C99 compiler (`cc`, or `$CC`). The SuperCollider shim also needs a C++17 compiler (`c++`, or `$CXX`).

- For Pd externals: `m_pd.h`, from `$PD_INCLUDE`, `/usr/include/pd` or a macOS `Pd*.app`.

- For SuperCollider UGens: the plugin headers, from `$SC_INCLUDE` or `/usr/include/SuperCollider`. The macOS app does not include them.

- Without either, `cignal fetch-headers` downloads `m_pd.h` from Pd 0.57.0 and the SuperCollider 3.14.1 headers into the user cache, verified against pinned SHA-256 hashes. cignal does not ship them: the SuperCollider headers are GPL-2.0-or-later.

```sh
uv sync
```

## A kernel

```python
import math

from cignal import Kernel, f32, param


class Section(Kernel):
    b0: f32 = 1.0
    b1: f32 = 0.0
    b2: f32 = 0.0
    a1: f32 = 0.0
    a2: f32 = 0.0
    z1: f32 = 0.0
    z2: f32 = 0.0

    def tick(self, x: f32) -> f32:
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y


class Lowpass(Kernel):
    cutoff: f32 = param(1000.0, lo=20.0, hi=20000.0)
    q: f32 = param(0.707, lo=0.1, hi=20.0)
    section: Section  # a sub-kernel: its state is part of Lowpass's struct

    def control(self) -> None:  # once per block
        w = 2.0 * math.pi * self.cutoff / self.sample_rate
        alpha = math.sin(w) / (2.0 * self.q)
        c = math.cos(w)
        a0 = 1.0 + alpha
        self.section.b0 = (1.0 - c) / 2.0 / a0
        self.section.b1 = (1.0 - c) / a0
        self.section.b2 = self.section.b0
        self.section.a1 = -2.0 * c / a0
        self.section.a2 = (1.0 - alpha) / a0

    def tick(self, x: f32) -> f32:  # once per sample
        return self.section.tick(x)
```

As Python:

```python
lp = Lowpass(sample_rate=48000)
lp.set_param("cutoff", 500.0)
out = lp.process([[1.0] + [0.0] * 63], 64)  # one list per input channel
```

Fields are state. `tick` sets the inputs and outputs: its parameters are input samples, and it returns one sample or a tuple. `init` runs once, and again on `reset`.

### Events

Parameters hold a value; events are calls. A method marked `@event` and named `on_<name>` runs when the host sends `<name>`, between blocks:

```python
from cignal import Array, Kernel, event, f32, i32


class Synth(Kernel):
    voices: Array[Voice, 4]  # Voice: a sub-kernel with `start` and `tick`
    next: i32 = 0

    @event
    def on_note(self, note: i32, velocity: f32) -> None:
        self.voices[self.next % 4].start(note, velocity)  # `% 4` proves the index
        self.next = (self.next + 1) % 4

    def tick(self) -> f32:
        acc = 0.0
        for i in range(4):
            acc += self.voices[i].tick()
        return acc
```

In Python, `synth.event("note", 60, 0.8)`. A note-off and a note-on in one block both run, in order, and re-sending a note retriggers it. Arguments are `f32`, `f64`, `i32` or `bool`; a host value converts to `i32` by truncation, saturating, with NaN as 0. `tests/fixtures/synth.py` is a full example with voice allocation and sustain.

## Build

```sh
uv run cignal check lowpass.py                       # diagnostics only
uv run cignal build lowpass.py -o out                # out/lowpass.h, out/lowpass.c
uv run cignal build lowpass.py -o out --shim pd      # out/lowpass~.pd_darwin (or .pd_linux)
uv run cignal build lowpass.py -o out --shim sc      # out/CignalLowpass.so (.scx) and CignalLowpass.sc
uv run cignal build lowpass.py -o out --prefix fx_   # fx_Lowpass, fx_Section, ...
uv run cignal fetch-headers                          # or `pd` / `sc`: host headers into the user cache
```

A file with several kernels needs `--kernel NAME`, unless only one of them is not a field of another.

- **Pd.** Inlets are the signal inputs, then one float inlet per parameter. Creation arguments set the parameters in order, and a `reset` message clears state. Events are messages on the left inlet: `note 60 0.8`. Pick a name that is not a Pd built-in with `--pd-name`: `biquad~` would load Pd's own object, without an error.

- **SuperCollider.** Copy the plugin and the `.sc` class file into an extensions folder. UGen inputs are the signal inputs, then one input per parameter: `CignalLowpass.ar(sig, 800, 2)`. Parameters are read once per block. `/u_cmd node ugen "reset"` clears state, and `/u_cmd node ugen "note" 60 0.8` sends an event.

Both shims flush denormals to zero while the kernel runs, and restore the host's mode afterwards.

## Things that differ from plain Python

- **Types are explicit.** `f32`, `f64`, `i32`, `bool`. A bare `float` or `int` annotation is an error. Mixing types without a cast (`f32(x)`) is an error; literals adapt.

- **Python semantics win** where C differs. `//` and `%` floor, as in Python; `-7 // 2` is `-4`. `>>` floors on negative values.

- **`i32` wraps** at 32 bits in C. CPython ints do not, so a value that leaves `i32` range is outside the contract, unless it is masked back into range.

- **Sizes are build-time constants.** `Array[f32, 16]`, `Delay[f32, MAX]`, `Array[Comb, 8]` of sub-kernels, `@table(size=512, ...)`. Module-level constants and `math` expressions on them are folded in CPython.

- **Array indices must be provably in range.** Use a constant, a `range` loop variable with constant bounds, or `e % N`. Anything else is a compile error, and there are no runtime bounds checks.

- **`Delay.read(d)` counts from 1.** `read(1)` is the newest sample, so `read(D)` before `write(x)` delays by exactly D samples.

- **Smoothed parameters** (`param(..., smooth=0.01)`) glide per sample. `control` may not read them, because the value it saw would depend on block size. It may read `self.target("cutoff")`, the last value set.

- **`f64` parameters** (`freq: f64 = param(...)`) keep their value and glide in `double` in the float build too. Use them for a pitch that a phase integrates.

- **Array parameters** (`cutoffs: Array[f32, 8] = param(...)`) give each voice its own value. Hosts see `cutoffs_0` to `cutoffs_7`.

- **Event methods need both `@event` and the `on_` prefix.** Names Pd uses itself, such as `bang`, `float` and `reset`, are reserved.

- **Tables** (`@table(size=..., lo=..., hi=...)` on a module-level function) are sampled in CPython at build time and interpolated linearly. The function body can use anything, SciPy included.

- **Module-level functions** called from kernels are compiled; others, such as build-time filter design, are left alone.

- **`while` compiles, with a warning:** its termination is not checked against the audio deadline.

## Performance

`K_process` matches hand-written C for a biquad on clang and GCC, and beats it for a 16-tap FIR. `make bench` prints ns per sample for every fixture. Float `%` inside a recursion is slow on clang. For a phase accumulator whose increment is in [0, 1), use `phase += inc; if phase >= 1.0: phase -= 1.0`. See spec section 9.5.

## Tests

```sh
make test                     # pytest; Pd and SuperCollider tests skip when not installed
make qa                       # lint, format and type checks, then tests, as in CI
make linux-image test-linux   # Debian in an Apple `container` VM: GCC, glibc, Pd 0.55, SuperCollider 3.13
make linux-image test-linux ARCH=amd64
make bench
```

Each fixture in `tests/fixtures/` is compiled and checked four ways:

- the `double` build against CPython, bit for bit;

- the `float` build against the `double` build, SNR >= 100 dB by default, set per case for a measured reason (the self-oscillating ladder passes at 90 dB);

- six block-size patterns, which must give identical output;

- in Pd and in SuperCollider, against the driver, bit for bit, with mid-stream parameter changes, resets and events.

## Layout

```text
src/cignal/kernel.py       the Python runtime: Kernel, param, Array, Delay, table
src/cignal/compiler.py     source -> typed walk -> C
src/cignal/build.py        writing C, rendering shims, invoking the compiler
src/cignal/headers.py      `cignal fetch-headers`: pinned host header downloads
src/cignal/shims/          driver, Pd, SuperCollider, benchmark and flush-to-zero templates
tests/                     fixtures and the differential, host, diagnostics and codegen tests
thirdparty/                Pd and SuperCollider headers for the tests, not packaged
bench/                     the benchmark runner and hand-written C references
ci/linux.Containerfile     the Linux test image
docs/dev/spec.md           the specification
docs/dev/                  decision records: SIMD, events
```

## License

MIT. See [LICENSE](LICENSE). The headers in `thirdparty/` keep their own licenses: [Pd](thirdparty/pd/LICENSE.txt) (BSD-3-Clause) and [SuperCollider](thirdparty/supercollider/COPYING) (GPL-2.0-or-later).
