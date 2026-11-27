# TODO

Design questions are argued in [`docs/dev/spec.md`](docs/dev/spec.md) section 13. This file tracks the work.

## Critical

## High

### Before v1.0

- [x] **Built-ins:** `abs`, `min`, `max`, `math.fmod`, `math.log2`, `math.log10`, `math.copysign` (spec 5.2).

- [x] **CI workflow.** GitHub Actions on Ubuntu x86-64, with Pd and SuperCollider, so the host tests run on every push.

- [x] **PyPI release.** 0.0.1, 2026-11-22.

### Found by writing Freeverb and a Moog ladder (2026-11-23)

- [x] **Index module-level tuples** (spec 3.5). Freeverb's tuning table needs `COMBS[i]`.

- [x] **Arrays of sub-kernels and of `Delay`** (spec 3.6). Freeverb went from 96 lines with 12 named fields to 60. Revisit with the SIMD ABI: it exposes parallelism inside one instance.

- [x] **Freeverb and ladder fixtures.**

- [x] **Float SNR threshold per case** (spec 9.1). The ladder passes at 90 dB.

### Found by writing an 8-voice synth (2026-11-23)

`tests/fixtures/poly.py`: saw, envelope and lowpass per voice, 19 parameters, 20 ns/frame on clang `-O2`.

- [x] **Notes are not events.** A gate is a parameter, so a note-off and note-on at the same frame collapse to the last value, and re-sending a held gate does nothing. Fixed by `@event` methods (`docs/dev/events.md`).

- [x] **Voice allocation is the host's job.** `tests/fixtures/synth.py` takes notes and allocates its voices.

- [x] **`smooth=` state is `f32` in the float build.** Smoothing the voice frequency dropped the float SNR from 116 to 98 dB, because the phase integrates every rounding of the pitch. Fixed by `f64` parameters: 115 dB.

- [x] **Hand-written glides repeat the snap rule.** `glide(value, target, coef)` is the parameter step as a function (`tests/fixtures/portamento.py`).

- [x] **Pd inlet count.** 19 parameters give 19 inlets. A `name value` message on the left inlet now sets any of them (`freqs_3 440`).

- [x] **A sub-kernel's `tick` must take `f32` samples**, even when only its parent calls it. The rule now applies to the exported kernel only.

- [x] **Sub-kernels cannot read their parent.** Won't do (2026-11-25). A sub-kernel that names its parent's fields compiles only inside that one parent class, so `Voice` could not serve two synths. Arguments, or writes into each voice from `control` as Freeverb does, cover the need.

### Found by writing the event synth (2026-11-23)

- [x] **Index proofs do not follow assignments out of a loop.** A local assigned only proven values now proves an index (spec 5.4).

- [x] **SuperCollider events see last block's parameters.** The shim now queues unit commands until it has read the block's inputs (`tests/fixtures/grab.py`).

- [x] **clang rewrites `pow(2.0, x)` as `exp2(x)`.** CPython calls `pow`. The synth fixture's `2.0 ** x` matched by chance; `modfilter.py` did not (Apple's `exp2` and `pow` differ on 0.19% of inputs). Fixed 2026-11-27 by calling `pow` through an asm-label alias. Not yet re-run on glibc.

### Found by writing swept filters (2026-11-27)

`tests/fixtures/modfilter.py`: an SVF whose cutoff follows an LFO, retuned with a `tan` and an `exp2`. ns/frame, clang `-O2`, arm64:

| | per sample | counter, every 32 | hand-chunked C, every 32 | no retune |
|-|-|-|-|-|
| mono | 13.9 | 8.2 | 4.4 | 4.3 |
| 8 voices | 113.1 | 14.3 | | 10.9 |

- [ ] **`@every(N)` methods?** A counter field in `tick` already gives a fixed control rate that does not depend on host blocks. Lowering it as a chunked loop, with no libm call in the inner loop, made mono 1.9x faster again and bit-identical. The 8-voice bank already reaches its floor with the counter. Decide after a second kernel that needs it.

## Medium

### Reference model

- [x] **`i32` range in the reference** (spec 13.3). `CHECK_I32` checks stored values after each `tick`; the tests set it.

- [ ] **`f32` rounding in the reference** (spec 13.4). `numpy.float32` is ruled out: its `sin` and `tanh` differ from libm. Remaining option: a test-only shadow reference calling libm `f` functions through ctypes.

### Before v1.0

- [x] **Multi-channel instances for SIMD** (spec 13.6). Not needed: `Array[K, N]` gives the parallelism. See `docs/dev/simd.md`.

- [x] **Is field layout ABI?** No (spec 6, 13.6). Hosts use `sizeof`, parameters and events only.

- [x] **Per-voice parameters** (spec 3.4). `Array[f32, N] = param(...)`, seen by hosts as N scalar parameters.

- [x] **Coefficients from smoothed parameters.** `control` reads `self.target("cutoff")`. The premise was wrong: moving the ladder's `exp` to `control` and gliding the coefficient instead left it at 83 ns/frame, since its 20 `tanh` calls dominate. A hand-written `f32` coefficient glide also cost 15 to 30 dB of float SNR (`f64` state fixed it), so the ladder keeps its smoothed cutoff.

- [x] **macOS CI job.** Pd and SuperCollider from Homebrew casks.

## Low

- [x] **New fixtures in `bench/run.py`.**

- [ ] **Loops over sub-kernel arrays at clang `-O2`.** Freeverb with `Array[Comb, 8]` measured 1.2 to 1.5x slower than hand-unrolled, and level at `-O3 -mcpu=native`. Timings were noisy. Forcing unrolling on clang slowed the ladders by 12%, so it is GCC-only (`docs/dev/simd.md`).

- [x] **Loops over sub-kernel arrays at GCC `-O2`.** Short constant loops are unrolled under GCC: 3.9x on the 8-biquad bank, 1.15x on Freeverb.

- [x] **Struct-of-arrays layout for `Array[K, N]`**. 2.8x on an 8-biquad bank, 1.3 to 1.4x on Poly, bit-exact (`docs/dev/soa.md`). Freeverb is 5% slower at clang `-O2`.

- [ ] **Vectorised `@table` lookups.** Untested guess: a table `tanh` is arithmetic plus an indexed load, so it may vectorise under SoA and stay bit-exact.

### Later

- [ ] **Fixed-point output** (spec 13.5): Vult's q16.16, for Teensy and Eurorack. Only if hardware targets become near-term.

- [ ] **More hosts:** candidates are CLAP and Max. Each needs a shim in `src/cignal/shims/` and a bit-exact host test, as Pd and SuperCollider have.
