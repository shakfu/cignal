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

- [ ] **Notes are not events.** A gate is a parameter, so a note-off and note-on at the same frame collapse to the last value, and re-sending a held gate does nothing: a voice cannot retrigger. Options: a per-parameter "changed" flag readable in `control`, or event methods hosts call between blocks (an ABI addition).

- [ ] **Voice allocation is the host's job.** The kernel cannot take "note 60, velocity 100"; the host must pick a voice and set `freqs_k` and `gates_k`. Event methods (above) would let the kernel allocate.

- [ ] **`smooth=` state is `f32` in the float build.** Smoothing the voice frequency dropped the float SNR from 116 to 98 dB, because the phase integrates every rounding of the pitch. The fixture glides an `f64` field by hand instead. Option: keep smoother state in `f64`, or allow `f64` parameters.

- [ ] **Hand-written glides repeat the snap rule.** The ladder experiment and this fixture both copied `smooth_step`'s snap to avoid a float stall. Option: a built-in `glide(value, target, coef)`, or `smooth=` on plain fields.

- [ ] **Pd inlet count.** 19 parameters give 19 inlets. Option: a `name value` message on the left inlet (`freqs_3 440`); a shim-only change.

- [ ] **A sub-kernel's `tick` must take `f32` samples**, even when only its parent calls it. The voice method is `play` to take an `f64` pitch. Minor: the rule could apply only to the exported kernel.

- [ ] **Sub-kernels cannot read their parent.** Each voice call passes 7 shared values as arguments. Writing them into every voice in `control`, as Freeverb does, is the alternative. Minor.

## Medium

### Reference model

- [ ] **`i32` range in the reference** (spec 13.3). C wraps at 32 bits; CPython ints do not. Detect a stored value that leaves `i32` range. Option: a test-mode AST rewrite that asserts range.

- [ ] **`f32` rounding in the reference** (spec 13.4). Option: emulate `f32` with `numpy.float32`, for a bit-near check of the `float` build. Cost: a slower reference and NumPy at runtime.

### Before v1.0

- [x] **Multi-channel instances for SIMD** (spec 13.6). Not needed: `Array[K, N]` gives the parallelism. See `docs/dev/simd.md`.

- [ ] **Is field layout ABI?** Decide at the v1.0 freeze. Not ABI allows a struct-of-arrays `Array[K, N]`; ABI lets hosts read fields.

- [x] **Per-voice parameters** (spec 3.4). `Array[f32, N] = param(...)`, seen by hosts as N scalar parameters.

- [x] **Coefficients from smoothed parameters.** `control` reads `self.target("cutoff")`. The premise was wrong: moving the ladder's `exp` to `control` and gliding the coefficient instead left it at 83 ns/frame, since its 20 `tanh` calls dominate. A hand-written `f32` coefficient glide also cost 15 to 30 dB of float SNR (`f64` state fixed it), so the ladder keeps its smoothed cutoff.

- [ ] **macOS CI job.** CI covers GCC and glibc only; clang and Apple's libm are checked only on a developer machine.

## Low

- [x] **New fixtures in `bench/run.py`.**

- [ ] **Loops over sub-kernel arrays at clang `-O2`.** Freeverb with `Array[Comb, 8]` measured 1.2 to 1.5x slower than hand-unrolled, and level at `-O3 -mcpu=native`. Timings were noisy. Forcing unrolling on clang slowed the ladders by 12%, so it is GCC-only (`docs/dev/simd.md`).

- [x] **Loops over sub-kernel arrays at GCC `-O2`.** Short constant loops are unrolled under GCC: 3.9x on the 8-biquad bank, 1.15x on Freeverb.

- [ ] **Struct-of-arrays layout for `Array[K, N]`**. 1.5 to 1.9x on an 8-voice biquad bank, bit-exact (`docs/dev/simd.md`).

- [ ] **Vectorised `@table` lookups.** Untested guess: a table `tanh` is arithmetic plus an indexed load, so it may vectorise under SoA and stay bit-exact.

### Later

- [ ] **Fixed-point output** (spec 13.5): Vult's q16.16, for Teensy and Eurorack. Only if hardware targets become near-term.

- [ ] **More hosts:** candidates are CLAP and Max. Each needs a shim in `src/cignal/shims/` and a bit-exact host test, as Pd and SuperCollider have.
