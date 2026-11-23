# TODO

Design questions are argued in [`docs/dev/spec.md`](docs/dev/spec.md) section 13. This file tracks the work.

## Critical

## High

### Before v1.0

- [x] **Built-ins:** `abs`, `min`, `max`, `math.fmod`, `math.log2`, `math.log10`, `math.copysign` (spec 5.2).

- [x] **CI workflow.** GitHub Actions on Ubuntu x86-64, with Pd and SuperCollider, so the host tests run on every push.

- [x] **PyPI release.** 0.0.1, 2026-11-22.

## Medium

### Reference model

- [ ] **`i32` range in the reference** (spec 13.3). C wraps at 32 bits; CPython ints do not. Detect a stored value that leaves `i32` range. Option: a test-mode AST rewrite that asserts range.

- [ ] **`f32` rounding in the reference** (spec 13.4). Option: emulate `f32` with `numpy.float32`, for a bit-near check of the `float` build. Cost: a slower reference and NumPy at runtime.

### Before v1.0

- [ ] **Multi-channel instances for SIMD** (spec 7, 13.6). Changes the kernel ABI, so decide before v1.0 freezes it.

- [ ] **macOS CI job.** CI covers GCC and glibc only; clang and Apple's libm are checked only on a developer machine.

## Low

### Later

- [ ] **Fixed-point output** (spec 13.5): Vult's q16.16, for Teensy and Eurorack. Only if hardware targets become near-term.

- [ ] **More hosts:** candidates are CLAP and Max. Each needs a shim in `src/cignal/shims/` and a bit-exact host test, as Pd and SuperCollider have.

