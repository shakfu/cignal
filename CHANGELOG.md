# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.0.1]

First version. Milestones M0 to M5 are in [`docs/dev/spec.md`](docs/dev/spec.md) section 14.

### Added

- Compiler from a static Python subset to C99. Kernels have typed fields (`f32`, `f64`, `i32`, `bool`), `init`, `control` (per block) and `tick` (per sample).

- Kernel ABI: a state struct, `init`, `reset`, `process`, `set_param`, named setters and `info`. No allocation, locks or I/O in `process`.

- Parameters with ranges and optional per-sample smoothing. Sub-kernels as fields, several inputs and outputs, tuple returns.

- `Array` and `Delay` with build-time sizes. Indices must be provably in range; there are no runtime checks.

- `@table`: module-level functions sampled in CPython at build time, interpolated linearly.

- Python semantics for `//`, `%`, `**`, shifts and chained comparisons. `i32` arithmetic wraps at 32 bits, where plain C signed overflow is undefined.

- Module-level functions, compiled only when a kernel calls them. `while`, with a `CIGNAL.UNBOUNDED_LOOP` warning.

- CLI: `cignal check` (text or JSON diagnostics) and `cignal build` with `--shim none|driver|pd|sc` and `--prefix`.

- Pd external and SuperCollider UGen shims, plus an sclang class file. Both shims flush denormals to zero while the kernel runs. macOS Pd 0.57 does not flush, and Debian Pd 0.55 does.

- Differential tests:
  - the `double` build against CPython, bit for bit;

  - the `float` build against the `double` build, SNR >= 100 dB;

  - six block-size patterns, identical output;

  - Pd and scsynth against the driver, bit for bit, with mid-stream parameter changes and resets.

- Linux test images for arm64 and x86-64 (`make test-linux`), and `make bench` with hand-written C references.
