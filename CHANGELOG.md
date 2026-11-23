# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Module-level tuples can be indexed, as spec 3.5 already stated: a constant index folds, and any other index is proven in range like an `Array` index and reads a `static const` array. The element type is `i32` when every element is an int, so `COMBS[i]` keeps its integer meaning; one float makes the tuple `f32`.

- `Array[K, N]` of sub-kernels and `Array[Delay[T, M], N]` of delay lines, used through a proven index: `self.combs[i].tick(x)`. A file whose kernels are held only through such arrays no longer reports "several top-level kernels".

- Freeverb and Moog-ladder fixtures, written to find what a real kernel needs; the three items above came from them. The float-build SNR threshold is now per case: the ladder self-oscillates at `res=1.1`, where the builds drift in phase, and passes at 90 dB (spec 9.1).

## [0.0.2]

### Added

- `cignal fetch-headers [pd] [sc]` downloads `m_pd.h` (Pd 0.57.0) and the SuperCollider 3.14.1 plugin headers into the user cache. Each file is pinned by commit and SHA-256, and a failed fetch installs nothing. The build uses the cache after `$PD_INCLUDE`/`$SC_INCLUDE` and installed copies. The headers are downloaded, not shipped, so the package stays MIT: the SuperCollider headers are GPL-2.0-or-later. The macOS SuperCollider app has no plugin headers, so before this an SC build there needed a source checkout.

- `thirdparty/` holds the same headers for the tests, which fall back to it. It is in neither the wheel nor the sdist.

- Built-ins `abs`, `min` and `max`, and `math.fmod`, `math.log2`, `math.log10` and `math.copysign`. `min` and `max` compile to `b < a ? b : a` helpers, not `fmin`/`fmax`, which differ from CPython on NaN and signed zeros: `min(nan, 1.0)` is `nan` and `min(0.0, -0.0)` is `0.0`. `abs` on `i32` wraps at `INT32_MIN`.

- CI installs Pd and SuperCollider, so the host tests run on Ubuntu x86-64 (GCC, Pd 0.56, SuperCollider 3.13). A missing host fails the job, where the tests would skip.

### Fixed

- `scsynth` and Pd are found in app bundles up to two folders below `/Applications`, such as `/Applications/Studio/supercollider-3.14.1/`.

- SuperCollider tests on macOS. The scsynth plugin path was hard-coded to Linux's `/usr/lib/SuperCollider/plugins`, so `Out` did not load and scsynth rendered silence. The tests also find `sclang` in the same app bundle as `scsynth`.

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
