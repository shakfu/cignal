# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Module-level tuples can be indexed, as spec 3.5 already stated: a constant index folds, and any other index is proven in range like an `Array` index and reads a `static const` array. The element type is `i32` when every element is an int, so `COMBS[i]` keeps its integer meaning; one float makes the tuple `f32`.

- `Array[K, N]` of sub-kernels and `Array[Delay[T, M], N]` of delay lines, used through a proven index: `self.combs[i].tick(x)`. A file whose kernels are held only through such arrays no longer reports "several top-level kernels".

- Freeverb, Moog-ladder and 8-voice synth fixtures, written to find what a real kernel needs; the items above came from them, and the synth's gaps are in `TODO.md`. The float-build SNR threshold is now per case: the ladder self-oscillates at `res=1.1`, where the builds drift in phase, and passes at 90 dB (spec 9.1).

- `self.target("cutoff")` reads the value a parameter glides towards. It changes only between blocks, so `control` may read it, where reading the smoothed parameter is an error. On an unsmoothed parameter it reads the parameter. `target` is now a reserved method name.

- `Array` parameters: `cutoffs: Array[f32, 4] = param(1000.0, lo=20.0, hi=20000.0, smooth=0.002)` gives each voice its own value. Hosts see 4 scalar parameters, `cutoffs_0` to `cutoffs_3`, so Pd gets one inlet and SuperCollider one input per element, and neither shim changed. The alternative, an indexed setter in the ABI, needed new routing in every host. `K_set_cutoffs(k, element, v)` takes the element index.

- `f64` parameters. The host still sets a `sample`, which is widened; the value, its target and its glide stay `double` in the float build. The synth fixture's smoothed `f64` pitch measures 115 dB of float SNR, against 98 dB as `f32`, and replaces a hand-written glide. Keeping all smoother state in `double` was the alternative: it changes every smoothed kernel's output and still rounds each read to `f32`.

- Events: a method marked `@event` and named `on_<name>` is a function hosts call between blocks. Pd gets a message `note 60 0.8`, SuperCollider a unit command, and Python `kernel.event("note", 60, 0.8)`. Parameters cannot express notes: re-sending a held gate does nothing, and a note-off and note-on in one block collapse to one value. Events also let the kernel allocate voices (`tests/fixtures/synth.py`). A "changed" query on parameters was the alternative; it fixes only retriggering (`docs/dev/events.md`).

  ```python
  @event
  def on_note(self, note: i32, velocity: f32) -> None: ...
  ```

  Host values convert to `i32` by truncation, saturating, with NaN as 0, since a C cast is undefined there. `event` is a reserved method name, and so are event names Pd uses itself (`bang`, `float`, `reset`, ...).

### Changed

- The ABI adds `K_event(k, index, args)` and one `K_on_<name>` per event, and `cignal_kernel_info` ends with `n_events` and `events`.

- Under GCC, a `range` loop with constant bounds and 2 to 16 iterations is fully unrolled. GCC `-O2` ran each sub-kernel of an `Array[K, 8]` in turn, so their dependency chains did not overlap: an 8-biquad bank went from 25.4 to 6.5 ns/frame, Freeverb from 13.1 to 11.2. clang already unrolled the bank, and forcing it slowed an 8-voice ladder by 12%, so the macro is empty there (`docs/dev/simd.md`).

- Kernels without inputs run the differential tests on one signal, not five, so the suite reports no skips.

### Fixed

- The sclang class of a kernel with neither inputs nor parameters had `*ar { || ...`, which sclang rejects.

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
