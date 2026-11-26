# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `cignal build --manifest` writes `<stem>.cignal.json`: file names, C symbol prefix, I/O counts, parameters and events (spec 6). gen-dsp reads it to wrap a kernel as a CLAP plugin without importing cignal. Reading `K_info` at run time was the alternative, but a generator needs parameter names before anything is compiled.

- `K_set_param_immediate(k, index, v)` and `Kernel.set_param_immediate` set a smoothed parameter without a glide, for a preset or state load. `set_param_immediate` is a reserved method name.

- A `CIGNAL.BAD_PARAM` warning for `param()` on a sub-kernel, whose range and smoothing never apply.

- Spec 5.2 lists the float operations where CPython raises or goes complex and C returns inf or NaN. They are outside the contract. A complex `tick` output now raises a `ValueError` that names the cause.

- `glide(value, target, coef)`: one step of the `smooth=` glide, snap included, for state that is not a parameter. A sub-kernel has no parameters, so its glides copied the snap rule by hand, or stalled short of the target in the float build. `smooth=` on plain fields was the alternative; it hides when the step runs, and a voice steps only while it plays (spec 5.2).

- `cignal.kernel.CHECK_I32`: `Kernel.process` raises when a `tick` leaves an `i32` field out of range, where C wraps. The tests set it; no fixture trips it. An AST rewrite checking every operation was the alternative; it would run code other than the user's (spec 5.3, 13.3).

### Changed

- The kernel struct's fields are no longer ABI. Hosts allocate it by size and reach it through parameters and events only, so the compiler may change its layout, for example to a struct-of-arrays `Array[K, N]` (spec 6).

- A sub-kernel's `tick` may take and return any scalar types. The `f32` rule exists for host buffers, which only the exported kernel sees.

- `Array[K, N]` of sub-kernels is stored as one struct of arrays, with N lanes per member. An `Array` of scalars inside a voice puts the lane last, so the voices' copies of one element are adjacent. Under clang, an 8-biquad bank went from 10.8 to 3.8 ns/frame at `-O2`, Poly from 21.7 to 15.2, and 8 voices of a 4-tap FIR from 19.0 to 3.5; output is bit-identical. Freeverb is 5% slower at `-O2` and level at `-O3`. GCC 14 does not vectorise the voice loop and gains nothing; the bank is 12% slower at its `-O3`. `cignal build --layout aos` keeps one struct per element, and the manifest records the layout. It is a generation option, not an `#ifdef` on the C compiler: the shim and the kernel may be built by different compilers and must agree on the struct. SoA as opt-in was the alternative; it would cost every clang user the gain to spare GCC 12% on one kernel (`docs/dev/soa.md`).

- CI runs the tests on macOS too (clang, Apple libm, Pd and SuperCollider from Homebrew).

- `make bench-layout` times every `Array[K, N]` fixture in both layouts, alternating the builds, and fails if their outputs differ. CI runs it on x86-64 Ubuntu with GCC and clang and on macOS, with the table in the job summary; the timings do not gate the job, since shared runners are noisy.

- `--pd-name`, `--pd-include`, `--sc-name` and `--sc-include` are errors without the matching `--shim`. They were ignored.

### Fixed

- SuperCollider: an event now sees a parameter set in the same block, as in Pd. A unit command runs before the block's inputs exist, so the event used the previous block's value. The shim queues up to 32 events and resets per block and runs them once it has read the inputs. Reading the pending value from SuperCollider's `Control` unit was the alternative; it works only for an input fed directly by a `Control`.

- A smoothed parameter set before the first block glided from its default. It now starts at the value set between `K_init` and the first `K_process`. Shims set parameters after `K_init`, so each new Pd object or SC synth glided from the default to its creation value, and Pd repeated the glide after a sample-rate change. A flag in the struct does this, so no host needed a change; requiring hosts to call `K_set_param_immediate` was the alternative, and a host that forgot would still glide.

- Evaluation order. C leaves the order of operands and call arguments unspecified, and every `i32` `+ - *` is a helper call, so `self.n + self.bump()` could read `self.n` after the call. A statement that calls a method and touches `self` elsewhere now hoists each state read and call into a temporary, in Python's order. An order-sensitive operand of `and`, `or` or `if`-`else` is an error. A `range` start with a non-constant stop was evaluated after the stop; it now goes first. Rejecting statements with two calls was the alternative; it rejects `self.a.tick() + self.b.tick()` (spec 5.6).

- `i32(x)` of a float was a raw C cast, undefined for NaN and out-of-range values. It now saturates and maps NaN to 0, as event arguments do. CPython raises there, so neither result is part of the contract (spec 5.2).

- A NaN parameter value passed the setter's clamp and stayed in filter state until `reset`. Setters now map it to `lo`, in C and in `Kernel.set_param`, as table lookups and `read_frac` already did.

- A file stem with `-` or `.` (`my-gain.py`) gave a header guard that does not compile. Those characters now map to `_`.

- An int constant beyond double range (`10**400`) crashed the compiler. It now reports `CONSTANT_RANGE`.

- `--pd-name 2pole~` was accepted and failed in the C compiler. Pd names must now be ASCII identifiers plus `~`.

- `--shim sc` wrote an sclang class file that sclang rejects when a parameter was named `Gain`, `var` or `in0`. The build now reports these names (spec 8).

- The shims now flush denormals on 32-bit ARM with a VFP unit (Bela, 32-bit Raspberry Pi OS), where FTZ was a no-op. This is checked by cross-compiling only.

- The FTZ `asm` statements now clobber memory. Without that, clang moved inlined floating-point work out of the flushed window; a multiply ran after the host's mode was restored. The shims call `K_process` in another file, which stayed inside, so no current build was affected.

- `Kernel.process(..., 0)` returned `[]`. It now returns one empty list per output, as C writes zero frames to each channel.

- `make clean` deleted the extension modules in `.venv`.

- Two concurrent `cignal fetch-headers` runs could fail with a traceback: the second rename met the first's directory. It now succeeds when the cache holds the verified files, and any other install failure is a fetch error.

- `make docs` is gone: `docs/` has no Sphinx project.

- `pyproject.toml` no longer lists Python 3.15 or PyPy, which CI does not test, and its keywords were placeholders.

- `make test-linux` did not copy `thirdparty/`, so 7 `test_headers` tests failed in the container.

## [0.1.0]

### Added

- Module-level tuples can be indexed, as spec 3.5 already stated: a constant index folds, and any other index is proven in range like an `Array` index and reads a `static const` array. The element type is `i32` when every element is an int, so `COMBS[i]` keeps its integer meaning; one float makes the tuple `f32`.

- `Array[K, N]` of sub-kernels and `Array[Delay[T, M], N]` of delay lines, used through a proven index: `self.combs[i].tick(x)`. A file whose kernels are held only through such arrays no longer reports "several top-level kernels".

- Gigaverb fixture, and long-tail tests for cases with `Case.tail_s` set (spec 9.1). Over 10 s, Gigaverb's float build measured 68 dB, because its decay rate was computed in `f32`. The 4800-frame test had passed above 100 dB, and never wrapped its delay lines.

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

- A local proves an index when every assignment to it does: `best = i` inside `for i in range(4)` makes `self.voices[best]` valid after the loop. Voice allocation needed this; it was written `self.voices[best % 4]`. The range is the union over all assignments, ignoring control flow, plus 0 for a hoisted declaration.

- Pd: a message `<param> value` on the left inlet sets a parameter, as its inlet does. The 8-voice synth has 19 parameters, so 19 inlets; this lets a patch reach them by name. A parameter named like a Pd selector (`float`, `reset`, ...) keeps only its inlet, and an event with a parameter's name is an error.

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
