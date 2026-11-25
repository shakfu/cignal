# cignal specification

Status: draft, 2026-11-22.

cignal compiles DSP kernels written in a static subset of Python to C99. The output targets real-time audio hosts: plugins and unit generators (ugens). The same kernel source runs unchanged under CPython as the reference model.

[Vult](https://github.com/vult-dsp/vult) is the model. cignal keeps Vult's execution model: a context struct per stateful unit, explicit init, per-sample processing, build-time tables, host templates. It replaces Vult's language with Python syntax, and Vult's `@[table]` and constant folding with build-time evaluation in CPython.

## 1. Goals

1. Write a kernel once in Python. Test, plot and design it with NumPy and SciPy.
2. Emit C99 that is real-time safe: no allocation, locks, I/O or exceptions in the processing path.
3. Emit C whose hot loop has no checks or calls that stop the C compiler from optimising it.
4. Show that the C matches the Python reference, through automated differential tests.
5. Keep the host API out of the kernel. One kernel ABI; one thin shim per host.

## 2. Non-goals (v1)

- C++, Rust or LLVM output. C++ and Rust hosts call the C ABI.
- NumPy arrays or array expressions inside kernels.
- FFT, oversampling, multi-rate or variable block-size processing inside a kernel.
- Fixed-point output. See section 13.
- Dynamic allocation of any kind, including at init. All sizes are build-time constants.
- Strings, lists, dicts, sets, exceptions, generators, closures, recursion, `print` in kernels.
- GUIs and plugin metadata beyond parameter descriptors.

## 3. Source model

### 3.1 Types

The `cignal` package defines the annotation types. At runtime each is an alias, so mypy and CPython accept kernels unchanged. The compiler distinguishes them by name.

| cignal type | Runtime alias | C type | Notes |
|-|-|-|-|
| `f32` | `float` | `float` | The default sample type |
| `f64` | `float` | `double` | For accumulators and coefficient maths |
| `i32` | `int` | `int32_t` | Wraps two's-complement in C (section 5.3) |
| `bool` | `bool` | `bool` (`<stdbool.h>`) | |
| `Array[T, N]` | fixed-length list subclass | `T name[N]` | `N` is a build-time constant |
| `Delay[T, N]` | ring-buffer class | struct with `T buf[N]` and a write index | Section 3.6 |

- Bare `float` and `int` annotations are errors (`CIGNAL.IMPLICIT_PRECISION`). This prevents silent `double` maths in an `f32` kernel.
- Mypy treats `f32` and `f64` as the same type, so `cignal check` enforces precision, not mypy.

### 3.2 Kernels

A kernel is a class derived from `cignal.Kernel`. Its fields are its state.

```python
import math

from cignal import Kernel, f32, param


class Biquad(Kernel):
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
    filt: Biquad                       # sub-kernel: one context, initialised automatically

    def control(self) -> None:
        w = 2.0 * math.pi * self.cutoff / self.sample_rate
        alpha = math.sin(w) / (2.0 * self.q)
        a0 = 1.0 + alpha
        c = math.cos(w)
        self.filt.b0 = (1.0 - c) / 2.0 / a0
        self.filt.b1 = (1.0 - c) / a0
        self.filt.b2 = self.filt.b0
        self.filt.a1 = -2.0 * c / a0
        self.filt.a2 = (1.0 - alpha) / a0

    def tick(self, x: f32) -> f32:
        return self.filt.tick(x)
```

Rules:

- Every field is declared at class level with an annotation. Assigning an undeclared `self.x` is an error (`CIGNAL.UNDECLARED_STATE`).
- Scalar fields need a default. `Array`, `Delay` and sub-kernel fields are zero-filled or default-constructed.
- `self.sample_rate: f32` is predefined and set by init.
- A sub-kernel field gives one context per field. This replaces Vult's call-site contexts (section 12.1). `Array[K, N]` gives `N` contexts (section 3.6).
- Inheritance between kernels is out of scope for v1 (`CIGNAL.UNSUPPORTED_SYNTAX`).

### 3.3 Kernel methods

| Method | Required | Called | C function |
|-|-|-|-|
| `init(self) -> None` | no | once, after fields take their defaults and `sample_rate` is set | `K_init` |
| `control(self) -> None` | no | once per block, before the first `tick` | inside `K_process` |
| `tick(self, *inputs: f32) -> f32 \| tuple[f32, ...]` | yes, on exported kernels | once per sample | `static inline` in the `.c` file |
| `@event` `on_<name>(self, ...) -> None` | no | by the host, between `K_process` calls | `K_on_<name>`, and `K_event` by index |
| other methods | no | by kernel code only | `static inline` |

- `tick`'s parameters set the number of audio inputs. Its return type sets the number of outputs, at least one.
- Init calls down into sub-kernels in field order before the parent's own `init` runs.
- `control` and `tick` do not call down. The parent calls its sub-kernels explicitly.
- `control` runs at the start of every `K_process` call. Its effect is independent of the host's block size only if it computes from parameters and from fields `tick` does not change. The block-pattern tests (9.2) rely on this for the fixtures.
- Methods of a kernel and its sub-kernels may call each other. The call graph must be acyclic (`CIGNAL.RECURSION`).
- `process`, `set_param`, `set_param_immediate`, `reset`, `target` and `event` are reserved method names.
- An event method needs both `@event` and the `on_` prefix; either alone is an error. Hosts see `<name>` without the prefix. Names Pd uses itself (`bang`, `float`, `list`, `reset`, ...) are reserved. Arguments are `f32`, `f64`, `i32` or `bool`. On a sub-kernel, an event method is a plain method its parent may call. The design is in [events.md](events.md).
- An event may read anything, smoothed parameters included: it runs at a frame the host chose, not at a block start, so its effect does not depend on block size. Hosts get sample-accurate events by splitting `K_process` at the event's frame.
- `K_reset` restores every field except the parameters and the sample rate, clears arrays and delays, resets sub-kernels (their `param` fields included, since those are plain fields), then re-runs `init`. `Kernel.reset()` does the same in the reference. There is no separate user hook: `init` already runs on reset.
- Methods that nothing reachable from `tick`, `control` or `init` calls are type-checked but not emitted, since `-Wunused-function` rejects an unused static function.

### 3.4 Parameters

`param(default, lo=, hi=)` marks a field as a host-controlled parameter. It is honoured only on the exported kernel. On a sub-kernel it is a plain field with a default.

- Parameters are `f32` or `f64`, and read-only in kernel code. Hosts set every parameter as a `sample`; an `f64` parameter widens the value, then clamps and smooths it in `double` in both builds. A pitch needs this: a phase integrates every rounding of its frequency, and an `f32` smoothed pitch measured 98 dB on the float build of the synth fixture, against 115 dB for `f64`.
- Setters clamp to `[lo, hi]`, with NaN mapped to `lo`. Stored, a NaN would stay in recursive state until `reset`, and a SuperCollider input can carry one. `Kernel.set_param` clamps the same way in the reference.
- Parameters are set between blocks. Signal-rate parameters would make each one a buffer in the ABI; they are out of scope until a kernel needs them.
- `cutoffs: Array[f32, N] = param(...)` declares N parameters with one range, default and smoothing, read through a proven index. Hosts see N scalar parameters named `cutoffs_0` to `cutoffs_{N-1}`, so the ABI and the shims are unchanged; a field with one of those names is an error. The named setter takes an element index and ignores one out of range: `K_set_cutoffs(k, element, v)`. Polyphony needs this: without it, voice 3's cutoff in an `Array[Voice, 8]` is one of 8 hand-declared parameters.
- Setters run on the audio thread or between `K_process` calls. There are no atomics in v1. Any cross-thread handoff is the shim's job.
- `param(..., smooth=seconds)` (M3) makes the field a one-pole glide towards the last value set. It steps once per sample, before `tick`. The setter changes only the target, except between `K_init` and the first `K_process`, where it sets the value too. Shims set parameters after `K_init`, so without this every new instance, and every Pd sample-rate change, glides from the default. `K_set_param_immediate` sets value and target at any time, for a preset or state load. `Kernel.set_param` and `Kernel.set_param_immediate` do the same in the reference. The coefficient is `1 - exp(-1 / (seconds * sample_rate))`, computed in `double` at init. `kernel.smooth_step` defines the step, and the C repeats it operation for operation.
- The step snaps to the target when it makes no progress (`value + coef * d == value`), or when `|d| < 1e-20`. Without the first condition, a float smoother stalls once `coef * d` is below half a unit in the last place: the float build of the glide fixture stopped at 439.99927 for a target of 440, while the double build converged. The second condition snaps a glide towards 0 before it reaches denormals.
- `control` may not read a smoothed parameter, directly or through methods it calls (`CIGNAL.BAD_PARAM`). The value it saw would depend on where block boundaries fall. Read it in `tick`.
- `self.target("name")` (`self.target("name", i)` for an `Array` parameter) reads the last value set, which changes only between blocks, so `control` may read it. For an unsmoothed parameter it is the parameter itself, so adding or dropping `smooth=` does not break the call. The name must be a string literal naming a parameter of the exported kernel.
- `reset` keeps smoothed parameters as they are, including a glide in progress.

### 3.5 Free functions and constants

- A module-level `def` with full annotations, called from kernel code, becomes `static inline cignal_fn_<name>`. It has no `self` (`CIGNAL.UNKNOWN_NAME`) and may call tables, `math`, and other module-level functions.
- A module-level function is lowered only when kernel code calls it. Build-time helpers, such as a coefficient design using SciPy, can sit in the same file without being valid kernel code.
- The call graph, methods and functions together, must be acyclic (`CIGNAL.RECURSION`).
- Module-level names that are not kernels, functions or imports are build-time constants (section 4). They may be numbers, booleans, or tuples of them.
- A tuple is indexed like an `Array` and is read-only. Its element type is `i32` if every element is an int, `f32` if any is a float, and `bool` if all are bools. Mixing bools and numbers is not a tuple constant.
- A constant index folds to the element, which then acts as a literal. A negative constant counts from the end, as in Python. Any other index follows section 5.4 and reads a `static const` array, emitted only for tuples indexed that way.

### 3.6 `Array` and `Delay`

- Fields are written `taps: Array[f32, 16]` and `line: Delay[f32, 4800]`, with no default. `N` is evaluated in CPython, so any build-time expression works. Arrays are zero-filled.
- `Array[T, N]`: `T` is any scalar type, `1 <= N <= 2**24`. Indexing must be provably in `[0, N)` (section 5.4). An array cannot be used as a value.
- `Array[K, N]` for a kernel class `K` holds `N` sub-kernels, and `Array[Delay[T, M], N]` holds `N` delay lines. Elements are used through a proven index: `self.combs[i].tick(x)`, `self.lines[i].read(d)`. They initialise and reset in index order, at the field's place in field order. Freeverb's 8 combs need this: without it they are 8 named fields and unrolled loops.
- `Delay[T, N]`: `T` is `f32` or `f64`, `2 <= N <= 2**24`. `kernel.Delay` defines the semantics; the C helpers follow it operation for operation.
  - `write(x)` stores the newest sample.
  - `read(d: i32)` returns the sample written `d` writes ago, with `d` clamped to `[1, N]`. `read(1)` is the newest, so `read(D)` before `write(x)` delays by exactly D samples.
  - `read_frac(d: f64)` interpolates linearly, with `d` clamped to `[1, N-1]` and NaN mapped to 1. NaN-to-integer is undefined behaviour in C. The position is `f64` for any element type: at 1,600 samples an `f32` position resolves only 1.2e-4 samples, which measured as an 86 dB float SNR on the echo fixture. An `f32` argument is widened without loss.
  - `clear()` zeroes the buffer.
- The ring index wraps with compare-and-add, which works for any `N`; a power-of-two mask would save nothing measurable.
- The delay length is a build-time constant. To size by time, compute `N` at build time from a declared maximum rate, for example `MAX_DELAY = int(2.0 * 96000)`.

### 3.7 Tables

```python
@table(size=256, lo=0.0, hi=1.0)
def shape(x: f32) -> f32:
    return math.tanh(4.0 * x)
```

- At import time, CPython evaluates `shape` at `size + 1` evenly spaced points in `f64`. The body can use anything, SciPy included, because it never becomes C. The C code gets `static const float cignal_table_shape[257]` and a lookup.
- The lookup clamps to `[lo, hi]` (NaN to `lo`), scales to a position, and interpolates linearly. `kernel.Table.__call__` defines it, and the C repeats it operation for operation, so the double build matches CPython exactly. Kernels call the table like a function.
- `shape.function(x)` calls the exact function, for measuring table error.
- The table maps `f32` to `f32` or `f64` to `f64`. `1 <= size <= 2**24`, `lo < hi`, and every sampled value must be finite (`CIGNAL.BAD_TABLE`).
- Only tables that reachable code calls are emitted.
- Vult tables use second-order interpolation. Linear is the v1 choice because it is cheaper and easier to check. Higher orders are a later option.

## 4. Build-time evaluation

`cignal build` imports the kernel module in CPython. It then:

1. Reads module constants, `param` metadata, `Array`/`Delay` sizes and `@table` contents from the live objects.
2. Re-parses the source with `ast`, for the bodies that become C.
3. Folds any expression made only of build-time constants and `math` calls on them. The result is emitted as a literal.

This replaces Vult's constant folding and `@[table]`. It gives the full Python language at build time, including SciPy for filter design.

Importing the module runs the user's code. That is expected for a build tool and is documented, not sandboxed.

## 5. Kernel-body semantics

The contract: compiled with `f64` for `f32`, the C output matches CPython within the tolerances in section 9. Where Python and C disagree, cignal emits Python's semantics, and the C-native form is available as an explicit call.

### 5.1 Allowed syntax in kernel methods and free functions

- Assignment, annotated assignment, augmented assignment, tuple unpacking from tuple-returning calls.
- `if`/`elif`/`else`, conditional expressions.
- `for i in range(a)`, `range(a, b)` and `range(a, b, step)` where `step` is a non-zero build-time constant, with `break` and `continue`. Bounds are evaluated once, as in Python. The loop variable must be a new name and is read-only in the body. `for ... else` is rejected.
- `while`, with `break` and `continue`, and no `else`. It compiles with a warning (`CIGNAL.UNBOUNDED_LOOP`) because termination is not checked. Warnings do not stop the build; `cignal check` and `cignal build` print them, and `CKernel.warnings` carries them.
- Calls to free functions, kernel methods, sub-kernel methods, `Delay` methods, and the `math` subset in 5.2.
- `return`. Tuple returns become a small struct returned by value.

Anything else is rejected with a diagnostic that names the construct.

### 5.2 Numeric operations

| Python | C (`f32`) | Notes |
|-|-|-|
| `+ - *` on floats | same | |
| `+ - *` and unary `-` on `i32` | `cignal_add_i32` etc. | Wrap through `uint32_t` (section 5.3) |
| `/` | `/` | Floats only. `i32 / i32` is an error; cast first. |
| `//` on `i32` | floor-division helper | Python floors; C truncates. A divisor of 0 gives 0, where CPython raises; -1 wraps like negation. |
| `//` on floats | CPython's `float_floor_div`, step for step | Bit-identical to CPython |
| `%` on `i32` | floor-mod helper | Sign follows the divisor, as in Python. A divisor of 0 gives 0, where CPython raises; -1 gives 0 because `INT32_MIN % -1` traps in C. |
| `%` on `i32` by a constant `M > 0` | `& (M - 1)` for a power of two, else `r = a % M; r < 0 ? r + M : r` | Exact, without the guards |
| `%` on floats | `fmodf`, then adjusted as CPython's `float_rem` does | The result takes the divisor's sign, including for zero. Bit-identical to CPython. |
| `x ** y` on floats | `powf` / `pow`, even for a constant integer `y` | CPython computes float `**` with libm `pow`; `x * x * x` would round differently |
| `a ** k` on `i32` | exponentiation by squaring, wrapping | `k` must be a non-negative constant: for negative `k` Python returns a float |
| `math.sin cos tan tanh exp log log2 log10 sqrt fabs floor ceil atan2 pow fmod copysign` | `f`-suffixed `<math.h>` functions | `double` variants for `f64`. CPython calls the same libm functions. |
| `math.pi`, `math.tau`, `math.e` | literals | |
| `& \| ^ ~` on `i32` | same | Equal on two's complement. Non-atom operands are parenthesised (GCC `-Wparentheses`). |
| `<<`, `>>` on `i32` | helpers | `<<` shifts through `uint32_t`. `>>` floors, as Python does, without shifting a negative value. Counts outside 0..31, where Python's result leaves `i32` or raises, give the limiting value. |
| `and or not`, comparisons | same | `a < b < c` becomes `a < b && b < c`, with Python's short-circuit order. Middle operands are evaluated twice, so they may not contain calls. |
| `abs(x)` | `fabsf` / `fabs`; a wrapping helper on `i32` | `abs(INT32_MIN)` wraps to `INT32_MIN`, where C's `abs` is undefined |
| `min(a, b, ...)`, `max(a, b, ...)` | helpers, nested left to right: `b < a ? b : a` | CPython keeps the first argument unless a later one is strictly less (greater). `fmin`/`fmax` differ on NaN and signed zeros: `min(nan, 1.0)` is `nan`, `min(0.0, -0.0)` is `0.0`. Only when the module does not rebind the name. |
| `f32(x)`, `f64(x)` | casts | |
| `i32(x)` of a float | `cignal_to_i32` | Truncates towards zero in both languages. A C cast is undefined for NaN and out-of-range values (C99 6.3.1.4), so C saturates and maps NaN to 0, as event arguments do. CPython raises on NaN and inf, and leaves `i32` range otherwise: both are outside the contract (5.3). |

- Where C returns inf or NaN, CPython raises or leaves the reals, so the reference gives no value to compare. These cases are outside the contract:

  | Expression | CPython | C |
  |-|-|-|
  | `x / 0.0`, `x // 0.0`, `x % 0.0`, `0.0 ** -1.0` | `ZeroDivisionError` | inf or NaN |
  | `math.log(0.0)`, `math.sqrt(-1.0)`, `math.fmod(x, 0.0)`, `math.pow(-1.0, 0.5)`, `math.sin(inf)` (and `cos`, `tan`), `math.floor(nan)` | `ValueError` | -inf or NaN |
  | `math.exp(1000.0)`, `math.floor(inf)` | `OverflowError` | inf |
  | `(-1.0) ** 0.5` | a complex number; `Kernel.process` raises when it reaches an output | NaN |

  Float overflow in `+ - *` gives inf in both, and `inf - inf` NaN in both. `math.floor` and `math.ceil` return `int` in CPython and a float in C; the values agree for finite input.
- Numeric literals take the type their context needs: `0.5` in an `f32` expression is emitted as `0.5f`.
- Mixing `i32` and float variables without a cast is an error (`CIGNAL.MIXED_NUMERIC`), as in Vult.
- Mixing `f32` and `f64` variables without a cast is an error. Literals are exempt.

### 5.3 Integer overflow

`i32` `+`, `-`, `*`, unary `-`, `**` and `<<` wrap two's-complement in C, through `uint32_t` arithmetic. Signed overflow is undefined behaviour in C; unsigned overflow is not. The conversion back to `int32_t` is implementation-defined, and GCC and clang wrap. Before the M2.5 work the generated code used plain signed operators, so this section described behaviour the code did not have.

CPython ints do not wrap. A kernel whose intermediate values leave `i32` still matches CPython when it masks the result back into range, as the 31-bit LCG in `tests/fixtures/ops.py` does: its multiply overflows on every step and matches bit for bit. A kernel whose stored values leave range is outside the contract. With `cignal.kernel.CHECK_I32` set, as the tests set it, `Kernel.process` raises `OverflowError` after any `tick` that leaves an `i32` field or `Array[i32]` element out of range, in the kernel or its sub-kernels. `Delay[i32, N]` lines are not checked: scanning them costs N per sample.

### 5.4 Index safety

An `Array[T, N]` index, or a runtime index into an `N`-element module tuple, is accepted if a local interval analysis shows it is in `[0, N)`. Accepted forms:

- a build-time constant;
- the variable of a `range` loop with constant bounds, over the values the range actually takes;
- `e % M` for a constant `M > 0`: floor-mod lands in `[0, M - 1]` whatever `e` is;
- a local whose every assignment is one of these forms, over the union of their ranges and 0, the value of a hoisted declaration. The union ignores control flow, so `best = i` in a loop proves `self.voices[best]` after it. An augmented assignment, a tuple unpack or a cycle (`k = k + 1`) leaves the local unproven;
- `+` and `-` of those, by interval arithmetic. Both are exact mod 2**32, so a result inside `[0, N)` holds under C's wrapping too.

An index may not contain calls. `a[i] += x` evaluates the index twice in C, so a call in it would run twice. A local or parameter is never treated as a constant, even if a module constant has the same name. Anything else is rejected (`CIGNAL.UNPROVEN_INDEX`). The remediation names the accepted forms. There are no runtime bounds checks and no SMT solver. Making the safe forms the only forms keeps the hot loop free of checks.

### 5.5 Identifiers

Python names that are C keywords, start with `_` or `cignal_`, or end with `_` get a `_` suffix. That keeps the mapping one-to-one: `int` becomes `int_`, so a user's `int_` becomes `int__`. Generated identifiers start with `cignal_` and never end in `_`, so they cannot collide with user names. Every symbol derived from a kernel is named after it (`Lowpass_init`, `Lowpass` for the struct), sub-kernels included. `--prefix P` (`compile_kernel(..., prefix=P)`) starts all of them, and the header guard, with `P`. Without it, two generated kernels that hold sub-kernels of the same name cannot share a program: their headers share an include guard and the linker finds duplicate symbols. `tests/test_codegen.py` shows both outcomes.

### 5.6 Evaluation order

CPython evaluates operands, call arguments and `range` bounds left to right. C leaves operand and argument order unspecified (C99 6.5p3, 6.5.2.2p10), and `i32` arithmetic lowers to helper calls, so its operands are arguments too. A non-constant `range` stop is hoisted into a temporary, so a non-constant start is hoisted before it.

A statement is order-sensitive when it calls a method of `self` or a sub-kernel and mentions `self` elsewhere. In such a statement, every field read and method call is hoisted into a `const` temporary in Python's order; other statements are emitted unchanged. `and`, `or`, `if`-`else` and chained comparisons are C sequence points, so each is hoisted whole, with its operands inline. An operand that is itself order-sensitive is an error, since hoisting it would evaluate it unconditionally. A `while` test with temporaries becomes `for (;;) { ...; if (!test) break; ... }`, and an `elif` test with temporaries nests in `else`. Delay reads and `target` do not write state, and module-level functions cannot reach it.

The alternative was to reject any statement with two calls. That rejects `self.a.tick() + self.b.tick()`, a common form. `tests/fixtures/order.py` covers each case above.

## 6. C ABI

`cignal build lowpass.py --kernel Lowpass` emits `lowpass.h` and `lowpass.c`. Both are C99 and depend only on `<math.h>`, `<stdint.h>`, `<stdbool.h>` and `<stddef.h>`.

```c
/* lowpass.h */
typedef struct Lowpass { /* fields, in declaration order */ } Lowpass;

void Lowpass_init(Lowpass *k, float sample_rate);
void Lowpass_process(Lowpass *k, const float *const *in, float *const *out, int n);

void Lowpass_set_cutoff(Lowpass *k, float v);
void Lowpass_set_q(Lowpass *k, float v);
void Lowpass_set_param(Lowpass *k, int index, float v);
void Lowpass_set_param_immediate(Lowpass *k, int index, float v); /* no glide */

/* Events: one function per `on_<name>`, and a dispatcher taking host values by index. */
void Lowpass_event(Lowpass *k, int index, const float *args);

extern const cignal_kernel_info Lowpass_info;
```

```c
/* Guarded by CIGNAL_INFO_TYPES, so several generated headers can share a translation unit. */
typedef struct { const char *name; double lo, hi, def; } cignal_param_info;
typedef struct { const char *name; int n_args; } cignal_event_info;
typedef struct {
    const char *name;
    int n_inputs, n_outputs, n_params;
    const cignal_param_info *params;
    size_t size;                /* sizeof the kernel struct */
    int n_events;
    const cignal_event_info *events;
} cignal_kernel_info;
```

ABI rules:

- **The struct definition is public.** Hosts allocate it with their own allocator (`RTAlloc`, `getbytes`, or a member of a C++ object) using `sizeof` or `info.size`. cignal never allocates.
- **`K_init` does everything needed before the first `K_process`.** It zero-fills, applies defaults, sets `sample_rate`, runs sub-kernel `init`, then runs `init`.
- **`n` may be any value from 0 up**, and may vary between calls. Kernel state carries across calls. Processing in blocks of 64 then 64 gives the same output as one block of 128 (section 9.2).
- **Input and output buffers may alias each other**, because Pd and other hosts process in place, and SuperCollider may reuse an input's buffer for an output. The buffer pointers are therefore not `restrict`. For each sample, the loop reads every input before it writes any output, which is safe for any pairing of channels.
- **Buffers must not point into the kernel struct.** `K_process` declares `self` as `restrict` (in the definition only; the header stays valid C++). Without it, every write to `out` might change a field, and the compiler reloaded and stored all state on every sample: the biquad ran 1.7x slower than hand-written C.
- **`K_event` converts each host value** to its argument's type: `i32` truncates towards zero, saturates, and maps NaN to 0, where a C cast is undefined; `bool` is `!= 0`. `kernel.event_i32` is the reference. An index out of range does nothing; `args` must hold `n_args` values.
- **The ABI is single-threaded**: one thread per kernel instance at a time.
- **The header has an `extern "C"` guard**, so C++ hosts link against the C symbols. The `.c` file is compiled as C, not C++: it uses compound literals, which C++ lacks.

`cignal build --manifest` also writes `<stem>.cignal.json`: the file names, `c_name`, sample type, I/O counts, parameters (`name`, `lo`, `hi`, `default`) and events (`name`, `args`). It lets a tool generate a wrapper without importing cignal or parsing C. gen-dsp reads it to build AU, CLAP, LV2, SuperCollider and VST3 plugins, and maps MIDI notes to events named `note(pitch, velocity)` and `off(pitch)`. A change that breaks a reader bumps its `format` field, now 1. The alternative, reading `K_info` at run time, gives the tool nothing at generation time, when plugin formats such as LV2 need parameter names.

## 7. Generated code quality

Every generated file must compile cleanly with `-std=c99 -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -Werror` on Clang and GCC. `-Wdouble-promotion` catches accidental `double` maths in an `f32` kernel.

Generated code must contain no references to `malloc`, `free`, `printf`, locks, `setjmp` or any symbol outside libm. The test suite checks the undefined symbols of the compiled object with `nm` (section 9.3).

`tick` and helpers are `static inline`. Parentheses follow C precedence, and a right operand keeps them at equal precedence, so the C evaluation tree is the Python one: float `a + (b + c)` is never reassociated. Locals are declared at their first assignment, unless that assignment is inside a branch; then the declaration is hoisted with a zero initialiser. Prototypes are emitted only for methods called before their definition. `K_process` is a loop over `n` that calls `tick`. Branches on build-time constants are folded. A `range` loop with constant bounds and 2 to 16 iterations is marked `CIGNAL_UNROLL(n)`, which forces full unrolling under GCC and is empty elsewhere ([simd.md](simd.md#unrolling)).

`-ffast-math` is not the default because reassociation breaks the comparison with the `f64` reference. `cignal build --fast-math` adds the flags to the build recipe it prints, and the tests then switch to the SNR check only.

Vectorisation note: a recursive filter (IIR) has a dependency between consecutive samples and does not vectorise along time. Stateless and FIR kernels can. For IIR kernels, `Array[K, N]` inside one kernel runs N voices in one instance: 2.2 to 3.4x faster than N instances, measured in [simd.md](simd.md). A struct-of-arrays layout for it adds 1.5 to 1.9x and stays bit-exact.

## 8. Host shims

A shim is a C template filled from `K_info`, produced with `string.Template` from the standard library. Each shim:

- maps host buffers and block size to `K_process`;
- maps host parameters, control messages or inlets to `K_set_param`;
- allocates the kernel struct with the host's allocator;
- sets flush-to-zero around `K_process` and restores the host's mode after (`ftz.h.in`): FTZ and DAZ in `MXCSR` on x86, `FPCR.FZ` on arm64, nothing elsewhere. Hosts differ: Debian's Pd 0.55 already runs kernels flushed, macOS Pd 0.57 does not. The probe kernel `tests/fixtures/denormal.py` computes `x * 1e-39 * 1e38`, whose denormal intermediate becomes 0 only when flushing is on. The kernel itself has no denormal handling, and the `driver` does not flush, so it stays comparable with CPython.

Shims, in order:

1. **`driver`** (v1, required): a standalone `main` that reads interleaved native-endian samples from stdin and writes them to stdout. The sample rate and a block-size pattern come from the command line. Tests and benchmarks use it.
2. **Pd** (v1, M0; parameters in M1). Inlets are the signal inputs, then one float inlet per parameter. A kernel without signal inputs takes its first parameter on the left inlet, as `osc~` does. Creation arguments set the parameters in order. The shim re-applies every parameter at the start of each block, which also restores them after a sample-rate change re-runs `K_init`. The object name defaults to the lowercased kernel name plus `~`, and `--pd-name` overrides it. A name that matches a Pd built-in (`biquad~`, `lop~`) loads the built-in instead, without an error. A message `<param> value` on the left inlet sets a parameter as its inlet does (`freqs_3 440`), so a kernel with many parameters needs no patch cords to each. A parameter named like a Pd selector (`float`, `reset`, ...) keeps only its inlet, and an event with a parameter's name is an error. Each event is a message on the left inlet, `note 60 0.8`; missing arguments are 0. The shim applies pending parameter inlets before an event, so the event sees every parameter sent before it. Pd parses `nan` in a message as a symbol, which arrives as 0.
3. **SuperCollider** (M4). `cignal build --shim sc` writes and compiles a UGen plugin (`.so` on Linux, `.scx` on macOS) and its sclang class file. Parameter names become sclang argument names, so the build rejects a name that starts with a capital, is an sclang keyword (`var`, `nil`, `thisProcess`, ...) or matches an input (`in0`).
   - The UGen name defaults to `Cignal` plus the kernel name, to stay clear of core UGens; `--sc-name` overrides it.
   - Inputs are the signal inputs, then one input per parameter, read once per block. A `reset` unit command (`/u_cmd node ugen "reset"`) calls `K_reset`.
   - Each event is a unit command: `/u_cmd node ugen "note" 60 0.8`. An event sees the parameters as of the last block, since inputs are read in the calculation function.
   - The constructor sets the first output sample to 0 instead of calling the calculation function once, as many UGens do. Running a sample in the constructor would process the first input sample twice.
   - The kernel struct sits inside the `Unit`, which scsynth allocates from its real-time pool; a delay line's buffer counts against that pool.
   - SuperCollider buffers are `float`; the shim refuses another sample type at compile time.
4. CLAP, Max: later milestones.

## 9. Testing

All tests run under `make test` with pytest. NumPy is a test-only dependency.

### 9.1 Differential tests

Each kernel fixture runs on standard signals, in three configurations:

| Run | Compared with | Pass criterion |
|-|-|-|
| C built with `f32 := double` and `-ffp-contract=off` | CPython reference | bit-identical |
| C built with `f32 := float` | the `double` C build | SNR >= 100 dB |
| Pd external, run with `pd -batch`, parameters set before DSP starts | the `float` driver, 64-sample blocks | bit-identical |
| Pd external with each change and reset sent mid-stream through `[delay F 1 samp]`, moved to the start of its block | the `float` driver with the same moved changes | bit-identical, after modelling `tabwrite~` (below) |
| SuperCollider UGen, rendered by `scsynth -N` from a hand-written SynthDef and OSC score, with each change and reset moved to the start of its block | the `float` driver with the same moved changes | bit-identical |
| The same UGen from a SynthDef that sclang builds with the generated class file (parameter changes, no resets) | the `float` driver | bit-identical |

Signals:

- impulse;
- step;
- a 440 Hz sine at 48 kHz;
- seeded white noise;
- silence followed by a burst, to test reset and denormals.

The first run checks the translation. The second checks single precision. The others check the shims.

Pd's `tabwrite~`, which the test patch records with, stores 0 for any sample with `|x| < 2**-63` or `|x| >= 2**65` (`PD_BIGORSMALL`). The glide fixture's decay to 0 showed it: every sample below 2**-63 came back as 0, every other sample exact. The Pd tests apply the same rule to the expected output (`support.pd_written`).

M0 calibration:

- The `double` build matched CPython exactly in all 15 fixture/signal pairs. The same IEEE operations run in the same order on the same libm.
- Without `-ffp-contract=off`, Apple clang fuses `a*b+c` into one FMA and the biquad drifts by 1.7e-15. Production builds may contract; reference builds must not.
- The lowest `float` SNR was the biquad's, at 106 dB.
- Parameter values in tests are rounded to `float32`, like the signals, so both builds see the same parameter. The rounding of a decimal like 0.0333 is an input difference, not arithmetic error.

M2 calibration: the lowest `float` SNRs are the biquad (106 dB), the resonant lowpass (107 dB) and the state-variable filter (115 dB). The echo reached 152 dB once its delay position was `f64`.

The SNR threshold is per case (`Case.min_snr_db`), 100 dB unless a measurement justifies less. A self-oscillating filter drifts in phase between the builds, so SNR measures the drift, not arithmetic error. The ladder's float SNR fell with resonance: 125 dB at `res=0.3`, 107 dB at 1.0, 96 dB at 1.1, 91 dB at 1.2. Its case reaches `res=1.1` and passes at 90 dB.

Cases with `tail_s` also run a burst followed by silence for that many seconds, with every parameter set to its `float32`-rounded default at frame 0. The `double` build must match CPython bit for bit. Every one-second window of the `float` build must meet the case threshold. The run is long enough to wrap delay lines longer than 4800 frames, and to let coefficient error compound through feedback. With its decay rate in `f32`, Gigaverb measured over 100 dB on 4800 frames, but 37.5 dB in its tenth second. Computing the rate in `f64` raised the worst window to 107 dB. A decay rate close to 1 needs `f64` for the reason a pitch does (3.4). Defaults are rounded because they can differ by input: unrounded, Echo's `time = 0.05` put its worst window at 64.6 dB, against 133.2 dB rounded.

Contraction can also change a non-finite result, not just the last bit. In the `float` build, clang fused `a * b - a * b`, whose product overflows, into `fma(a, b, -inf)`. That gives `-inf` where CPython gives NaN. The `builtins` fixture computes `inf` and `inf - inf` in separate statements, because C never contracts across statements. A kernel that relies on an exact `inf - inf` must do the same, or build with `-ffp-contract=off`.

Known failures are strict xfails with a reason, as in multigen's `KNOWN_FAILURES`. A fix fails the run until its entry is deleted.

### 9.1.1 Platforms

All tests pass, with the bit-exact CPython comparison, on:

| Platform | C compiler | libm | Python | Pd | SuperCollider |
|-|-|-|-|-|-|
| macOS 26, arm64 | Apple clang 21 | Apple libm | 3.14 | 0.57 | not installed: tests skip |
| Debian trixie, arm64 (`make test-linux`) | GCC 14.2 | glibc | 3.13.5 | 0.55.2 | 3.13.0 |
| Debian trixie, x86-64 under Rosetta (`make test-linux ARCH=amd64`) | GCC 14.2 | glibc | 3.13.5 | 0.55.2 | 3.13.0 |

Debian's sclang is the Qt build. The image sets `QT_QPA_PLATFORM=offscreen` and disables the QtWebEngine sandbox and GPU so it starts without a display.

The Linux runs found four portability problems, all now fixed:

- GCC's `-Wdiv-by-zero` rejected a division by `NIN` in a branch the driver never takes when `NIN` is 0. Clang does not warn there.
- GNU `nm -u` prints `U sincosf`, where macOS prints `_sincosf`.
- Python 3.12 and 3.13 evaluate class annotations eagerly. A kernel that names a later class needs `from __future__ import annotations`, and the string annotations then evaluate in `sys.modules[cls.__module__]`. The compiler now registers the module it loads, so build-time evaluation of `Array`/`Delay` sizes works.
- A test relied on 3.14's deferred annotations.

The x86-64 run uses Rosetta's translation, not x86 hardware.

### 9.2 Block-boundary tests

For every fixture, the output must be identical for:

- block sizes 1, 7, 64 and 128;
- a mixed sequence such as 3, 61, 64, 1;
- one run over the whole signal.

This catches state that does not persist across calls. That was multigen's C failure mode: it passed `Biquad` by value.

### 9.3 Real-time safety tests

- `nm -u` on each compiled kernel object lists only the libm functions the kernel calls, plus calls the compiler substitutes. Apple clang replaces a `sinf`/`cosf` pair with `__sincosf_stret`, and zero-fill loops with `bzero`. The allowed memory builtins (`bzero`, `memset`, `memcpy`, `memmove`) do not allocate, lock or do I/O.
- Every case that resets mid-stream must produce output different from the same case without the reset. Otherwise a reset that did nothing in both Python and C would pass.
- `-Wdouble-promotion` passes on all `f32` fixtures.

### 9.4 Validator tests

Each `CIGNAL.*` rule has a test with a minimal source that triggers it and a check of the rule id and source span.

### 9.5 Benchmarks

`make bench` (`bench/run.py`) times every fixture with the `bench.c.in` shim: `K_process` on 64-sample blocks of noise held in memory, no I/O, minimum of 5 runs. Hand-written C references in `bench/ref/` share the kernel ABI. They keep state in locals and wrap the FIR index with a mask; the FIR reference recomputes its coefficients every block, as the fixture's `control` does. A benchmark is a report, not a pass/fail test.

M5 results, ns per frame at `-O2` (the shims' flags):

| kernel | clang 21, Apple M-series | hand-written | GCC 14.2, Linux arm64 VM | hand-written |
|-|-|-|-|-|
| Biquad | 4.75 | 4.75 | 4.12 | 4.12 |
| FIR (16 taps, coefficients per block) | 3.88 | 4.67 | 7.10 | 8.03 |
| Saw (`% 1.0` phase wrap) | 5.70 | | 1.63 | |
| Glide (smoothed, at target) | 0.35 | | 0.35 | |

Changes made in M5, each verified by the bit-exact tests on both platforms:

- `restrict` on `self` in `K_process` (section 6). The biquad went from 8.16 to 4.75 ns; the lowpass, SVF, one-pole, drive and glide fixtures improved by 30-50%.
- `i32 % M` for a constant `M > 0` no longer calls the general helper. A power of two becomes `& (M - 1)`, exact for two's complement, which C requires of `int32_t`; other `M` use `r = a % M; r < 0 ? r + M : r`. The FIR's taps lost a zero/-1 guard and a sign fix-up each.
- The smoothing step skips its arithmetic when `value == target`. The full step yields `target` in that case too, signed zeros included, so the result is unchanged. Glide at target: 3.49 to 0.35 ns. While always gliding it costs 18% more (4.12 vs 3.49 ns), a trade towards the common case.

Tried and reverted: `__builtin_expect` on the float `%` helper's rare branches. Clang moved the whole sample loop into an outlined cold function and still used selects.

Findings for kernel authors:

- Clang turns rare branches inside a recurrence into selects, which lengthens the loop-carried chain. Float `%` in a phase accumulator costs 5.70 ns on clang and 1.63 ns on GCC. When the increment is known to be in [0, 1), `phase += inc; if phase >= 1.0: phase -= 1.0` costs 2.58 ns on clang. That form is not Python's `%` in general, so the compiler cannot make the substitution.
- `control` runs on every block. Recomputing 16 windowed-sinc coefficients there costs about 1 ns per frame at 64-frame blocks. A kernel that wants to skip unchanged parameters can compare against a cached value in `control`. The compiler does not do this itself: skipping is equivalent only if `control` is idempotent and reads nothing `tick` writes, which needs a flow analysis M5 did not attempt.
- `-O3 -mcpu=native` changes little except the FIR (3.88 to 3.29 ns on clang, 7.10 to 4.55 ns on GCC). Native flags are not portable for distributed binaries, so the shims stay at `-O2`; `extra_flags` takes others.

## 10. Pipeline

```
import module (CPython) --> collect build-time values
parse source (ast)      --> validate (CIGNAL.* diagnostics)
                        --> type check (annotations + literal typing)
                        --> lower to typed IR
                        --> fold constants, check indices
                        --> emit C (.h, .c) and, optionally, a shim
```

- M0 validates, type-checks and emits C in one typed walk over the AST. Unsupported constructs are reported during the walk, so no separate validator duplicates its knowledge. Index checking in M2 did not need one: it is an interval analysis over the index's AST. A typed IR comes in when a pass needs one. It will be a typed tree, with no CFG or SSA. The C compiler does the low-level optimisation.
- Every error in a file is reported before the build stops. Each diagnostic has a stable rule id, severity, source span, message and remediation.

## 11. CLI

```
cignal check  FILE [--kernel NAME] [--format text|json]
cignal build  FILE [--kernel NAME] [-o DIR] [--prefix P] [--shim none|driver|pd|sc]
              [--pd-include DIR] [--pd-name NAME~] [--sc-include DIR] [--sc-name Name]
```

`check` prints warnings and exits 0, or prints errors and exits 1; `--format json` prints the diagnostics as JSON. `build` prints warnings to stderr. `cignal test` and `--fast-math` are not implemented.

`--kernel` is required when the file defines more than one kernel that no other kernel uses as a field.

## 12. Design decisions

### 12.1 Explicit fields over Vult's implicit call-site contexts

Vult gives each call site its own `mem` context, so `lp(x)` twice means two filters with no declarations. Emulating that under CPython means finding the caller's bytecode offset through `sys._getframe()`. That depends on interpreter internals and is slow. Fields cost one declaration per sub-unit, run as plain Python, and map one-to-one to the C struct. Methods on a kernel replace Vult's `and` groups. Sugar that lowers implicit state to fields can come later without changing the ABI.

### 12.2 C99 over C++ or Rust

C embeds in every target host: C hosts directly, C++ hosts by `#include`, Rust through FFI. One generator means one target language to keep correct. Vult's C++ output passes contexts as `&_ctx`. A pointer works in both C and C++.

### 12.3 Python semantics over C semantics

Where the two disagree (floor division, modulo, `min`/`max` with NaN), cignal emits Python's semantics. That keeps the reference model authoritative. The C-native form stays available as an explicit call, for example `math.fmod`.

### 12.4 Safe-by-construction indexing over proofs or runtime checks

Runtime checks put branches in the hot loop. multigen's z3-based `BoundsProver` returns UNKNOWN for nested and attribute subscripts, so a strict mode built on it rejects common code. A small set of index forms that are always safe covers DSP use: loop counters, masks and modulo.

### 12.5 Build-time sizes only

Sizing at init from the sample rate needs a size query before allocation, plus host co-operation. Fixed sizes keep the struct a plain `sizeof`. The cost is memory sized for the largest supported rate.

## 13. Open questions

1. **First real host.** Decided: Pd, built in M0. It has a plain C API and runs headless.
2. **Parameter smoothing.** Decided in M3: per sample, readable only outside `control` (section 3.4).
3. **`i32` overflow in the reference.** Decided: `CHECK_I32` checks stored values after each `tick` (section 5.3). It needs no rewrite of the user's source, and the suite went from 48 s to 52 s. An AST rewrite asserting range on every operation was the alternative. A wrap inside an expression that the kernel masks back into range is not an error, and matches C (`ops.py`).
4. **`f32` rounding in the reference.** Open. `numpy.float32` cannot give a bit-exact reference: NumPy computes `sin` and `tanh` itself, and differed from Apple's `sinf` on 15% and `tanhf` on 38% of 20000 inputs (macOS arm64, 2026-11-25; `exp` and `log` matched). It would also make NumPy a runtime dependency and run arithmetic at least 2x slower. A bit-exact `float` reference would have to call the libm `f` functions through ctypes, in a test-only shadow of the reference. The SNR test (9.1) stands meanwhile.
5. **Fixed-point output** (Vult's `-real fixed`, q16.16) for Teensy and Eurorack hardware. Only if hardware targets are near-term.
6. **Multi-channel instances for SIMD.** Measured: not needed, since `Array[K, N]` gives the parallelism ([simd.md](simd.md)). Open: whether field layout becomes ABI at v1.0. Making it not ABI would allow a struct-of-arrays `Array[K, N]` without a host change.

## 14. Milestones

| Milestone | Scope | Exit criterion |
|-|-|-|
| M0 (done) | `Kernel`, scalar fields, `tick` (one output), `init`, arithmetic, comparisons, `if`, conditional expressions, casts, 12 `math` functions; `driver` and Pd shims; tests 9.1 to 9.4 | Biquad, one-pole and mixed-syntax fixtures pass 9.1 to 9.3 |
| M1 (done): playable ugens | `param`, `control`, `K_set_param`, `K_info`; sub-kernels as fields; methods; tuple returns, several inputs and outputs; `%`; Pd parameters through inlets and creation arguments | Resonant lowpass, saw oscillator and stereo balance fixtures pass 9.1 to 9.3 with mid-stream parameter changes, and in Pd |
| M2 (done) | `Array`, `Delay`, `for` loops with `break`/`continue`, index checking, `reset` (driver `--reset`, Pd `reset` message) | Echo, comb, state-variable filter and FIR fixtures pass 9.1 to 9.3, with mid-stream resets, and in Pd |
| M2.5 (done) | `//`, `**`, bitwise operators and shifts, chained comparisons, module-level functions, `while` with warnings; wrapping `i32` arithmetic; `--prefix`; flush-to-zero in the Pd and SuperCollider shims; Pd mid-stream tests | The `ops` fixture (31-bit LCG, floor division and modulo with negative operands, shifts, `**`, chained comparisons, free functions, `while`) passes 9.1 to 9.3, in Pd and in SuperCollider; denormal probes flush in both hosts |
| M3 (done) | `@table`, parameter smoothing | Waveshaper (table and smoothed drive), wavetable oscillator and glide fixtures pass 9.1 to 9.3 and in Pd; the float smoother reaches its target exactly with no denormals |
| M4 (done) | SuperCollider UGen shim, sclang class file, `extern "C"` header guard | All 13 fixtures match the driver bit for bit in scsynth, with mid-stream changes and resets, and through sclang-built SynthDefs (Linux arm64 and x86-64) |
| M5 (done) | `make bench`, hand-written references, assembly inspection; `restrict` on `self`, constant-divisor `%`, converged-smoothing branch | Biquad equal to hand-written C on clang and GCC; FIR faster than hand-written (section 9.5) |

## 15. Code taken from multigen

| multigen | Use in cignal |
|-|-|
| `src/multigen/frontend/diagnostics.py` | Copied in M0 with `RuleId` replaced: 75 non-blank lines (of cignal's source: 1,079 non-blank lines in M0, 1,519 in M1, 1,935 in M2, 2,106 in M3, 2,235 in M4, 2,301 in M5, 2,620 after M2.5). |
| `src/multigen/frontend/static_validation.py` | Not needed. The single walk reports in source order with no duplicates. |
| The `KNOWN_FAILURES` strict-xfail pattern in `tests/test_compilation.py` | Adopt when the first known failure appears. |
| C converter, container runtime, type inference engines, `BoundsProver` | Do not reuse. They implement general Python semantics (heap containers, setjmp exceptions, unbounded ints), which section 2 excludes. |
