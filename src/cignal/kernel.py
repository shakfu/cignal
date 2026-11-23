"""Annotation types, `param`, `Array`, `Delay`, and the `Kernel` base class, as they behave under CPython.

The compiler reads `f32`, `f64` and `i32` by name from the source. At runtime
they are plain aliases, so kernels run unchanged as the reference model.
`Array` and `Delay` here define the semantics the generated C reproduces.
"""

import inspect
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

f32: TypeAlias = float
f64: TypeAlias = float
i32: TypeAlias = int


# A smoothed value closer than this to its target snaps to it, before it can become denormal.
SMOOTH_SNAP = 1e-20


@dataclass(frozen=True)
class Param:
    """A host-controlled parameter declared with `param()`."""

    default: float
    lo: float
    hi: float
    smooth: float = 0.0  # one-pole time constant in seconds; 0 means no smoothing


def param(default: float, *, lo: float, hi: float, smooth: float = 0.0) -> float:
    """Declare a parameter field: `cutoff: f32 = param(1000.0, lo=20.0, hi=20000.0)`.

    With `smooth=seconds`, the field glides towards each value the host sets, one step per sample
    before `tick`. `control` may not read a smoothed parameter: the value it saw would depend on the
    host's block size.
    """
    # Typed as float so `self.cutoff` type-checks; the class attribute holds the Param.
    return cast(float, Param(default, lo, hi, smooth))


def smooth_coefficient(seconds: float, sample_rate: float) -> float:
    """The one-pole coefficient for a time constant; the generated init computes it the same way."""
    return 1.0 - math.exp(-1.0 / (seconds * sample_rate))


def smooth_step(value: float, target: float, coef: float) -> float:
    """One smoothing step. Snaps when the step makes no progress, where a float smoother would stall."""
    d = target - value
    stepped = value + coef * d
    return target if stepped == value or (-SMOOTH_SNAP < d < SMOOTH_SNAP) else stepped


class Table:
    """A function sampled at build time: `@table(size=256, lo=-4.0, hi=4.0)`. Calls interpolate linearly."""

    def __init__(self, function: Callable[[float], float], size: int, lo: float, hi: float) -> None:
        self.function = function  # the exact function, for measuring table error
        self.size, self.lo, self.hi = size, lo, hi
        self.scale = size / (hi - lo)
        self.values = [float(function(lo + (hi - lo) * k / size)) for k in range(size + 1)]

    def __call__(self, x: float) -> float:
        if not x >= self.lo:
            x = self.lo
        elif x > self.hi:
            x = self.hi
        p = (x - self.lo) * self.scale
        i = int(p)
        if i >= self.size:
            i = self.size - 1
        frac = p - i
        a = self.values[i]
        return a + frac * (self.values[i + 1] - a)


def table(*, size: int, lo: float, hi: float) -> Callable[[Callable[[float], float]], Table]:
    """Sample a one-argument function at `size + 1` evenly spaced points in [lo, hi], in f64."""
    return lambda function: Table(function, size, lo, hi)


@dataclass(frozen=True)
class Spec:
    """The annotation `Array[T, N]` or `Delay[T, N]` evaluates to one of these."""

    kind: type
    element: Any  # a scalar type, a `Kernel` subclass, or a `Delay` spec
    size: int


class Array(list[Any]):
    """A fixed-length array field: `taps: Array[f32, 16]`. Indices must be provably in range.

    Elements are zeros, sub-kernels (`Array[Comb, 8]`) or delay lines (`Array[Delay[f32, N], 4]`).
    """

    def __class_getitem__(cls, item: tuple[Any, int]) -> Spec:  # type: ignore[override]
        return Spec(cls, item[0], item[1])

    def __init__(self, spec: Spec, sample_rate: float = 0.0) -> None:
        element: Any = spec.element
        if isinstance(element, Spec):
            super().__init__(element.kind(element) for _ in range(spec.size))
        elif isinstance(element, type) and issubclass(element, Kernel):
            super().__init__(element(sample_rate) for _ in range(spec.size))  # each runs `init`, in order
        else:
            self.zero = element()
            super().__init__(self.zero for _ in range(spec.size))

    def clear(self) -> None:
        """Zero the elements, or reset each sub-kernel or delay line."""
        for k, item in enumerate(self):
            if isinstance(item, Kernel):
                item._reset(keep_params=False)  # a sub-kernel's parameters are plain fields
            elif isinstance(item, Delay):
                item.clear()
            else:
                self[k] = self.zero


class Delay:
    """A ring buffer field: `line: Delay[f32, 4800]`."""

    def __class_getitem__(cls, item: tuple[type, int]) -> Spec:
        return Spec(cls, item[0], item[1])

    def __init__(self, spec: Spec) -> None:
        self.n = spec.size
        self.buf: list[float] = [0.0] * self.n
        self.pos = 0

    def write(self, x: float) -> None:
        """Store `x` as the newest sample."""
        self.buf[self.pos] = x
        self.pos += 1
        if self.pos == self.n:
            self.pos = 0

    def read(self, d: int) -> float:
        """The sample written `d` writes ago, `d` clamped to [1, N]: `read(1)` is the newest."""
        d = 1 if d < 1 else self.n if d > self.n else d
        index = self.pos - d
        if index < 0:
            index += self.n
        return self.buf[index]

    def read_frac(self, d: float) -> float:
        """`read` at a fractional delay, linearly interpolated; `d` clamped to [1, N-1], NaN to 1.

        The position is f64 in C for any element type; an f32 argument is widened without loss.
        """
        hi = float(self.n - 1)
        if not d >= 1.0:
            d = 1.0
        elif d > hi:
            d = hi
        i = int(d)
        frac = d - i
        a = self.read(i)
        b = self.read(i + 1)
        return a + frac * (b - a)

    def clear(self) -> None:
        self.buf = [0.0] * self.n
        self.pos = 0


class Kernel:
    """Base class for DSP kernels. Fields are state; `tick` runs once per sample."""

    sample_rate: f32

    def __init__(self, sample_rate: float) -> None:
        # Same order as the generated `K_init`: parameters, sample rate, other fields, sub-kernels, `init`.
        cls = type(self)
        for name, value in vars(cls).items():
            if isinstance(value, Param):
                setattr(self, name, float(value.default))
        self.sample_rate = float(sample_rate)
        # Smoothed parameters: name -> [target, coefficient].
        self._smoothing = {
            name: [float(p.default), smooth_coefficient(p.smooth, self.sample_rate)]
            for name, p in vars(cls).items()
            if isinstance(p, Param) and p.smooth > 0
        }
        for name, hint in inspect.get_annotations(cls, eval_str=True).items():
            if isinstance(hint, Spec):
                setattr(self, name, Array(hint, self.sample_rate) if hint.kind is Array else hint.kind(hint))
            elif isinstance(hint, type) and issubclass(hint, Kernel) and name not in vars(cls):
                setattr(self, name, hint(sample_rate))
        self.init()

    def init(self) -> None:
        """Run once after fields take their defaults, and again on `reset`. Override to compute state."""

    def control(self) -> None:
        """Run once at the start of each `process` call. Override to recompute from parameters."""

    def reset(self) -> None:
        """Restore all state except parameters and the sample rate, then re-run `init`."""
        self._reset(keep_params=True)

    def _reset(self, keep_params: bool) -> None:
        cls = type(self)
        for name, hint in inspect.get_annotations(cls, eval_str=True).items():
            default = vars(cls).get(name)
            if isinstance(hint, Spec):
                getattr(self, name).clear()
            elif isinstance(hint, type) and issubclass(hint, Kernel):
                # A sub-kernel's parameters are plain fields, so they reset too.
                getattr(self, name)._reset(keep_params=False)
            elif isinstance(default, Param):
                if not keep_params:
                    setattr(self, name, float(default.default))
            elif name in vars(cls):
                setattr(self, name, default)
        self.init()

    def set_param(self, name: str, value: float) -> None:
        """Set a parameter, clamped to its range exactly as the generated setter does."""
        p = vars(type(self)).get(name)
        if not isinstance(p, Param):
            raise KeyError(f"{type(self).__name__} has no parameter {name!r}")
        clamped = p.lo if value < p.lo else p.hi if value > p.hi else value
        if name in self._smoothing:
            self._smoothing[name][0] = clamped
        else:
            setattr(self, name, clamped)

    def process(self, inputs: Sequence[Sequence[float]], n: int) -> list[list[float]]:
        """Run `control`, then `tick` over `n` samples of `inputs` (one sequence per channel)."""
        self.control()
        tick = getattr(self, "tick")  # noqa: B009 - subclasses define tick; the base class does not
        outputs: list[list[float]] = []
        for i in range(n):
            for name, (target, coef) in self._smoothing.items():
                setattr(self, name, smooth_step(getattr(self, name), target, coef))
            result = tick(*(channel[i] for channel in inputs))
            values = result if isinstance(result, tuple) else (result,)
            if not outputs:
                outputs = [[] for _ in values]
            for channel, value in zip(outputs, values, strict=True):
                channel.append(float(value))
        return outputs
