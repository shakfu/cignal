"""Exercises `abs`, `min`, `max`, `math.fmod`, `math.log2`, `math.log10` and `math.copysign`.

`min` and `max` must give CPython's results where `fmin` and `fmax` differ: a NaN first argument, and
zeros of opposite sign. The NaN and the zeros are made at run time, in `f64`, so both builds see them.
"""

import math

from cignal import Kernel, f32, f64, i32


class Builtins(Kernel):
    huge: f64 = 1e308

    def tick(self, x: f32) -> f32:
        q = i32(x * 1000.0)
        ints = abs(q) + min(q, 7, -q) + max(q, -3) + abs(-q)
        clipped = max(-0.5, min(x, 0.5))
        logs = math.log2(math.fabs(x) + 1.0) + math.log10(abs(x) + 2.0)
        rem = math.fmod(x * 7.0, 0.75) + math.copysign(0.25, x)

        inf = self.huge * 10.0  # CPython does not raise on overflow
        nan = inf - inf  # a separate statement: `a * b - a * b` may contract to an FMA, which gives -inf
        flags = f64(0.0)
        if min(nan, f64(x)) != min(nan, f64(x)):  # CPython keeps the NaN; fmin returns x
            flags += 1.0
        z = f64(x) * 0.0  # a zero with x's sign
        # CPython keeps the first of two equal zeros; fmin and fmax may pick either.
        flags += math.copysign(2.0, min(z, -z)) + math.copysign(4.0, max(-z, z))
        return clipped + f32(ints) * 1e-4 + logs * 0.1 + rem * 0.1 + f32(flags) * 0.01
