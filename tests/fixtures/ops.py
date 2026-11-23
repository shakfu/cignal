"""Exercises `//`, `**`, bitwise operators and shifts, chained comparisons, free functions, `while`, and
indexed module-level tuples.

Inputs are scaled by powers of two and divisors are exact, so the float and double builds compute the
same integers and wrap at the same points.
"""

import math

from cignal import Kernel, f32, i32

LCG_MUL = 1103515245
STEPS = (3, -1, 4, -1)  # i32: every element is an int
WEIGHTS = (0.5, 0.25, -0.125, 2)  # f32: one float makes the tuple f32


def clamp(x: f32, lo: f32, hi: f32) -> f32:
    return lo if x < lo else hi if x > hi else x


def shape(x: f32) -> f32:
    return clamp(x, -1.0, 1.0) * 0.9


class Ops(Kernel):
    seed: i32 = 22222
    count: i32 = 0

    def noise(self) -> f32:
        # A 31-bit LCG: the multiply wraps in C and is exact in Python; the mask makes them agree.
        self.seed = (self.seed * LCG_MUL + 12345) & 0x7FFFFFFF
        return f32(self.seed >> 8) / 8388608.0 - 0.5

    def tick(self, x: f32) -> f32:
        n = self.noise()
        q = i32(x * 1024.0)
        ints = q // 7 + q // -5 + q % -3 + (q << 3) - (q >> 2) + (q ^ 0x55) - (~q & 0xFF) + (q | 1) + 3**4
        floats = (x * 2.0) // 0.75 + x**2 + math.fabs(x) ** 0.5 + f32(q) ** 3 * 1e-7
        self.count += 5
        while self.count >= 7:
            self.count -= 7
        inside = 1.0 if -0.25 < x < 0.25 else 0.0
        picked = WEIGHTS[q % 4] * f32(STEPS[(q + 1) % 4]) + WEIGHTS[-1] * f32(STEPS[0])  # constants fold
        for k in range(4):
            picked += WEIGHTS[k] * f32(STEPS[k])
        return (
            shape(x + n * 0.1)
            + f32(ints) * 1e-5
            + floats * 0.01
            + f32(self.count) * 0.01
            + inside * 0.1
            + picked * 0.001
        )
