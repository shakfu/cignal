"""Windowed-sinc lowpass FIR: `Array` fields, `for` loops, and indices proven in range."""

import math

from cignal import Array, Kernel, f32, i32, param

TAPS = 16


class Fir(Kernel):
    cutoff: f32 = param(4000.0, lo=100.0, hi=20000.0)
    coef: Array[f32, TAPS]
    history: Array[f32, TAPS]
    pos: i32 = 0

    def control(self) -> None:
        fc = self.cutoff / self.sample_rate
        for i in range(TAPS):
            m = f32(i) - (TAPS - 1) / 2.0
            window = 0.54 - 0.46 * math.cos(2.0 * math.pi * f32(i) / (TAPS - 1))
            self.coef[i] = window * math.sin(2.0 * math.pi * fc * m) / (math.pi * m)

    def tick(self, x: f32) -> f32:
        self.history[self.pos % TAPS] = x
        acc: f32 = 0.0
        for i in range(TAPS):
            acc += self.coef[i] * self.history[(self.pos - i) % TAPS]
        self.pos = (self.pos + 1) % TAPS
        return acc
