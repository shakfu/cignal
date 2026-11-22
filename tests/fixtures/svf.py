"""Simper's trapezoidal state-variable filter: lowpass, bandpass and highpass outputs."""

import math

from cignal import Kernel, f32, param


class SVF(Kernel):
    cutoff: f32 = param(1000.0, lo=20.0, hi=20000.0)
    res: f32 = param(0.5, lo=0.0, hi=0.98)
    k: f32 = 1.0
    a1: f32 = 0.0
    a2: f32 = 0.0
    a3: f32 = 0.0
    ic1eq: f32 = 0.0
    ic2eq: f32 = 0.0

    def control(self) -> None:
        g = math.tan(math.pi * self.cutoff / self.sample_rate)
        self.k = 2.0 - 2.0 * self.res
        self.a1 = 1.0 / (1.0 + g * (g + self.k))
        self.a2 = g * self.a1
        self.a3 = g * self.a2

    def tick(self, x: f32) -> tuple[f32, f32, f32]:
        v3 = x - self.ic2eq
        v1 = self.a1 * self.ic1eq + self.a2 * v3
        v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3
        self.ic1eq = 2.0 * v1 - self.ic1eq
        self.ic2eq = 2.0 * v2 - self.ic2eq
        return v2, v1, x - self.k * v1 - v2
