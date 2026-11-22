"""Resonant lowpass: an RBJ biquad section held as a sub-kernel, coefficients recomputed per block."""

import math

from cignal import Kernel, f32, param


class Section(Kernel):
    """Transposed direct form II biquad. Coefficients are set by the owner."""

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
    section: Section

    def control(self) -> None:
        w = 2.0 * math.pi * self.cutoff / self.sample_rate
        alpha = math.sin(w) / (2.0 * self.q)
        c = math.cos(w)
        a0 = 1.0 + alpha
        self.section.b0 = (1.0 - c) / 2.0 / a0
        self.section.b1 = (1.0 - c) / a0
        self.section.b2 = self.section.b0
        self.section.a1 = -2.0 * c / a0
        self.section.a2 = (1.0 - alpha) / a0

    def tick(self, x: f32) -> f32:
        return self.section.tick(x)
