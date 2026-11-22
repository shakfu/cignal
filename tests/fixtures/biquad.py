"""RBJ lowpass biquad, transposed direct form II, coefficients computed at init."""

import math

from cignal import Kernel, f32

CUTOFF = 1000.0
Q = 0.707


class Biquad(Kernel):
    b0: f32 = 1.0
    b1: f32 = 0.0
    b2: f32 = 0.0
    a1: f32 = 0.0
    a2: f32 = 0.0
    z1: f32 = 0.0
    z2: f32 = 0.0

    def init(self) -> None:
        w = 2.0 * math.pi * CUTOFF / self.sample_rate
        alpha = math.sin(w) / (2.0 * Q)
        c = math.cos(w)
        a0 = 1.0 + alpha
        self.b0 = (1.0 - c) / 2.0 / a0
        self.b1 = (1.0 - c) / a0
        self.b2 = self.b0
        self.a1 = -2.0 * c / a0
        self.a2 = (1.0 - alpha) / a0

    def tick(self, x: f32) -> f32:
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y
