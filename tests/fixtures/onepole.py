"""One-pole lowpass with a cubic soft clip, continuous at +-1, to exercise `if`/`elif`."""

import math

from cignal import Kernel, f32

CUTOFF = 200.0


class OnePole(Kernel):
    a: f32 = 0.0
    y: f32 = 0.0

    def init(self) -> None:
        self.a = 1.0 - math.exp(-2.0 * math.pi * CUTOFF / self.sample_rate)

    def tick(self, x: f32) -> f32:
        self.y += self.a * (x - self.y)
        out = 2.0 * self.y
        if out > 1.0:
            out = 1.0
        elif out < -1.0:
            out = -1.0
        else:
            out = 1.5 * out - 0.5 * out * out * out
        return out
