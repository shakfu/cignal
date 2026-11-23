"""Moog-style ladder: four tanh-saturated one-pole stages with resonant feedback, 2x oversampled.

At `res` above about 1.0 the filter self-oscillates, and the float build drifts in phase from the double
build; its case in `tests/support.py` sets a lower SNR threshold for that reason (spec 9.1).
"""

import math

from cignal import Array, Kernel, f32, param


class Ladder(Kernel):
    cutoff: f32 = param(1000.0, lo=20.0, hi=18000.0, smooth=0.005)
    res: f32 = param(0.3, lo=0.0, hi=1.2)
    s: Array[f32, 4]
    prev: f32 = 0.0

    def tick(self, x: f32) -> f32:
        # `control` may not read the smoothed cutoff, so the coefficient is computed per sample.
        g = 1.0 - math.exp(-math.tau * self.cutoff / (2.0 * self.sample_rate))
        for k in range(2):
            u = (self.prev + x) * 0.5 if k == 0 else x  # the upsampled point between samples
            v = math.tanh(u - 4.0 * self.res * self.s[3])
            for i in range(4):
                self.s[i] += g * (v - math.tanh(self.s[i]))
                v = math.tanh(self.s[i])
        self.prev = x
        return self.s[3]
