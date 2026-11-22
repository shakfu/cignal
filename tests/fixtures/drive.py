"""Waveshaper: a tanh lookup table, driven by a smoothed parameter."""

import math

from cignal import Kernel, f32, param, table


@table(size=512, lo=-4.0, hi=4.0)
def soft(x: f32) -> f32:
    return math.tanh(x)


class Drive(Kernel):
    drive: f32 = param(1.0, lo=0.1, hi=4.0, smooth=0.01)

    def tick(self, x: f32) -> f32:
        return soft(x * self.drive) / soft(self.drive)
