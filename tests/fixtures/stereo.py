"""Stereo gain and equal-power balance: two inputs, two outputs, a helper method returning a tuple."""

import math

from cignal import Kernel, f32, param


class Stereo(Kernel):
    gain: f32 = param(1.0, lo=0.0, hi=4.0)
    balance: f32 = param(0.0, lo=-1.0, hi=1.0)
    left: f32 = 1.0
    right: f32 = 1.0

    def gains(self, b: f32) -> tuple[f32, f32]:
        angle = (b + 1.0) * math.pi / 4.0
        return math.cos(angle) * math.sqrt(2.0), math.sin(angle) * math.sqrt(2.0)

    def control(self) -> None:
        gl, gr = self.gains(self.balance)
        self.left = gl * self.gain
        self.right = gr * self.gain

    def tick(self, left_in: f32, right_in: f32) -> tuple[f32, f32]:
        return left_in * self.left, right_in * self.right
