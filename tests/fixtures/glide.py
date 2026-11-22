"""Outputs its smoothed parameter, so tests can check the glide itself."""

from cignal import Kernel, f32, param


class Glide(Kernel):
    level: f32 = param(0.0, lo=-1000.0, hi=1000.0, smooth=0.001)

    def tick(self) -> f32:
        return self.level
