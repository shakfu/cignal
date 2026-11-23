"""Outputs its smoothed parameter and the target `control` reads, so tests can check the glide itself."""

from cignal import Kernel, f32, param


class Glide(Kernel):
    level: f32 = param(0.0, lo=-1000.0, hi=1000.0, smooth=0.001)
    held: f32 = 0.0

    def control(self) -> None:
        self.held = self.target("level")

    def tick(self) -> tuple[f32, f32]:
        return self.level, self.held
