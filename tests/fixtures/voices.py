"""Four one-pole lowpass voices on one input, each with its own cutoff and gain: `Array` parameters.

The host sees `cutoffs_0` to `cutoffs_3`, then `gains_0` to `gains_3`.
"""

import math

from cignal import Array, Kernel, f32, param


class Pole(Kernel):
    a: f32 = 0.0
    y: f32 = 0.0

    def tick(self, x: f32) -> f32:
        self.y += self.a * (x - self.y)
        return self.y


class Voices(Kernel):
    cutoffs: Array[f32, 4] = param(1000.0, lo=20.0, hi=20000.0, smooth=0.002)
    gains: Array[f32, 4] = param(0.25, lo=0.0, hi=1.0)
    poles: Array[Pole, 4]
    heading: f32 = 0.0  # where voice 1's cutoff glides to, as `control` sees it

    def control(self) -> None:
        self.heading = self.target("cutoffs", 1)

    def tick(self, x: f32) -> tuple[f32, f32]:
        acc = 0.0
        for i in range(4):
            self.poles[i].a = 1.0 - math.exp(-math.tau * self.cutoffs[i] / self.sample_rate)
            acc += self.gains[i] * self.poles[i].tick(x)
        return acc, self.heading * 1e-4
