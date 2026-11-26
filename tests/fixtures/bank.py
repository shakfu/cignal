"""Eight biquad lowpass sections at fixed frequencies, summed: arithmetic voices, as `docs/dev/soa.md` measures."""

import math

from cignal import Array, Kernel, f32


class Section(Kernel):
    b0: f32 = 0.0
    b1: f32 = 0.0
    b2: f32 = 0.0
    a1: f32 = 0.0
    a2: f32 = 0.0
    x1: f32 = 0.0
    x2: f32 = 0.0
    y1: f32 = 0.0
    y2: f32 = 0.0

    def tick(self, x: f32) -> f32:
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y


class Bank(Kernel):
    sections: Array[Section, 8]

    def init(self) -> None:
        for i in range(8):
            w = math.tau * (200.0 * f32(i + 1)) / self.sample_rate
            alpha = math.sin(w) / (2.0 * 0.707)
            norm = 1.0 / (1.0 + alpha)
            self.sections[i].b0 = (1.0 - math.cos(w)) * 0.5 * norm
            self.sections[i].b1 = (1.0 - math.cos(w)) * norm
            self.sections[i].b2 = self.sections[i].b0
            self.sections[i].a1 = -2.0 * math.cos(w) * norm
            self.sections[i].a2 = (1.0 - alpha) * norm

    def tick(self, x: f32) -> f32:
        acc = 0.0
        for i in range(8):
            acc += self.sections[i].tick(x)
        return acc
