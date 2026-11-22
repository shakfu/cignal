"""Damped feedback comb (the Freeverb building block): an integer delay with a one-pole lowpass in the loop."""

from cignal import Delay, Kernel, f32, i32, param


class Comb(Kernel):
    delay: f32 = param(100.0, lo=1.0, hi=1000.0)  # samples
    feedback: f32 = param(0.7, lo=-0.99, hi=0.99)
    damp: f32 = param(0.2, lo=0.0, hi=1.0)
    line: Delay[f32, 1024]
    store: f32 = 0.0
    d: i32 = 1

    def control(self) -> None:
        self.d = i32(self.delay)

    def tick(self, x: f32) -> f32:
        y = self.line.read(self.d)
        self.store = y + self.damp * (self.store - y)
        self.line.write(x + self.feedback * self.store)
        return y
