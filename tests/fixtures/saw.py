"""Naive sawtooth oscillator. The phase accumulates in f64, so float and double builds stay in step."""

from cignal import Kernel, f32, f64, param


class Saw(Kernel):
    freq: f32 = param(440.0, lo=0.0, hi=20000.0)
    phase: f64 = 0.0

    def tick(self) -> f32:
        out = f32(2.0 * self.phase - 1.0)
        self.phase = (self.phase + f64(self.freq) / f64(self.sample_rate)) % 1.0
        return out
