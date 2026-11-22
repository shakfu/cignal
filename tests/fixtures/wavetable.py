"""Wavetable oscillator: a sine sampled at build time, read by an f64 phase."""

import math

from cignal import Kernel, f32, f64, param, table


@table(size=1024, lo=0.0, hi=1.0)
def sine(p: f32) -> f32:
    return math.sin(2.0 * math.pi * p)


class Wavetable(Kernel):
    freq: f32 = param(440.0, lo=0.0, hi=20000.0)
    phase: f64 = 0.0

    def tick(self) -> f32:
        out = sine(f32(self.phase))
        self.phase = (self.phase + f64(self.freq) / f64(self.sample_rate)) % 1.0
        return out
