"""Eight-voice synth: per voice a saw, an attack/release envelope and a one-pole lowpass, summed.

Written to find what polyphony needs (TODO.md). The host allocates voices and plays a note by setting
`freqs_k` and `gates_k`; a gate value is the velocity.
"""

import math

from cignal import Array, Kernel, f32, f64, param


class Voice(Kernel):
    # Phase and pitch are f64, since the phase integrates every rounding of the pitch. In f32 the float
    # build measured 83 dB with an f32 phase, and 98 dB with an f32 smoothed pitch.
    phase: f64 = 0.0
    env: f32 = 0.0
    y: f32 = 0.0

    def play(self, freq: f64, period: f64, gate: f32, attack: f32, release: f32, a: f32) -> f32:
        rate = attack if gate > self.env else release
        self.env += rate * (gate - self.env)
        self.phase += freq * period
        if self.phase >= 1.0:
            self.phase -= 1.0
        self.y += a * (f32(2.0 * self.phase - 1.0) - self.y)
        return self.y * self.env


class Poly(Kernel):
    freqs: Array[f64, 8] = param(220.0, lo=20.0, hi=8000.0, smooth=0.02)  # a 20 ms portamento
    gates: Array[f32, 8] = param(0.0, lo=0.0, hi=1.0)
    cutoff: f32 = param(2000.0, lo=50.0, hi=16000.0)
    attack: f32 = param(0.005, lo=0.0005, hi=2.0)
    release: f32 = param(0.05, lo=0.001, hi=5.0)
    voices: Array[Voice, 8]
    period: f32 = 0.0
    period64: f64 = 0.0
    a: f32 = 0.0
    ka: f32 = 0.0
    kr: f32 = 0.0

    def init(self) -> None:
        self.period = 1.0 / self.sample_rate
        self.period64 = 1.0 / f64(self.sample_rate)

    def control(self) -> None:
        self.a = 1.0 - math.exp(-math.tau * self.cutoff * self.period)
        self.ka = 1.0 - math.exp(-self.period / self.attack)
        self.kr = 1.0 - math.exp(-self.period / self.release)

    def tick(self) -> f32:
        acc = 0.0
        for i in range(8):
            acc += self.voices[i].play(self.freqs[i], self.period64, self.gates[i], self.ka, self.kr, self.a)
        return acc * 0.125
