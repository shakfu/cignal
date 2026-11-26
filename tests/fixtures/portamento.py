"""`glide` in a sub-kernel, which has no parameters: each voice slews to the pitch its event sets."""

import math

from cignal import Array, Kernel, event, f32, f64, glide, i32


class Voice(Kernel):
    pitch: f32 = 0.0
    goal: f32 = 0.0
    fine: f64 = 0.0  # the same glide in f64
    fine_goal: f64 = 0.0

    def tick(self, coef: f32, coef64: f64) -> f32:  # a sub-kernel's tick may take any scalar types
        self.pitch = glide(self.pitch, self.goal, coef)
        self.fine = glide(self.fine, self.fine_goal, coef64)
        return self.pitch + f32(self.fine) * 0.001


class Portamento(Kernel):
    voices: Array[Voice, 2]
    coef: f32 = 0.0
    coef64: f64 = 0.0

    def init(self) -> None:
        self.coef64 = 1.0 - math.exp(-1.0 / (0.002 * f64(self.sample_rate)))
        self.coef = f32(self.coef64)

    @event
    def on_go(self, voice: i32, pitch: f32) -> None:
        v = voice % 2
        self.voices[v].goal = pitch
        self.voices[v].fine_goal = f64(pitch) * 1000.0

    def tick(self) -> tuple[f32, f32]:
        return self.voices[0].tick(self.coef, self.coef64), self.voices[1].tick(self.coef, self.coef64)
