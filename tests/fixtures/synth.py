"""Four-voice synth played by events: the kernel takes notes and allocates its own voices.

`tests/fixtures/poly.py` is the same voice played through parameters, with the host allocating.
"""

import math

from cignal import Array, Kernel, event, f32, f64, i32, param


class Voice(Kernel):
    note: i32 = -1  # -1 when free
    age: i32 = 0  # note-ons since this voice started; the oldest is stolen
    sustained: bool = False  # released while the pedal was down
    freq: f64 = 0.0
    phase: f64 = 0.0
    gate: f32 = 0.0
    env: f32 = 0.0
    y: f32 = 0.0

    def start(self, note: i32, freq: f64, velocity: f32) -> None:
        self.note = note
        self.age = 0
        self.freq = freq
        self.gate = velocity
        self.env = 0.0  # a repeated note restarts its attack

    def release(self) -> None:
        self.note = -1
        self.gate = 0.0
        self.sustained = False

    def play(self, period: f64, bend: f64, attack: f32, release: f32, a: f32) -> f32:
        rate = attack if self.gate > self.env else release
        self.env += rate * (self.gate - self.env)
        self.phase += self.freq * bend * period
        if self.phase >= 1.0:
            self.phase -= 1.0
        self.y += a * (f32(2.0 * self.phase - 1.0) - self.y)
        return self.y * self.env


class Synth(Kernel):
    cutoff: f32 = param(2000.0, lo=50.0, hi=16000.0)
    attack: f32 = param(0.005, lo=0.0005, hi=2.0)
    release: f32 = param(0.05, lo=0.001, hi=5.0)
    voices: Array[Voice, 4]
    bend: f64 = 1.0  # frequency ratio
    held: bool = False  # sustain pedal
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

    @event
    def on_note(self, note: i32, velocity: f32) -> None:
        # The voice already playing this note, else the longest-free voice, else the oldest. `% 4`
        # proves the index: the range check does not follow `pick = i` out of the loop.
        pick = 0
        best = -1
        for i in range(4):
            self.voices[i].age += 1
            score = self.voices[i].age
            if self.voices[i].note < 0:
                score += 1000000
            if self.voices[i].note == note:
                score += 2000000
            if score > best:
                best = score
                pick = i
        self.voices[pick % 4].start(note, 440.0 * 2.0 ** (f64(note - 69) / 12.0), velocity)

    @event
    def on_off(self, note: i32) -> None:
        for i in range(4):
            if self.voices[i].note == note:
                if self.held:
                    self.voices[i].sustained = True
                else:
                    self.voices[i].release()

    @event
    def on_hold(self, down: bool) -> None:
        self.held = down
        if not down:
            for i in range(4):
                if self.voices[i].sustained:
                    self.voices[i].release()

    @event
    def on_bend(self, semitones: f64) -> None:
        self.bend = 2.0 ** (semitones / 12.0)

    def tick(self) -> f32:
        acc = 0.0
        for i in range(4):
            acc += self.voices[i].play(self.period64, self.bend, self.ka, self.kr, self.a)
        return acc * 0.25
