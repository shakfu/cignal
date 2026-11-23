"""Freeverb (Jezar at Dreampoint, public domain): 8 damped combs in parallel, then 4 allpasses in series.

Exercises an `Array` of sub-kernels, an `Array` of delay lines, and indexed module-level tuples.
"""

from cignal import Array, Delay, Kernel, f32, i32, param

COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)  # samples at 44.1 kHz
ALLPASSES = (556, 441, 341, 225)
MAX_RATE = 96000
COMB_MAX = 1617 * MAX_RATE // 44100 + 1
ALLPASS_MAX = 556 * MAX_RATE // 44100 + 1


class Comb(Kernel):
    feedback: f32 = 0.84
    damp: f32 = 0.2
    store: f32 = 0.0
    length: i32 = 1116
    line: Delay[f32, COMB_MAX]

    def tick(self, x: f32) -> f32:
        y = self.line.read(self.length)
        self.store = y * (1.0 - self.damp) + self.store * self.damp
        self.line.write(x + self.store * self.feedback)
        return y


class Freeverb(Kernel):
    room: f32 = param(0.5, lo=0.0, hi=1.0)
    damp: f32 = param(0.5, lo=0.0, hi=1.0)
    wet: f32 = param(0.3, lo=0.0, hi=1.0)
    combs: Array[Comb, 8]
    allpasses: Array[Delay[f32, ALLPASS_MAX], 4]
    allpass_lengths: Array[i32, 4]

    def init(self) -> None:
        scale = self.sample_rate / 44100.0
        for i in range(8):
            self.combs[i].length = i32(f32(COMBS[i]) * scale)
        for i in range(4):
            self.allpass_lengths[i] = i32(f32(ALLPASSES[i]) * scale)

    def control(self) -> None:
        for i in range(8):
            self.combs[i].feedback = self.room * 0.28 + 0.7
            self.combs[i].damp = self.damp * 0.4

    def tick(self, x: f32) -> f32:
        acc = 0.0
        for i in range(8):
            acc += self.combs[i].tick(x * 0.015)
        for i in range(4):
            b = self.allpasses[i].read(self.allpass_lengths[i])
            self.allpasses[i].write(acc + b * 0.5)
            acc = b - acc
        return x * (1.0 - self.wet) + acc * self.wet * 3.0
