"""LFO-swept state-variable filters, retuned every sample or every `EVERY` samples.

Written to measure whether a fixed control rate pays (TODO.md). Retuning costs a `tan` and an `exp2`.
The `K` kernels count samples in a field, so their retune frames do not depend on the host's blocks.
"""

import math

from cignal import Array, Kernel, f32, f64, i32, param

EVERY = 32
VOICES = 8


class Svf(Kernel):
    """Simper's trapezoidal SVF, lowpass output."""

    k: f32 = 1.0
    a1: f32 = 0.0
    a2: f32 = 0.0
    a3: f32 = 0.0
    ic1eq: f32 = 0.0
    ic2eq: f32 = 0.0

    def tune(self, g: f32, k: f32) -> None:
        self.k = k
        self.a1 = 1.0 / (1.0 + g * (g + k))
        self.a2 = g * self.a1
        self.a3 = g * self.a2

    def tick(self, x: f32) -> f32:
        v3 = x - self.ic2eq
        v1 = self.a1 * self.ic1eq + self.a2 * v3
        v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3
        self.ic1eq = 2.0 * v1 - self.ic1eq
        self.ic2eq = 2.0 * v2 - self.ic2eq
        return v2


class Sweep(Kernel):
    """One swept filter: an LFO phase and the SVF it retunes."""

    phase: f64 = 0.0
    svf: Svf

    def advance(self, inc: f64) -> None:
        self.phase += inc
        if self.phase >= 1.0:
            self.phase -= 1.0

    def retune(self, cutoff: f32, depth: f32, w: f32, k: f32, top: f32) -> None:
        lfo = math.sin(math.tau * f32(self.phase))
        fc = min(cutoff * 2.0 ** (depth * lfo), top)
        self.svf.tune(math.tan(w * fc), k)


class AutoFilter(Kernel):
    cutoff: f32 = param(800.0, lo=20.0, hi=8000.0)
    depth: f32 = param(2.0, lo=0.0, hi=4.0)  # octaves either side
    rate: f32 = param(0.5, lo=0.01, hi=20.0)  # Hz
    res: f32 = param(0.7, lo=0.0, hi=0.98)
    sweep: Sweep
    inc: f64 = 0.0
    w: f32 = 0.0
    k: f32 = 1.0
    top: f32 = 0.0

    def control(self) -> None:
        self.inc = f64(self.rate) / f64(self.sample_rate)
        self.w = math.pi / self.sample_rate
        self.k = 2.0 - 2.0 * self.res
        self.top = 0.45 * self.sample_rate

    def tick(self, x: f32) -> f32:
        self.sweep.retune(self.cutoff, self.depth, self.w, self.k, self.top)
        self.sweep.advance(self.inc)
        return self.sweep.svf.tick(x)


class AutoFilterK(Kernel):
    cutoff: f32 = param(800.0, lo=20.0, hi=8000.0)
    depth: f32 = param(2.0, lo=0.0, hi=4.0)
    rate: f32 = param(0.5, lo=0.01, hi=20.0)
    res: f32 = param(0.7, lo=0.0, hi=0.98)
    sweep: Sweep
    n: i32 = 0
    inc: f64 = 0.0
    w: f32 = 0.0
    k: f32 = 1.0
    top: f32 = 0.0

    def control(self) -> None:
        self.inc = f64(self.rate) / f64(self.sample_rate)
        self.w = math.pi / self.sample_rate
        self.k = 2.0 - 2.0 * self.res
        self.top = 0.45 * self.sample_rate

    def tick(self, x: f32) -> f32:
        if self.n == 0:
            self.sweep.retune(self.cutoff, self.depth, self.w, self.k, self.top)
        self.n = (self.n + 1) % EVERY
        self.sweep.advance(self.inc)
        return self.sweep.svf.tick(x)


class AutoBank(Kernel):
    """VOICES swept filters in parallel on one input, LFO rates spread by 13% each."""

    cutoff: f32 = param(800.0, lo=20.0, hi=8000.0)
    depth: f32 = param(2.0, lo=0.0, hi=4.0)
    rate: f32 = param(0.5, lo=0.01, hi=20.0)
    res: f32 = param(0.7, lo=0.0, hi=0.98)
    sweeps: Array[Sweep, VOICES]
    incs: Array[f64, VOICES]
    w: f32 = 0.0
    k: f32 = 1.0
    top: f32 = 0.0

    def control(self) -> None:
        for i in range(VOICES):
            self.incs[i] = f64(self.rate) * (1.0 + 0.13 * f64(i)) / f64(self.sample_rate)
        self.w = math.pi / self.sample_rate
        self.k = 2.0 - 2.0 * self.res
        self.top = 0.45 * self.sample_rate

    def tick(self, x: f32) -> f32:
        acc = 0.0
        for i in range(VOICES):
            self.sweeps[i].retune(self.cutoff, self.depth, self.w, self.k, self.top)
            self.sweeps[i].advance(self.incs[i])
            acc += self.sweeps[i].svf.tick(x)
        return acc * 0.125


class AutoBankK(Kernel):
    cutoff: f32 = param(800.0, lo=20.0, hi=8000.0)
    depth: f32 = param(2.0, lo=0.0, hi=4.0)
    rate: f32 = param(0.5, lo=0.01, hi=20.0)
    res: f32 = param(0.7, lo=0.0, hi=0.98)
    sweeps: Array[Sweep, VOICES]
    incs: Array[f64, VOICES]
    n: i32 = 0
    w: f32 = 0.0
    k: f32 = 1.0
    top: f32 = 0.0

    def control(self) -> None:
        for i in range(VOICES):
            self.incs[i] = f64(self.rate) * (1.0 + 0.13 * f64(i)) / f64(self.sample_rate)
        self.w = math.pi / self.sample_rate
        self.k = 2.0 - 2.0 * self.res
        self.top = 0.45 * self.sample_rate

    def tick(self, x: f32) -> f32:
        if self.n == 0:
            for i in range(VOICES):
                self.sweeps[i].retune(self.cutoff, self.depth, self.w, self.k, self.top)
        self.n = (self.n + 1) % EVERY
        acc = 0.0
        for i in range(VOICES):
            self.sweeps[i].advance(self.incs[i])
            acc += self.sweeps[i].svf.tick(x)
        return acc * 0.125
