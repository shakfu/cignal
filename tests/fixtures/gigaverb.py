"""Gigaverb, after Juhana Sadeharju's GVerb: input damper, early-reflection taps, a 4-line feedback
delay network with damped lines, and three series allpass diffusers per output channel.

Mono in, stereo out. Delay lengths follow `roomsize` at control rate, rounded as in GVerb's
`gverb_set_roomsize`; GVerb's constructor rounds the FDN lengths to primes instead. The diffusers
also follow `roomsize` and `spread`, where GVerb fixes them at construction.
"""

from cignal import Array, Delay, Kernel, f32, f64, i32, param

MAX_ROOM = 50.0  # metres
MAX_RATE = 96000
SOUND_SPEED = 340.0  # m/s
MAX_DELAY = int(MAX_ROOM * MAX_RATE / SOUND_SPEED) + 1
FDN_RATIOS = (1.0, 0.81649, 0.7071, 0.63245)  # 1, sqrt(2/3), sqrt(1/2), sqrt(2/5)
TAP_RATIOS = (0.41, 0.3, 0.155, 0.0)
DIFF_TOTAL = 1341.0  # 210 + 159 + 562 + 410: the left diffuser lengths at zero spread
DIFF_MAX = MAX_DELAY * 2 // 3  # the longest diffuser is below FDN_RATIOS[3] * MAX_DELAY


class Damper(Kernel):
    damping: f32 = 0.5
    y: f32 = 0.0

    def tick(self, x: f32) -> f32:
        self.y = x * (1.0 - self.damping) + self.y * self.damping
        return self.y


class Diffuser(Kernel):
    coeff: f32 = 0.75
    length: i32 = 1
    line: Delay[f32, DIFF_MAX]

    def tick(self, x: f32) -> f32:
        old = self.line.read(self.length)
        w = x - old * self.coeff
        self.line.write(w)
        return old + w * self.coeff


class Gigaverb(Kernel):
    roomsize: f32 = param(10.0, lo=1.0, hi=MAX_ROOM)  # metres
    revtime: f32 = param(11.0, lo=0.1, hi=60.0)  # seconds to -60 dB
    damping: f32 = param(0.7, lo=0.0, hi=1.0)
    spread: f32 = param(23.0, lo=0.0, hi=100.0)
    bandwidth: f32 = param(0.5, lo=0.0, hi=1.0)
    dry: f32 = param(1.0, lo=0.0, hi=1.0)
    early: f32 = param(0.25, lo=0.0, hi=1.0)
    tail: f32 = param(0.25, lo=0.0, hi=1.0)

    input_damper: Damper
    fdn_dampers: Array[Damper, 4]
    fdn_lines: Array[Delay[f32, MAX_DELAY + 1], 4]
    fdn_lens: Array[i32, 4]
    fdn_gains: Array[f32, 4]
    tap_line: Delay[f32, MAX_DELAY // 2 + 6]
    taps: Array[i32, 4]
    tap_gains: Array[f32, 4]
    ldifs: Array[Diffuser, 4]
    rdifs: Array[Diffuser, 3]  # GVerb's rdifs[1..3]; its rdifs[0] is never read
    u: Array[f32, 4]
    d: Array[f32, 4]

    def init(self) -> None:
        self.ldifs[2].coeff = 0.625
        self.ldifs[3].coeff = 0.625
        self.rdifs[1].coeff = 0.625
        self.rdifs[2].coeff = 0.625

    def spread_lengths(self, r1: f32, r2: f32, scale: f32) -> tuple[i32, i32, i32]:
        c = 369.0 + self.spread * r1
        d = 931.0 + 3.0 * self.spread * r2
        return i32(scale * (c - 210.0)), i32(scale * (d - c)), i32(scale * (DIFF_TOTAL - d))

    def control(self) -> None:
        largest = self.sample_rate * self.roomsize / SOUND_SPEED
        # f64: 1 - alpha is about 1.3e-5, so f32 misstates the decay rate by 0.22% and the float
        # build's 10 s tail fell to 68 dB SNR against the double build.
        alpha = f64(0.001) ** (1.0 / (f64(self.sample_rate) * f64(self.revtime)))
        for i in range(4):
            self.fdn_lens[i] = i32(FDN_RATIOS[i] * largest + 0.5)
            self.fdn_gains[i] = f32(-(alpha ** f64(self.fdn_lens[i])))
            self.taps[i] = 5 + i32(TAP_RATIOS[i] * largest + 0.5)
            self.tap_gains[i] = f32(alpha ** f64(self.taps[i]))
            self.fdn_dampers[i].damping = self.damping
        self.input_damper.damping = 1.0 - self.bandwidth

        scale = f32(self.fdn_lens[3]) / DIFF_TOTAL
        self.ldifs[0].length = i32(scale * 210.0)
        self.ldifs[1].length, self.ldifs[2].length, self.ldifs[3].length = self.spread_lengths(
            0.125541, 0.854046, scale
        )
        self.rdifs[0].length, self.rdifs[1].length, self.rdifs[2].length = self.spread_lengths(
            -0.568366, -0.126815, scale
        )

    def tick(self, x: f32) -> tuple[f32, f32]:
        s = x
        if s != s or abs(s) > 100000.0:
            s = 0.0
        z = self.ldifs[0].tick(self.input_damper.tick(s))

        for i in range(4):
            self.u[i] = self.tap_gains[i] * self.tap_line.read(self.taps[i])
        self.tap_line.write(z)
        for i in range(4):
            self.d[i] = self.fdn_dampers[i].tick(self.fdn_gains[i] * self.fdn_lines[i].read(self.fdn_lens[i]))

        acc = s * self.early
        sign = 1.0
        for i in range(4):
            acc += sign * (self.tail * self.d[i] + self.early * self.u[i])
            sign = -sign

        # GVerb's 4x4 feedback matrix: orthogonal, entries +-0.5.
        d0 = self.d[0]
        d1 = self.d[1]
        d2 = self.d[2]
        d3 = self.d[3]
        self.fdn_lines[0].write(self.u[0] + 0.5 * (d0 + d1 - d2 - d3))
        self.fdn_lines[1].write(self.u[1] + 0.5 * (d0 - d1 - d2 + d3))
        self.fdn_lines[2].write(self.u[2] + 0.5 * (-d0 + d1 - d2 + d3))
        self.fdn_lines[3].write(self.u[3] + 0.5 * (d0 + d1 + d2 + d3))

        left = acc
        right = acc
        for i in range(1, 4):
            left = self.ldifs[i].tick(left)
        for i in range(3):
            right = self.rdifs[i].tick(right)
        return x * self.dry + left, x * self.dry + right
