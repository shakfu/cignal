"""Struct-of-arrays nesting (docs/dev/soa.md): what an element kernel of an `Array[K, N]` may itself hold.

`Lane` holds a plain sub-kernel, an `Array` of sub-kernels, an `Array` of floats, a delay line and an `Array` of
delay lines, has its own `init`, and calls its own method. `Tap` is held both plainly and in arrays.
"""

from cignal import Array, Delay, Kernel, f32, i32, param


class Tap(Kernel):
    y: f32 = 0.0
    k: f32 = 0.5

    def tick(self, x: f32) -> f32:
        self.y += self.k * (x - self.y)
        return self.y


class Lane(Kernel):
    smooth: Tap
    taps: Array[Tap, 2]
    gains: Array[f32, 4]  # not 3, the lane count, so a swapped declaration would not fit
    line: Delay[f32, 64]
    lines: Array[Delay[f32, 16], 2]
    count: i32 = 0
    seed: f32 = 0.0

    def init(self) -> None:
        self.seed = 1.0 / self.sample_rate
        for j in range(4):
            self.gains[j] = 0.25 * f32(j + 1)

    def mixed(self, x: f32) -> f32:
        acc = 0.0
        for j in range(2):
            acc += self.taps[j].tick(x) * self.gains[j]
        return acc + self.gains[3] * self.lines[1].read(9)

    def tick(self, x: f32, depth: f32) -> f32:
        self.count = (self.count + 1) % 1000
        self.line.write(x + self.seed)
        self.lines[0].write(self.line.read(7))
        self.lines[1].write(self.lines[0].read(5))
        return self.smooth.tick(self.mixed(self.line.read_frac(depth)))


class Nest(Kernel):
    depth: f32 = param(10.0, lo=1.0, hi=60.0)
    lanes: Array[Lane, 3]
    plain: Tap

    def tick(self, x: f32) -> f32:
        acc = self.plain.tick(x)
        for i in range(3):
            acc += self.lanes[i].tick(x * f32(i + 1), self.depth + f32(self.lanes[i].count % 7))
        return acc
