"""Evaluation order: each statement below has a method call plus another operand that touches `self.n`.

CPython evaluates operands, arguments and `range` bounds left to right. C leaves operand and argument
order unspecified, so the generated code must fix it. `range` stops are hoisted, so they must not move ahead of
their start.
"""

from cignal import Array, Kernel, f32, i32


class Counter(Kernel):
    n: i32 = 0

    def next(self) -> i32:
        self.n = (self.n * 3 + 1) % 1000
        return self.n


class Order(Kernel):
    n: i32 = 0
    acc: i32 = 0
    taps: Array[i32, 4]
    counter: Counter

    def bump(self, k: i32) -> i32:
        self.n = (self.n * 10 + k) % 100000
        return k

    def tick(self, x: f32) -> f32:
        a = self.bump(1) - self.bump(2)  # i32 `-` lowers to a helper: its operands are call arguments
        b = self.n + self.bump(3)  # read before the call
        c = self.bump(4) * 10 + self.n  # read after the call
        self.acc += self.bump(5)  # the target is read before the value
        self.acc = self.acc % 1000
        self.taps[self.n % 4] = self.bump(6)  # the index is evaluated after the value
        d = self.counter.next() - self.counter.n
        flag = self.bump(7) > 0 and self.n % 2 == 1
        e = self.bump(8) if flag else self.n
        total = 0
        for i in range(self.bump(1), self.n % 7 + self.bump(2)):
            total += i
        k = 0
        while self.bump(1) + self.n % 3 > k:
            k += 1
        if self.n % 5 == 0:
            k += 10
        elif self.bump(9) > self.n % 10:
            k += 20
        h = a + b + c + self.acc + self.taps[0] + self.taps[1] * 3 + self.taps[2] * 5 + d + e + total + k
        return x + f32(h % 4096)
