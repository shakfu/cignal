"""Exercises i32/f64/bool state, casts, `not`/`and`, a conditional expression, `%` with negative
operands, and C-keyword names."""

from cignal import Kernel, f32, f64, i32

PERIOD = 100


class Mix(Kernel):
    count: i32 = 0
    gate: bool = False
    level: f64 = 0.0

    def tick(self, x: f32) -> f32:
        double = x * 0.5
        self.count += 1
        if self.count >= PERIOD:
            self.count = 0
            self.gate = not self.gate
        self.level = self.level * 0.99 + f64(x * x) * 0.01
        g = 1.0 if self.gate and self.level > 0.001 else 0.25
        wrapped = (x * 2.0) % -0.75  # exact in f32 and f64, so both builds wrap at the same samples
        k = (self.count - 50) % 7
        if self.gate:
            h = 1.0  # first assigned in a branch, so the C declaration is hoisted
        else:
            h = 2.0
        # Right-nested operands of equal precedence: reassociating any of these changes the rounding.
        assoc = x - (double - x * 0.25) + x / (g * 4.0) + (x + (double + 0.1))
        y = double * g + f32(self.level) - f32(self.count) / 1000.0 + wrapped * 0.1 + f32(k) * 0.01
        return self.shape(y) + h * assoc * 0.01

    def shape(self, v: f32) -> f32:
        """Defined after its caller, so the C needs a prototype."""
        return v - v * v * v / 3.0
