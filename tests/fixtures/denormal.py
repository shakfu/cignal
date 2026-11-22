"""Shows whether the kernel runs with denormals flushed to zero.

`x * 1e-39` is denormal; scaling it by 1e38 gives about `0.1 * x`, a normal value that survives any
output stage. With flush-to-zero the intermediate becomes 0, and so does the result.
"""

from cignal import Kernel, f32


class Denormal(Kernel):
    def tick(self, x: f32) -> f32:
        return x * 1e-39 * 1e38
