"""Outputs the arguments of its last event: checks how hosts' values convert to each argument type."""

from cignal import Kernel, event, f32, f64, i32


class Latch(Kernel):
    i: i32 = 0
    b: bool = False
    x: f64 = 0.0

    @event
    def on_set(self, i: i32, b: bool, x: f64) -> None:
        self.i = i
        self.b = b
        self.x = x

    @event
    def on_clear(self) -> None:
        self.i = 0

    def tick(self) -> tuple[f32, f32, f32]:
        return f32(f64(self.i) * 1e-9), 1.0 if self.b else 0.0, f32(self.x)
