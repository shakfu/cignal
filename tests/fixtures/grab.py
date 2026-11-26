"""An event that reads a parameter: `grab` holds `level`. Tests that an event sees a parameter set in its block."""

from cignal import Kernel, event, f32, param


class Grab(Kernel):
    level: f32 = param(0.0, lo=0.0, hi=1.0)
    held: f32 = 0.0

    @event
    def on_grab(self) -> None:
        self.held = self.level

    def tick(self) -> f32:
        return self.held
