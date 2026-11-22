"""Feedback echo with a fractionally interpolated delay time."""

from cignal import Delay, Kernel, f32, f64, param

MAX_DELAY = 4800  # 100 ms at 48 kHz. Sizes are build-time constants.


class Echo(Kernel):
    time: f32 = param(0.05, lo=0.001, hi=0.09)  # seconds
    feedback: f32 = param(0.5, lo=0.0, hi=0.95)
    mix: f32 = param(0.5, lo=0.0, hi=1.0)
    line: Delay[f32, MAX_DELAY]
    samples: f64 = 1.0  # a delay position needs more precision than its samples

    def control(self) -> None:
        self.samples = f64(self.time) * f64(self.sample_rate)

    def tick(self, x: f32) -> f32:
        delayed = self.line.read_frac(self.samples)
        self.line.write(x + self.feedback * delayed)
        return x + self.mix * (delayed - x)
