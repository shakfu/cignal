"""Time every fixture kernel, and the hand-written references, in ns per sample frame.

Usage: uv run python bench/run.py [SECONDS]. Each figure is the minimum of RUNS runs of SECONDS each.
"""

import platform
import subprocess
import sys
import tempfile
from importlib import resources
from pathlib import Path
from string import Template

from cignal.build import cc, compile_kernel, write_kernel

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
REF = ROOT / "bench" / "ref"
RUNS = 5
KERNELS = ["Biquad", "OnePole", "Mix", "Lowpass", "Saw", "Stereo", "Echo", "Comb", "SVF", "Fir", "Drive",
           "Wavetable", "Glide", "Ops"]  # fmt: skip
# Hand-written equivalents: generated kernel -> (reference name, source stem).
REFERENCES = {"Biquad": ("RefBiquad", "ref_biquad"), "Fir": ("RefFir", "ref_fir")}
NATIVE = "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"
FLAGS = {"O2": ["-std=c99", "-O2"], f"O3 {NATIVE}": ["-std=c99", "-O3", NATIVE]}


def _bench_source(name: str, header: str, n_inputs: int, n_outputs: int) -> str:
    template = resources.files("cignal.shims").joinpath("bench.c.in").read_text()
    return Template(template).substitute(name=name, header=header, n_inputs=n_inputs, n_outputs=n_outputs)


def _time(sources: list[Path], include: Path, flags: list[str], work: Path, seconds: float) -> float:
    exe = work / "bench"
    subprocess.run([cc(), *flags, "-I", str(include), *map(str, sources), "-o", str(exe), "-lm"], check=True)
    return min(float(subprocess.run([str(exe), str(seconds)], capture_output=True, text=True, check=True).stdout)
               for _ in range(RUNS))  # fmt: skip


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    fixtures = {path.stem: path for path in FIXTURES.glob("*.py")}
    print(f"ns per frame, minimum of {RUNS} runs; {platform.machine()}, {cc()}\n")
    print("| kernel | " + " | ".join(FLAGS) + " | hand-written " + " | hand-written ".join(FLAGS) + " |")
    print("|-|" + "-|" * (2 * len(FLAGS)))
    for kernel in KERNELS:
        path = next(p for p in fixtures.values() if f"class {kernel}(Kernel)" in p.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            ck = compile_kernel(path, kernel)
            _, source = write_kernel(ck, work)
            bench = work / "bench_main.c"
            bench.write_text(_bench_source(ck.c_name, ck.header_name, ck.n_inputs, ck.n_outputs))
            row = [f"{_time([source, bench], work, f, work, seconds):.2f}" for f in FLAGS.values()]
            if kernel in REFERENCES:
                ref, stem = REFERENCES[kernel]
                ref_bench = work / "ref_bench.c"
                ref_bench.write_text(_bench_source(ref, f"{stem}.h", ck.n_inputs, ck.n_outputs))
                row += [f"{_time([REF / f'{stem}.c', ref_bench], REF, f, work, seconds):.2f}" for f in FLAGS.values()]
            else:
                row += ["" for _ in FLAGS]
        print(f"| {kernel} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
