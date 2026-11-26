"""Time each kernel that holds `Array[K, N]` in both layouts (docs/dev/soa.md), alternating the builds.

Usage: uv run python bench/layout.py [SECONDS] [RUNS]. Prints a Markdown table of ns per frame, best of RUNS
runs of SECONDS each, and exits 1 if the two layouts' outputs differ. `$CC` picks the compiler.
"""

import array
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from cignal.build import cc, compile_kernel, render_shim, write_kernel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from run import NATIVE, _bench_source  # noqa: E402 - the bench harness of run.py

FIXTURES = ROOT / "tests" / "fixtures"
KERNELS = {"Bank": "bank.py", "Poly": "poly.py", "Synth": "synth.py", "Voices": "voices.py",
           "Freeverb": "freeverb.py", "Gigaverb": "gigaverb.py"}  # fmt: skip
FLAGS = {"-O2": ["-O2"], f"-O3 {NATIVE}": ["-O3", NATIVE]}
FRAMES = 48000


def _build(work: Path, kernel: str, fixture: str, layout: str) -> tuple[dict[str, Path], bytes]:
    """Bench executables per flag set, and the driver's output for one input, for `layout`."""
    ck = compile_kernel(FIXTURES / fixture, kernel, layout=layout)
    d = work / layout
    _, source = write_kernel(ck, d)
    (d / "bench_main.c").write_text(_bench_source(ck.c_name, ck.header_name, ck.n_inputs, ck.n_outputs))
    (d / "drv.c").write_text(render_shim("driver", ck))
    common = ["-std=c99", "-ffp-contract=off", "-I", str(d), str(source)]
    subprocess.run([cc(), *common, "-O2", str(d / "drv.c"), "-o", str(d / "drv"), "-lm"], check=True)
    rng = random.Random(1)
    signal = array.array("f", (rng.uniform(-0.5, 0.5) for _ in range(FRAMES * ck.n_inputs))).tobytes()
    args = [str(d / "drv"), "48000", "64"] + ([] if ck.n_inputs else ["--frames", str(FRAMES)])
    output = subprocess.run(args, input=signal, capture_output=True, check=True).stdout
    benches = {}
    for name, flags in FLAGS.items():
        exe = d / f"bench{len(benches)}"
        subprocess.run([cc(), *common, *flags, str(d / "bench_main.c"), "-o", str(exe), "-lm"], check=True)
        benches[name] = exe
    return benches, output


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    version = subprocess.run([cc(), "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"ns per frame, best of {runs} alternating runs; {platform.machine()}, {version}\n")
    print("| kernel | " + " | ".join(f"{layout} {f}" for f in FLAGS for layout in ("aos", "soa")) + " | same output |")
    print("|-|" + "-|" * (2 * len(FLAGS) + 1))
    same_everywhere = True
    for kernel, fixture in KERNELS.items():
        with tempfile.TemporaryDirectory() as tmp:
            builds = {layout: _build(Path(tmp), kernel, fixture, layout) for layout in ("aos", "soa")}
            best = {(layout, f): float("inf") for layout in builds for f in FLAGS}
            for _ in range(runs):
                for layout, f in best:
                    exe = builds[layout][0][f]
                    out = subprocess.run([str(exe), str(seconds)], capture_output=True, text=True, check=True).stdout
                    best[layout, f] = min(best[layout, f], float(out))
            same = builds["aos"][1] == builds["soa"][1]
            same_everywhere &= same
            cells = " | ".join(f"{best[layout, f]:.2f}" for f in FLAGS for layout in ("aos", "soa"))
            print(f"| {kernel} | {cells} | {'yes' if same else 'NO'} |", flush=True)
    return 0 if same_everywhere else 1


if __name__ == "__main__":
    sys.exit(main())
