# SIMD experiment (spec 13.6)

Date: 2026-11-23. Question: does SIMD for IIR kernels need an ABI change before v1.0?

Answer: no. `Array[K, N]` in one kernel already gives the parallelism. Hosts use only the struct size and the functions, so a struct-of-arrays layout would break no current host. Whether field layout becomes ABI is left to the v1.0 freeze.

## Method

Two kernels, 8 voices each:

- **Bank**: 8 biquad lowpass sections at fixed frequencies, outputs summed. Arithmetic only.
- **Ladders**: 8 copies of the `tests/fixtures/ladder.py` voice (4 stages, 2x oversampled, 10 `tanhf` per sample per voice), cutoff spread by 5% per voice.

Three layouts per kernel:

- **AoS**: `voices: Array[Voice, 8]` in one kernel, as cignal emits it today (array of structs).
- **SoA**: hand-written C, one array per field across voices (struct of arrays), with the voice loop innermost and the same operation order per voice.
- **8 inst.**: 8 separate instances of the one-voice kernel, as a host runs polyphony today.

The harness ran 4096 frames in blocks of 64 and took the best of 5 runs. It compared every variant with AoS bit for bit. Machine: Apple silicon (arm64), macOS. GCC 14.2 ran in an arm64 Linux container on the same machine.

## Results

ns per frame, all 8 voices.

| | clang `-O2` | clang `-O3 -mcpu=native` | GCC `-O2` | GCC `-O3 -mcpu=native` |
|-|-|-|-|-|
| Bank, AoS | 8.4 | 8.8 | 25.6 | 6.5 |
| Bank, SoA | 4.4 | 5.6 | 7.4 | 4.3 |
| Bank, 8 inst. | 24.0 | 27.5 | 25.5 | 22.3 |
| Ladders, AoS | 345 | 320 | 762 | 629 |
| Ladders, SoA | 299 | 261 | 537 | 524 |
| Ladders, 8 inst. | 755 | 677 | 1098 | 1035 |

`-O2` builds used `-ffp-contract=off`; `-O3` builds left contraction on. Every variant matched AoS bit for bit, except under `-fveclib=Accelerate`: Ladders SoA then ran at 262 ns, with different output. Compare GCC numbers within their column only: the libm and OS differ.

clang's optimisation remarks (`-Rpass=slp-vectorizer`) showed the SoA bank body packed by the SLP vectorizer (cost -23). The loop vectorizer did not fire. No assembly was read beyond the remarks.

## Findings

1. **Layout is not ABI in practice.** The Pd and SuperCollider shims use `sizeof`, `info.size` and the four functions. None reads a field.
2. **AoS in one instance beats 8 instances by 2.2 to 3.4x** on clang `-O2` and GCC `-O3`. One biquad is a single chain of dependent operations; 8 independent chains overlap in the pipeline. That the gain is latency and not throughput is an inference from the per-sample cost, not a measurement. Multi-channel instances would add nothing beyond `Array[Voice, N]` in a wrapper kernel.
3. **SoA adds 1.5 to 1.9x for arithmetic kernels** (3.5x on GCC `-O2`). It stays bit-exact, because each lane does the same IEEE operations in the same order.
4. **Transcendental-heavy kernels gain 1.2 to 1.4x.** Scalar `tanhf` calls dominate. A vector libm gave no further gain and broke bit-exactness.
5. **GCC `-O2` does not interleave the loop over sub-kernels**: AoS and 8 instances both run at 25.5 ns. Full unrolling of small constant `range` loops in the compiler would likely fix it (untested).

## Unrolling

Follow-up to finding 5, same harness. `#pragma GCC unroll N` on every constant loop, ns per frame:

| | GCC `-O2` before | after | GCC `-O3 -mcpu=native` before | after | clang `-O2` before | after |
|-|-|-|-|-|-|-|
| Bank, AoS | 25.4 | 6.5 | 6.5 | 6.5 | 8.1 | 8.1 |
| Ladders, AoS | 764 | 609 | 631 | 603 | 338 | 378 |

Unrolling only the 8-voice loops left clang's ladders about 10% slower. The compiler therefore emits `CIGNAL_UNROLL(n)`, which expands to the pragma only under GCC 8 or later. Freeverb under GCC `-O2`: 13.1 to 11.2 ns/frame. Unrolling does not reorder floating-point operations, so output should not change; CI's GCC differential tests check this bit for bit.

## Limits

- One machine and one architecture. No x86 measurement.
- One session of runs; machine load was not controlled.
- SoA was hand-written here. The compiler emits it since 2026-11-26 ([soa.md](soa.md)).
- The harness lived in a session scratchpad and was not kept.
