/* Hand-written reference for tests/fixtures/biquad.py: state cached in locals across the block. */
#ifndef REF_BIQUAD_H
#define REF_BIQUAD_H
typedef struct RefBiquad { float b0, b1, b2, a1, a2, z1, z2; } RefBiquad;
enum { RefBiquad_N_INPUTS = 1, RefBiquad_N_OUTPUTS = 1 };
void RefBiquad_init(RefBiquad *s, float sample_rate);
void RefBiquad_process(RefBiquad *s, const float *const *in, float *const *out, int n);
#endif
