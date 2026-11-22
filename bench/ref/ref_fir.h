/* Hand-written reference for tests/fixtures/fir.py: a 16-tap ring buffer indexed with a mask,
   coefficients recomputed per block. */
#ifndef REF_FIR_H
#define REF_FIR_H
typedef struct RefFir { float coef[16]; float history[16]; int pos; float sample_rate; } RefFir;
enum { RefFir_N_INPUTS = 1, RefFir_N_OUTPUTS = 1 };
void RefFir_init(RefFir *s, float sample_rate);
void RefFir_process(RefFir *s, const float *const *in, float *const *out, int n);
#endif
