#include "ref_fir.h"

#include <math.h>

/* Recomputed every block, as the fixture's `control` does, so both do the same work. */
static void coefficients(RefFir *s) {
    float fc = 4000.0f / s->sample_rate;
    for (int i = 0; i < 16; i++) {
        float m = (float)i - 7.5f;
        float window = 0.54f - 0.46f * cosf(6.2831853f * (float)i / 15.0f);
        s->coef[i] = window * sinf(6.2831853f * fc * m) / (3.1415927f * m);
    }
}

void RefFir_init(RefFir *s, float sample_rate) {
    s->sample_rate = sample_rate;
    for (int i = 0; i < 16; i++) s->history[i] = 0.0f;
    s->pos = 0;
}

void RefFir_process(RefFir *s, const float *const *in, float *const *out, int n) {
    coefficients(s);
    const float *x = in[0];
    float *y = out[0];
    int pos = s->pos;
    for (int i = 0; i < n; i++) {
        s->history[pos] = x[i];
        float acc = 0.0f;
        for (int k = 0; k < 16; k++) acc += s->coef[k] * s->history[(pos - k) & 15];
        pos = (pos + 1) & 15;
        y[i] = acc;
    }
    s->pos = pos;
}
