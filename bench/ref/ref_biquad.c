#include "ref_biquad.h"

#include <math.h>

void RefBiquad_init(RefBiquad *s, float sample_rate) {
    float w = 6.2831853f * 1000.0f / sample_rate;
    float alpha = sinf(w) / 1.414f, c = cosf(w), a0 = 1.0f + alpha;
    s->b0 = (1.0f - c) / 2.0f / a0;
    s->b1 = (1.0f - c) / a0;
    s->b2 = s->b0;
    s->a1 = -2.0f * c / a0;
    s->a2 = (1.0f - alpha) / a0;
    s->z1 = s->z2 = 0.0f;
}

void RefBiquad_process(RefBiquad *s, const float *const *in, float *const *out, int n) {
    const float b0 = s->b0, b1 = s->b1, b2 = s->b2, a1 = s->a1, a2 = s->a2;
    float z1 = s->z1, z2 = s->z2;
    const float *x = in[0];
    float *y = out[0];
    for (int i = 0; i < n; i++) {
        float xi = x[i], yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        y[i] = yi;
    }
    s->z1 = z1;
    s->z2 = z2;
}
