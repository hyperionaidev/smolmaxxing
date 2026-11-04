// gdn_test: the fused one-pass DeltaNet update (exact NEON code from the ggml patch) vs the stock 5-pass float
// update, both against a double-precision reference, 2000 recurrent steps on a 128x128 state. aarch64 only.
// cc -O2 -o gdn_test gdn_test.c -lm && ./gdn_test
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define S 128
static float dot(const float *a, const float *b) { float s = 0; for (int i = 0; i < S; i++) s += a[i] * b[i]; return s; }
static double ddot(const double *a, const double *b) { double s = 0; for (int i = 0; i < S; i++) s += a[i] * b[i]; return s; }
#include <arm_neon.h>
static float fused_row(const float *r, float *w, const float *k_d, const float *q_d, float gs, float dj) {
    const float32x4_t vg = vdupq_n_f32(gs), vd = vdupq_n_f32(dj);
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0; int i = 0;
    for (; i + 16 <= S; i += 16) {
        float32x4_t r0 = vfmaq_f32(vmulq_f32(vld1q_f32(r + i +  0), vg), vld1q_f32(k_d + i +  0), vd);
        float32x4_t r1 = vfmaq_f32(vmulq_f32(vld1q_f32(r + i +  4), vg), vld1q_f32(k_d + i +  4), vd);
        float32x4_t r2 = vfmaq_f32(vmulq_f32(vld1q_f32(r + i +  8), vg), vld1q_f32(k_d + i +  8), vd);
        float32x4_t r3 = vfmaq_f32(vmulq_f32(vld1q_f32(r + i + 12), vg), vld1q_f32(k_d + i + 12), vd);
        vst1q_f32(w + i + 0, r0); vst1q_f32(w + i + 4, r1); vst1q_f32(w + i + 8, r2); vst1q_f32(w + i + 12, r3);
        a0 = vfmaq_f32(a0, r0, vld1q_f32(q_d + i +  0)); a1 = vfmaq_f32(a1, r1, vld1q_f32(q_d + i +  4));
        a2 = vfmaq_f32(a2, r2, vld1q_f32(q_d + i +  8)); a3 = vfmaq_f32(a3, r3, vld1q_f32(q_d + i + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}
static float rnd(void) { return (float) rand() / RAND_MAX * 2 - 1; }
int main(void) {
    static float s1[S * S], s2[S * S], o1[S], o2[S]; static double sd[S * S], od[S]; float delta[S];
    for (int i = 0; i < S * S; i++) { s1[i] = s2[i] = rnd() * 0.1f; sd[i] = s1[i]; }
    double maxd_stock = 0, maxd_new = 0;
    for (int t = 0; t < 2000; t++) {
        float q[S], k[S], v[S]; double qd[S], kd[S];
        float nk = 0; for (int i = 0; i < S; i++) { q[i] = rnd(); k[i] = rnd(); v[i] = rnd(); nk += k[i] * k[i]; }
        for (int i = 0; i < S; i++) { k[i] /= sqrtf(nk); qd[i] = q[i]; kd[i] = k[i]; }   // l2-normalized k like the model
        float g = -0.05f * (rnd() + 1.2f), beta = 0.5f * (rnd() + 1), gs = expf(g), sc = 1 / sqrtf(S);
        // stock (float)
        for (int i = 0; i < S * S; i++) s1[i] *= gs;
        for (int j = 0; j < S; j++) delta[j] = (v[j] - dot(&s1[j * S], k)) * beta;
        for (int j = 0; j < S; j++) for (int i = 0; i < S; i++) s1[j * S + i] += k[i] * delta[j];
        for (int j = 0; j < S; j++) o1[j] = dot(&s1[j * S], q) * sc;
        // fused (float)
        for (int j = 0; j < S; j++) { float *r = &s2[j * S]; float d = dot(r, k); float dj = (v[j] - gs * d) * beta; o2[j] = fused_row(r, r, k, q, gs, dj) * sc; }
        // reference (double)
        for (int j = 0; j < S; j++) { double *r = &sd[j * S]; double d = ddot(r, kd); double dj = (v[j] - (double) gs * d) * beta; double o = 0;
            for (int i = 0; i < S; i++) { double x = gs * r[i] + dj * kd[i]; r[i] = x; o += x * qd[i]; } od[j] = o * sc; }
        for (int j = 0; j < S; j++) { double a = fabs(o1[j] - od[j]), b = fabs(o2[j] - od[j]); if (a > maxd_stock) maxd_stock = a; if (b > maxd_new) maxd_new = b; }
        if (t == 0 || t == 9 || t == 99 || t == 1999) {
            double m = 0; for (int j = 0; j < S; j++) if (fabs(od[j]) > m) m = fabs(od[j]);
            printf("t=%4d  |out|max %.3f   max abs err vs double: stock %.2e  fused %.2e\n", t + 1, m, maxd_stock, maxd_new);
        }
    }
    return 0;
}
