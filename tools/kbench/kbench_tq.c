// kbench_tq: ternary TQ1_0 / TQ2_0 x Q8_K mat-vec on the Cortex-A55 (dotprod, no i8mm).
// plain = llama.cpp v0.8.0's NEON vec_dot (one row at a time); r4 = 8 rows interleaved 4 bytes at a time,
// vdotq_laneq against the activation (one decoded 16-byte vector serves 4 rows' worth of dot products).
// Checks against a scalar reference, then G weights/s for a cache-resident and a DRAM-sized matrix.
// usage: kbench_tq
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define QK_K 256
typedef struct { uint8_t qs[48]; uint8_t qh[4]; __fp16 d; } block_tq1_0;
typedef struct { uint8_t qs[64]; __fp16 d; } block_tq2_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } block_q8_K;
typedef struct { __fp16 d[8]; uint8_t qs[416]; } block_tq1_0x8;   // qs[c*32 + j*4 + i] = row j's byte 4c+i of (qs|qh), c = 0..12
typedef struct { __fp16 d[8]; uint8_t qs[512]; } block_tq2_0x8;   // qs[c*32 + j*4 + i] = row j's qs[4c+i], c = 0..15

// ---------- scalar reference ----------
static const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };
static double ref_tq1(int n, const block_tq1_0 *x, const block_q8_K *y) {
    double s = 0;
    for (int b = 0; b < n / QK_K; b++) {
        long acc = 0; int e = 0;
        for (int k = 0; k < 5; k++) for (int j = 0; j < 32; j++) { uint8_t q = x[b].qs[j] * pow3[k]; acc += (long) (((uint16_t) q * 3 >> 8) - 1) * y[b].qs[k * 32 + j]; }
        for (int k = 0; k < 5; k++) for (int j = 0; j < 16; j++) { uint8_t q = x[b].qs[32 + j] * pow3[k]; acc += (long) (((uint16_t) q * 3 >> 8) - 1) * y[b].qs[160 + k * 16 + j]; }
        for (int k = 0; k < 4; k++) for (int j = 0; j < 4; j++) { uint8_t q = x[b].qh[j] * pow3[k]; acc += (long) (((uint16_t) q * 3 >> 8) - 1) * y[b].qs[240 + k * 4 + j]; }
        (void) e; s += (double) acc * (double) x[b].d * y[b].d;
    }
    return s;
}
static double ref_tq2(int n, const block_tq2_0 *x, const block_q8_K *y) {
    double s = 0;
    for (int b = 0; b < n / QK_K; b++) {
        long acc = 0;
        for (int j = 0; j < 64; j += 32) for (int l = 0; l < 4; l++) for (int m = 0; m < 32; m++)
            acc += (long) (((x[b].qs[j + m] >> (2 * l)) & 3) - 1) * y[b].qs[j * 4 + l * 32 + m];
        s += (double) acc * (double) x[b].d * y[b].d;
    }
    return s;
}

// ---------- plain (llama.cpp v0.8.0 NEON dotprod path, logic unchanged) ----------
#define T3(v) vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8((v), vshrq_n_u8((v), 1)), 6))
static float dot_tq1_plain(int n, const block_tq1_0 *x, const block_q8_K *y) {
    const int nb = n / QK_K; float sumf = 0;
    const uint8_t k_shift[16] = { 1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27 };
    const uint8x16_t shift = vld1q_u8(k_shift);
    for (int i = 0; i < nb; ++i) {
        int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
        { uint8x16_t q0 = vld1q_u8(x[i].qs), q1 = vld1q_u8(x[i].qs + 16);
          uint8x16_t q2 = vmulq_u8(q0, vdupq_n_u8(3)), q3 = vmulq_u8(q1, vdupq_n_u8(3)), q4 = vmulq_u8(q0, vdupq_n_u8(9)), q5 = vmulq_u8(q1, vdupq_n_u8(9));
          uint8x16_t q6 = vmulq_u8(q0, vdupq_n_u8(27)), q7 = vmulq_u8(q1, vdupq_n_u8(27)), q8 = vmulq_u8(q0, vdupq_n_u8(81)), q9 = vmulq_u8(q1, vdupq_n_u8(81));
          const int8_t *py = y[i].qs;
          s0 = vdotq_s32(s0, T3(q0), vld1q_s8(py)); s1 = vdotq_s32(s1, T3(q1), vld1q_s8(py + 16));
          s0 = vdotq_s32(s0, T3(q2), vld1q_s8(py + 32)); s1 = vdotq_s32(s1, T3(q3), vld1q_s8(py + 48));
          s0 = vdotq_s32(s0, T3(q4), vld1q_s8(py + 64)); s1 = vdotq_s32(s1, T3(q5), vld1q_s8(py + 80));
          s0 = vdotq_s32(s0, T3(q6), vld1q_s8(py + 96)); s1 = vdotq_s32(s1, T3(q7), vld1q_s8(py + 112));
          s0 = vdotq_s32(s0, T3(q8), vld1q_s8(py + 128)); s1 = vdotq_s32(s1, T3(q9), vld1q_s8(py + 144)); }
        { uint8x16_t q0 = vld1q_u8(x[i].qs + 32);
          uint8x16_t q1 = vmulq_u8(q0, vdupq_n_u8(3)), q2 = vmulq_u8(q0, vdupq_n_u8(9)), q3 = vmulq_u8(q0, vdupq_n_u8(27)), q4 = vmulq_u8(q0, vdupq_n_u8(81));
          uint32_t qh; memcpy(&qh, x[i].qh, 4);
          uint8x16_t q5 = vmulq_u8(vreinterpretq_u8_u32(vdupq_n_u32(qh)), shift);
          const int8_t *py = y[i].qs;
          s0 = vdotq_s32(s0, T3(q0), vld1q_s8(py + 160)); s1 = vdotq_s32(s1, T3(q1), vld1q_s8(py + 176));
          s0 = vdotq_s32(s0, T3(q2), vld1q_s8(py + 192)); s1 = vdotq_s32(s1, T3(q3), vld1q_s8(py + 208));
          s0 = vdotq_s32(s0, T3(q4), vld1q_s8(py + 224)); s1 = vdotq_s32(s1, T3(q5), vld1q_s8(py + 240)); }
        const int16x8_t ys0 = vld1q_s16(y[i].bsums), ys1 = vld1q_s16(y[i].bsums + 8);
        s0 = vaddq_s32(s0, s1); s0 = vsubq_s32(s0, vpaddlq_s16(vaddq_s16(ys0, ys1)));
        sumf += (float) x[i].d * y[i].d * (float) vaddvq_s32(s0);
    }
    return sumf;
}
static float dot_tq2_plain(int n, const block_tq2_0 *x, const block_q8_K *y) {
    const int nb = n / QK_K; float sumf = 0; const uint8x16_t m3 = vdupq_n_u8(3);
    for (int i = 0; i < nb; ++i) {
        int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);
        for (int j = 0; j < 64; j += 32) {
            uint8x16_t q0 = vld1q_u8(x[i].qs + j), q1 = vld1q_u8(x[i].qs + j + 16);
            const int8_t *py = y[i].qs + j * 4;
            s0 = vdotq_s32(s0, vreinterpretq_s8_u8(vandq_u8(q0, m3)), vld1q_s8(py));
            s1 = vdotq_s32(s1, vreinterpretq_s8_u8(vandq_u8(q1, m3)), vld1q_s8(py + 16));
            s0 = vdotq_s32(s0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 2), m3)), vld1q_s8(py + 32));
            s1 = vdotq_s32(s1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 2), m3)), vld1q_s8(py + 48));
            s0 = vdotq_s32(s0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 4), m3)), vld1q_s8(py + 64));
            s1 = vdotq_s32(s1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 4), m3)), vld1q_s8(py + 80));
            s0 = vdotq_s32(s0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q0, 6), m3)), vld1q_s8(py + 96));
            s1 = vdotq_s32(s1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q1, 6), m3)), vld1q_s8(py + 112));
        }
        const int16x8_t ys0 = vld1q_s16(y[i].bsums), ys1 = vld1q_s16(y[i].bsums + 8);
        s0 = vaddq_s32(s0, s1); s0 = vsubq_s32(s0, vpaddlq_s16(vaddq_s16(ys0, ys1)));
        sumf += (float) x[i].d * y[i].d * (float) vaddvq_s32(s0);
    }
    return sumf;
}
static void gemv_tq1_plain(int n, float *s, const void *vx, const void *vy, int nc) { for (int r = 0; r < nc; r++) s[r] = dot_tq1_plain(n, (const block_tq1_0 *) vx + (size_t) r * (n / QK_K), vy); }
static void gemv_tq2_plain(int n, float *s, const void *vx, const void *vy, int nc) { for (int r = 0; r < nc; r++) s[r] = dot_tq2_plain(n, (const block_tq2_0 *) vx + (size_t) r * (n / QK_K), vy); }

// ---------- r4 repacked ----------
static void repack_tq1(const block_tq1_0 *in, block_tq1_0x8 *out, int nrows, int nb) {
    for (int x = 0; x < nrows / 8; x++) for (int b = 0; b < nb; b++) { block_tq1_0x8 *o = &out[x * nb + b];
        for (int j = 0; j < 8; j++) { const block_tq1_0 *r = &in[(x * 8 + j) * nb + b]; o->d[j] = r->d;
            for (int c = 0; c < 12; c++) memcpy(&o->qs[c * 32 + j * 4], &r->qs[c * 4], 4);
            memcpy(&o->qs[12 * 32 + j * 4], r->qh, 4); } }
}
static void repack_tq2(const block_tq2_0 *in, block_tq2_0x8 *out, int nrows, int nb) {
    for (int x = 0; x < nrows / 8; x++) for (int b = 0; b < nb; b++) { block_tq2_0x8 *o = &out[x * nb + b];
        for (int j = 0; j < 8; j++) { const block_tq2_0 *r = &in[(x * 8 + j) * nb + b]; o->d[j] = r->d;
            for (int c = 0; c < 16; c++) memcpy(&o->qs[c * 32 + j * 4], &r->qs[c * 4], 4); } }
}
// TQ1_0: chunk c < 8 -> trit k of elements 32k + 4c..; 8 <= c < 12 -> 160 + 16k + 4(c-8)..; c = 12 (qh) -> 240 + 4k + (0..3)
static void gemv_tq1_r4(int n, float *s, const void *vx, const void *vy, int nc) {
    const int nb = n / QK_K; const block_q8_K *y = vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_tq1_0x8 *w = (const block_tq1_0x8 *) vx + (size_t) x * nb;
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            const int8_t *q8 = y[b].qs; const uint8_t *qs = w[b].qs;
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, e0 = a0, e1 = a0;
#define TQ1_CH(QC, A, L) { const uint8x16_t u0 = vld1q_u8(QC), u1 = vld1q_u8((QC) + 16); \
            a0 = vdotq_laneq_s32(a0, T3(u0), A[0], L); a1 = vdotq_laneq_s32(a1, T3(u1), A[0], L); \
            uint8x16_t v0 = vmulq_u8(u0, vdupq_n_u8(3)), v1 = vmulq_u8(u1, vdupq_n_u8(3)); \
            e0 = vdotq_laneq_s32(e0, T3(v0), A[1], L); e1 = vdotq_laneq_s32(e1, T3(v1), A[1], L); \
            v0 = vmulq_u8(u0, vdupq_n_u8(9)); v1 = vmulq_u8(u1, vdupq_n_u8(9)); \
            a0 = vdotq_laneq_s32(a0, T3(v0), A[2], L); a1 = vdotq_laneq_s32(a1, T3(v1), A[2], L); \
            v0 = vmulq_u8(u0, vdupq_n_u8(27)); v1 = vmulq_u8(u1, vdupq_n_u8(27)); \
            e0 = vdotq_laneq_s32(e0, T3(v0), A[3], L); e1 = vdotq_laneq_s32(e1, T3(v1), A[3], L); \
            v0 = vmulq_u8(u0, vdupq_n_u8(81)); v1 = vmulq_u8(u1, vdupq_n_u8(81)); \
            a0 = vdotq_laneq_s32(a0, T3(v0), A[4], L); a1 = vdotq_laneq_s32(a1, T3(v1), A[4], L); }
            for (int g = 0; g < 2; g++) {   // qs 0..31: elements 32k + 16g + 4*lane
                int8x16_t A[5]; for (int k = 0; k < 5; k++) A[k] = vld1q_s8(q8 + 32 * k + 16 * g);
                const uint8_t *qc = qs + g * 4 * 32;
                TQ1_CH(qc, A, 0) TQ1_CH(qc + 32, A, 1) TQ1_CH(qc + 64, A, 2) TQ1_CH(qc + 96, A, 3)
            }
            { int8x16_t A[5]; for (int k = 0; k < 5; k++) A[k] = vld1q_s8(q8 + 160 + 16 * k);   // qs 32..47
              const uint8_t *qc = qs + 8 * 32;
              TQ1_CH(qc, A, 0) TQ1_CH(qc + 32, A, 1) TQ1_CH(qc + 64, A, 2) TQ1_CH(qc + 96, A, 3) }
#undef TQ1_CH
            { const int8x16_t A = vld1q_s8(q8 + 240);   // qh: trit k of the 4 bytes -> elements 240 + 4k + (0..3) = lane k
              const uint8x16_t u0 = vld1q_u8(qs + 12 * 32), u1 = vld1q_u8(qs + 12 * 32 + 16);
              a0 = vdotq_laneq_s32(a0, T3(u0), A, 0); a1 = vdotq_laneq_s32(a1, T3(u1), A, 0);
              uint8x16_t v0 = vmulq_u8(u0, vdupq_n_u8(3)), v1 = vmulq_u8(u1, vdupq_n_u8(3));
              e0 = vdotq_laneq_s32(e0, T3(v0), A, 1); e1 = vdotq_laneq_s32(e1, T3(v1), A, 1);
              v0 = vmulq_u8(u0, vdupq_n_u8(9)); v1 = vmulq_u8(u1, vdupq_n_u8(9));
              a0 = vdotq_laneq_s32(a0, T3(v0), A, 2); a1 = vdotq_laneq_s32(a1, T3(v1), A, 2);
              v0 = vmulq_u8(u0, vdupq_n_u8(27)); v1 = vmulq_u8(u1, vdupq_n_u8(27));
              e0 = vdotq_laneq_s32(e0, T3(v0), A, 3); e1 = vdotq_laneq_s32(e1, T3(v1), A, 3); }
            const int32_t ysum = vaddlvq_s16(vaddq_s16(vld1q_s16(y[b].bsums), vld1q_s16(y[b].bsums + 8)));
            const int32x4_t ys = vdupq_n_s32(ysum);
            const float32x4_t yd = vdupq_n_f32(y[b].d);
            f0 = vfmaq_f32(f0, vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].d)), yd),     vcvtq_f32_s32(vsubq_s32(vaddq_s32(a0, e0), ys)));
            f1 = vfmaq_f32(f1, vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].d + 4)), yd), vcvtq_f32_s32(vsubq_s32(vaddq_s32(a1, e1), ys)));
        }
        vst1q_f32(s + x * 8, f0); vst1q_f32(s + x * 8 + 4, f1);
    }
}
// TQ2_0: like Q2_K (chunk c = 8h + 4g + lane: elements h*128 + 32s + 16g + 4*lane), one scale per row
static void gemv_tq2_r4(int n, float *s, const void *vx, const void *vy, int nc) {
    const int nb = n / QK_K; const block_q8_K *y = vy;
    const uint8x16_t m03 = vdupq_n_u8(0x03), m0c = vdupq_n_u8(0x0C), m30 = vdupq_n_u8(0x30);
    for (int x = 0; x < nc / 8; x++) {
        const block_tq2_0x8 *w = (const block_tq2_0x8 *) vx + (size_t) x * nb;
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            const int8_t *q8 = y[b].qs; const uint8_t *qs = w[b].qs;
            int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
            for (int h = 0; h < 2; h++) for (int g = 0; g < 2; g++) {
                const int8x16_t a0 = vld1q_s8(q8 + h * 128 + 16 * g), a1 = vld1q_s8(q8 + h * 128 + 32 + 16 * g),
                                a2 = vld1q_s8(q8 + h * 128 + 64 + 16 * g), a3 = vld1q_s8(q8 + h * 128 + 96 + 16 * g);
                const uint8_t *qc = qs + (h * 8 + g * 4) * 32;
#define TQ2_L(L) { const uint8x16_t w0 = vld1q_u8(qc + (L) * 32), w1 = vld1q_u8(qc + (L) * 32 + 16); \
                c00 = vdotq_laneq_s32(c00, vreinterpretq_s8_u8(vandq_u8(w0, m03)), a0, L); c01 = vdotq_laneq_s32(c01, vreinterpretq_s8_u8(vandq_u8(w1, m03)), a0, L); \
                c10 = vdotq_laneq_s32(c10, vreinterpretq_s8_u8(vandq_u8(w0, m0c)), a1, L); c11 = vdotq_laneq_s32(c11, vreinterpretq_s8_u8(vandq_u8(w1, m0c)), a1, L); \
                c20 = vdotq_laneq_s32(c20, vreinterpretq_s8_u8(vandq_u8(w0, m30)), a2, L); c21 = vdotq_laneq_s32(c21, vreinterpretq_s8_u8(vandq_u8(w1, m30)), a2, L); \
                c30 = vdotq_laneq_s32(c30, vreinterpretq_s8_u8(vshrq_n_u8(w0, 6)), a3, L); c31 = vdotq_laneq_s32(c31, vreinterpretq_s8_u8(vshrq_n_u8(w1, 6)), a3, L); }
                TQ2_L(0) TQ2_L(1) TQ2_L(2) TQ2_L(3)
#undef TQ2_L
            }
            const int32_t ysum = vaddlvq_s16(vaddq_s16(vld1q_s16(y[b].bsums), vld1q_s16(y[b].bsums + 8)));
            const int32x4_t ys = vdupq_n_s32(ysum);
            const int32x4_t t0 = vsubq_s32(vaddq_s32(vaddq_s32(c00, c30), vaddq_s32(vshrq_n_s32(c10, 2), vshrq_n_s32(c20, 4))), ys);
            const int32x4_t t1 = vsubq_s32(vaddq_s32(vaddq_s32(c01, c31), vaddq_s32(vshrq_n_s32(c11, 2), vshrq_n_s32(c21, 4))), ys);
            const float32x4_t yd = vdupq_n_f32(y[b].d);
            f0 = vfmaq_f32(f0, vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].d)), yd), vcvtq_f32_s32(t0));
            f1 = vfmaq_f32(f1, vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].d + 4)), yd), vcvtq_f32_s32(t1));
        }
        vst1q_f32(s + x * 8, f0); vst1q_f32(s + x * 8 + 4, f1);
    }
}

typedef void (*kfn)(int, float *, const void *, const void *, int);
typedef struct { kfn f; int n, nc; const uint8_t *w; const void *y; float *s; int reps; } job;
static void *run(void *a) { job *j = a; for (int r = 0; r < j->reps; r++) j->f(j->n, j->s, j->w, j->y, j->nc); return 0; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double bench(kfn f, int n, int nc, int nth, const uint8_t *w, size_t blk, const void *y, float *s, int reps) {
    pthread_t th[4]; job j[4]; int per = nc / nth; size_t rowbytes = (size_t) (n / QK_K) * blk; double best = 1e9;
    for (int rep = 0; rep < 3; rep++) { double t0 = now();
        for (int i = 0; i < nth; i++) { j[i] = (job) { f, n, per, w + (size_t) i * per * rowbytes, y, s + i * per, reps }; pthread_create(&th[i], 0, run, &j[i]); }
        for (int i = 0; i < nth; i++) pthread_join(th[i], 0);
        double t = now() - t0; if (t < best) best = t; }
    return (double) nc * n * reps / best / 1e9;
}

int main(void) {
    const int n = 2560, nb = n / QK_K;
    srand(3);
    block_q8_K *y = aligned_alloc(64, (sizeof(block_q8_K) * nb + 63) / 64 * 64);
    for (int b = 0; b < nb; b++) { y[b].d = 0.01f; for (int i = 0; i < QK_K; i++) y[b].qs[i] = rand() % 255 - 127;
        for (int i = 0; i < 16; i++) { int t = 0; for (int k = 0; k < 16; k++) t += y[b].qs[i * 16 + k]; y[b].bsums[i] = t; } }
    int ncs[2] = { 64, 29184 };   // 29184 rows = the Bonsai-4B FFN up+gate+down
    for (int k = 0; k < 2; k++) {
        int nc = ncs[k]; size_t nblk = (size_t) nc * nb; int reps = k == 0 ? 20000 : 10;
        block_tq1_0 *x1 = aligned_alloc(64, (nblk * sizeof(block_tq1_0) + 63) / 64 * 64);
        block_tq2_0 *x2 = aligned_alloc(64, (nblk * sizeof(block_tq2_0) + 63) / 64 * 64);
        for (size_t i = 0; i < nblk; i++) {
            for (int t = 0; t < 48; t++) { int v = rand() % 243; x1[i].qs[t] = (uint8_t) ((v * 256 + 242) / 243); }
            for (int t = 0; t < 4; t++) { int v = rand() % 81; x1[i].qh[t] = (uint8_t) ((v * 3 * 256 + 242) / 243); }
            for (int t = 0; t < 64; t++) { uint8_t q = 0; for (int l = 0; l < 4; l++) q |= (rand() % 3) << (2 * l); x2[i].qs[t] = q; }
            x1[i].d = x2[i].d = 0.001f * (1 + rand() % 9); }
        block_tq1_0x8 *r1 = aligned_alloc(64, (nblk * sizeof(block_tq1_0) + 63) / 64 * 64);
        block_tq2_0x8 *r2 = aligned_alloc(64, (nblk * sizeof(block_tq2_0) + 63) / 64 * 64);
        repack_tq1(x1, r1, nc, nb); repack_tq2(x2, r2, nc, nb);
        float *sa = calloc(nc, 4), *sb = calloc(nc, 4), *sc = calloc(nc, 4), *sd = calloc(nc, 4);
        gemv_tq1_plain(n, sa, x1, y, nc); gemv_tq1_r4(n, sb, r1, y, nc); gemv_tq2_plain(n, sc, x2, y, nc); gemv_tq2_r4(n, sd, r2, y, nc);
        double e[4] = { 0 };
        for (int r = 0; r < (nc < 256 ? nc : 256); r++) {
            double f1 = ref_tq1(n, x1 + (size_t) r * nb, y), f2 = ref_tq2(n, x2 + (size_t) r * nb, y);
            double v[4] = { (sa[r] - f1) / (1e-3 + (f1 < 0 ? -f1 : f1)), (sb[r] - f1) / (1e-3 + (f1 < 0 ? -f1 : f1)), (sc[r] - f2) / (1e-3 + (f2 < 0 ? -f2 : f2)), (sd[r] - f2) / (1e-3 + (f2 < 0 ? -f2 : f2)) };
            for (int i = 0; i < 4; i++) { double a = v[i] < 0 ? -v[i] : v[i]; if (a > e[i]) e[i] = a; } }
        printf("rows %5d  max rel err vs scalar: tq1 plain %.1e r4 %.1e | tq2 plain %.1e r4 %.1e\n", nc, e[0], e[1], e[2], e[3]);
        printf("   TQ1_0 plain 3T %5.2f  r4 3T %5.2f  r4 4T %5.2f G weights/s\n", bench(gemv_tq1_plain, n, nc, 3, (uint8_t *) x1, sizeof(block_tq1_0), y, sa, reps),
               bench(gemv_tq1_r4, n, nc, 3, (uint8_t *) r1, sizeof(block_tq1_0), y, sb, reps), bench(gemv_tq1_r4, n, nc, 4, (uint8_t *) r1, sizeof(block_tq1_0), y, sb, reps));
        printf("   TQ2_0 plain 3T %5.2f  r4 3T %5.2f  r4 4T %5.2f G weights/s\n", bench(gemv_tq2_plain, n, nc, 3, (uint8_t *) x2, sizeof(block_tq2_0), y, sc, reps),
               bench(gemv_tq2_r4, n, nc, 3, (uint8_t *) r2, sizeof(block_tq2_0), y, sd, reps), bench(gemv_tq2_r4, n, nc, 4, (uint8_t *) r2, sizeof(block_tq2_0), y, sd, reps));
        free(x1); free(x2); free(r1); free(r2); free(sa); free(sb); free(sc); free(sd);
    }
    return 0;
}
