// kbench_q2k: Q2_K x Q8_K mat-vec on the Cortex-A55 (dotprod, no i8mm).
// "plain" = llama.cpp v0.8.0's NEON ggml_vec_dot_q2_K_q8_K, one row at a time (what the TV runs today:
//           llama.cpp has no repacked Q2_K kernel for ARM, only x86 AVX-512 and RISC-V).
// "r4"    = new: 8 rows interleaved 4 bytes at a time (block_q2_Kx8r4) so one 16-byte load feeds 4 rows,
//           vdotq_laneq against the activation, no horizontal adds; int16 narrowing for the scale multiply.
// Checks both against a scalar reference, then measures weights/s for a cache-resident and a DRAM-sized matrix.
// usage: kbench_q2k
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define QK_K 256
typedef struct { uint8_t scales[16]; uint8_t qs[64]; __fp16 d; __fp16 dmin; } block_q2_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } block_q8_K;
// 8 rows; scales[sb*8 + j] = row j's scales[sb]; qs[c*32 + j*4 + i] = row j's qs[c*4 + i], c = 0..15
typedef struct { __fp16 d[8]; __fp16 dmin[8]; uint8_t scales[128]; uint8_t qs[512]; } block_q2_Kx8r4;

static void repack(const block_q2_K *in, block_q2_Kx8r4 *out, int nrows, int nb) {
    for (int x = 0; x < nrows / 8; x++) for (int b = 0; b < nb; b++) {
        block_q2_Kx8r4 *o = &out[x * nb + b];
        for (int j = 0; j < 8; j++) {
            const block_q2_K *r = &in[(x * 8 + j) * nb + b];
            o->d[j] = r->d; o->dmin[j] = r->dmin;
            for (int sb = 0; sb < 16; sb++) o->scales[sb * 8 + j] = r->scales[sb];
            for (int c = 0; c < 16; c++) memcpy(&o->qs[c * 32 + j * 4], &r->qs[c * 4], 4);
        }
    }
}

static double ref_dot(int n, const block_q2_K *x, const block_q8_K *y) {
    double s = 0;
    for (int b = 0; b < n / QK_K; b++) {
        double d = (double) x[b].d * y[b].d, dm = (double) x[b].dmin * y[b].d;
        for (int e = 0; e < QK_K; e++) {
            int h = e / 128, r = e % 128, sh = r / 32, l = r % 32, sb = e / 16;
            int q = (x[b].qs[h * 32 + l] >> (2 * sh)) & 3;
            s += (d * (x[b].scales[sb] & 15) * q - dm * (x[b].scales[sb] >> 4)) * y[b].qs[e];
        }
    }
    return s;
}

// ---- plain: llama.cpp v0.8.0 NEON path, logic unchanged ----
static void dot_plain(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    const uint8x16_t m3 = vdupq_n_u8(0x3), m4 = vdupq_n_u8(0xF);
    const int32x4_t vzero = vdupq_n_s32(0);
    int8x16x2_t q2bytes; uint8_t aux[16]; float sum = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * (float) x[i].d, dmin = -y[i].d * (float) x[i].dmin;
        const uint8_t *q2 = x[i].qs; const int8_t *q8 = y[i].qs; const uint8_t *sc = x[i].scales;
        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        vst1q_u8(aux, vandq_u8(mins_and_scales, m4));
        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8_t mn0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), mn1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16(mn0), vget_low_s16(q8sums.val[0])), vmull_s16(vget_high_s16(mn0), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16(mn1), vget_low_s16(q8sums.val[1])), vmull_s16(vget_high_s16(mn1), vget_high_s16(q8sums.val[1])));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));
        int isum = 0, is = 0;
#define MAC(index) \
        isum += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is + (index)]; \
        isum += vaddvq_s32(vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is + 1 + (index)];
#define SMAC(shift, index) \
        q8bytes = vld1q_s8_x2(q8); q8 += 32; \
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3)); \
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3)); \
        MAC((index));
        for (int j = 0; j < QK_K / 128; ++j) {
            const uint8x16x2_t q2bits = vld1q_u8_x2(q2); q2 += 32;
            int8x16x2_t q8bytes = vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));
            MAC(0); SMAC(2, 2); SMAC(4, 4); SMAC(6, 6);
            is += 8;
        }
        sum += d * isum;
    }
    *s = sum;
}
static void gemv_plain(int n, float *s, const void *vx, const void *vy, int nc) {
    const int nb = n / QK_K;
    for (int r = 0; r < nc; r++) dot_plain(n, s + r, (const block_q2_K *) vx + (size_t) r * nb, vy);
}

// ---- r4: new repacked kernel ----
static int g_pfd = 0;
static void gemv_r4(int n, float *s, const void *vx, const void *vy, int nc) {
    const int nb = n / QK_K;
    const block_q8_K *y = vy;
    const uint8x16_t m03 = vdupq_n_u8(0x03), m0c = vdupq_n_u8(0x0C), m30 = vdupq_n_u8(0x30);
    const uint8x8_t m4 = vdup_n_u8(0x0F);
    for (int x = 0; x < nc / 8; x++) {
        const block_q2_Kx8r4 *w = (const block_q2_Kx8r4 *) vx + (size_t) x * nb;
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            const int8_t *q8 = y[b].qs; const uint8_t *qs = w[b].qs, *sc = w[b].scales;
            if (g_pfd) { const char *p = (const char *) &w[b] + g_pfd; for (int o = 0; o < 672; o += 64) __builtin_prefetch(p + o); }
            int32x4_t t0 = vdupq_n_s32(0), t1 = vdupq_n_s32(0);
            for (int h = 0; h < 2; h++) for (int g = 0; g < 2; g++) {
                const int8x16_t a0 = vld1q_s8(q8 + h * 128 + 16 * g), a1 = vld1q_s8(q8 + h * 128 + 32 + 16 * g),
                                a2 = vld1q_s8(q8 + h * 128 + 64 + 16 * g), a3 = vld1q_s8(q8 + h * 128 + 96 + 16 * g);
                int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
                const uint8_t *qc = qs + (h * 8 + g * 4) * 32;
#define LANE(L) { \
                const uint8x16_t w0 = vld1q_u8(qc + (L) * 32), w1 = vld1q_u8(qc + (L) * 32 + 16); \
                c00 = vdotq_laneq_s32(c00, vreinterpretq_s8_u8(vandq_u8(w0, m03)), a0, L); \
                c01 = vdotq_laneq_s32(c01, vreinterpretq_s8_u8(vandq_u8(w1, m03)), a0, L); \
                c10 = vdotq_laneq_s32(c10, vreinterpretq_s8_u8(vandq_u8(w0, m0c)), a1, L); \
                c11 = vdotq_laneq_s32(c11, vreinterpretq_s8_u8(vandq_u8(w1, m0c)), a1, L); \
                c20 = vdotq_laneq_s32(c20, vreinterpretq_s8_u8(vandq_u8(w0, m30)), a2, L); \
                c21 = vdotq_laneq_s32(c21, vreinterpretq_s8_u8(vandq_u8(w1, m30)), a2, L); \
                c30 = vdotq_laneq_s32(c30, vreinterpretq_s8_u8(vshrq_n_u8(w0, 6)), a3, L); \
                c31 = vdotq_laneq_s32(c31, vreinterpretq_s8_u8(vshrq_n_u8(w1, 6)), a3, L); }
                LANE(0) LANE(1) LANE(2) LANE(3)
#undef LANE
                // sub-block of shift s = 8h + 2s + g; per-row dots fit in int16 (|dot| <= 16*3*128)
                const int base = 8 * h + g;
#define SCALE(S, D0, D1) { \
                const int16x8_t sc16 = vreinterpretq_s16_u16(vmovl_u8(vand_u8(vld1_u8(sc + (base + 2 * (S)) * 8), m4))); \
                const int16x8_t dd = vcombine_s16(D0, D1); \
                t0 = vmlal_s16(t0, vget_low_s16(dd), vget_low_s16(sc16)); t1 = vmlal_high_s16(t1, dd, sc16); }
                SCALE(0, vmovn_s32(c00), vmovn_s32(c01))
                SCALE(1, vshrn_n_s32(c10, 2), vshrn_n_s32(c11, 2))
                SCALE(2, vshrn_n_s32(c20, 4), vshrn_n_s32(c21, 4))
                SCALE(3, vmovn_s32(c30), vmovn_s32(c31))
#undef SCALE
            }
            // mins: bias[j] = sum_sb min[sb][j] * bsums[sb]
            int32x4_t b0 = vdupq_n_s32(0), b1 = vdupq_n_s32(0);
            const int16x8_t bs0 = vld1q_s16(y[b].bsums), bs1 = vld1q_s16(y[b].bsums + 8);
#define MINS(P, BS, L0, L1) { \
            const uint8x16_t raw = vld1q_u8(sc + (P) * 16); \
            const uint8x16_t mn = vshrq_n_u8(raw, 4); \
            const int16x8_t m0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mn))), m1 = vreinterpretq_s16_u16(vmovl_high_u8(mn)); \
            b0 = vmlal_laneq_s16(b0, vget_low_s16(m0), BS, L0); b1 = vmlal_high_laneq_s16(b1, m0, BS, L0); \
            b0 = vmlal_laneq_s16(b0, vget_low_s16(m1), BS, L1); b1 = vmlal_high_laneq_s16(b1, m1, BS, L1); }
            MINS(0, bs0, 0, 1) MINS(1, bs0, 2, 3) MINS(2, bs0, 4, 5) MINS(3, bs0, 6, 7)
            MINS(4, bs1, 0, 1) MINS(5, bs1, 2, 3) MINS(6, bs1, 4, 5) MINS(7, bs1, 6, 7)
#undef MINS
            const float32x4_t yd = vdupq_n_f32(y[b].d);
            const float32x4_t d0 = vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].d)), yd), d1 = vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].d + 4)), yd);
            const float32x4_t m0 = vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].dmin)), yd), m1 = vmulq_f32(vcvt_f32_f16(vld1_f16(w[b].dmin + 4)), yd);
            f0 = vfmaq_f32(f0, d0, vcvtq_f32_s32(t0)); f1 = vfmaq_f32(f1, d1, vcvtq_f32_s32(t1));
            f0 = vfmsq_f32(f0, m0, vcvtq_f32_s32(b0)); f1 = vfmsq_f32(f1, m1, vcvtq_f32_s32(b1));
        }
        vst1q_f32(s + x * 8, f0); vst1q_f32(s + x * 8 + 4, f1);
    }
}

typedef void (*kfn)(int, float *, const void *, const void *, int);
typedef struct { kfn f; int n, nc; const uint8_t *w; const void *y; float *s; int reps; } job;
static void *run(void *a) { job *j = a; for (int r = 0; r < j->reps; r++) j->f(j->n, j->s, j->w, j->y, j->nc); return 0; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
// returns G weights/s
static double bench(kfn f, int n, int nc, int nth, const uint8_t *w, const void *y, float *s, int reps) {
    pthread_t th[4]; job j[4]; int per = nc / nth; size_t rowbytes = (size_t) (n / QK_K) * sizeof(block_q2_K);
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        for (int i = 0; i < nth; i++) {
            j[i] = (job) { f, n, per, w + (size_t) i * per * rowbytes, y, s + i * per, reps };
            pthread_create(&th[i], 0, run, &j[i]);
        }
        for (int i = 0; i < nth; i++) pthread_join(th[i], 0);
        double t = now() - t0; if (t < best) best = t;
    }
    return (double) nc * n * reps / best / 1e9;
}

int main(void) {
    const int n = 2560, nb = n / QK_K;
    srand(1);
    block_q8_K *y = aligned_alloc(64, (sizeof(block_q8_K) * nb + 63) / 64 * 64);
    for (int b = 0; b < nb; b++) { y[b].d = 0.01f; for (int i = 0; i < QK_K; i++) y[b].qs[i] = rand() % 255 - 127;
        for (int i = 0; i < 16; i++) { int t = 0; for (int k = 0; k < 16; k++) t += y[b].qs[i * 16 + k]; y[b].bsums[i] = t; } }
    int ncs[2] = { 48, 27648 };   // 48 rows = 40 KB (cache-resident); 27648 rows = 23 MB (= the 4B's FFN up+gate+down)
    for (int k = 0; k < 2; k++) {
        int nc = ncs[k];
        size_t nblk = (size_t) nc * nb, wb = nblk * sizeof(block_q2_K);
        block_q2_K *x = aligned_alloc(64, (wb + 63) / 64 * 64);
        for (size_t i = 0; i < nblk; i++) { for (int t = 0; t < 16; t++) x[i].scales[t] = rand(); for (int t = 0; t < 64; t++) x[i].qs[t] = rand();
            x[i].d = 0.001f * (1 + rand() % 7); x[i].dmin = 0.0005f * (1 + rand() % 5); }
        block_q2_Kx8r4 *xr = aligned_alloc(64, (wb + 63) / 64 * 64);
        repack(x, xr, nc, nb);
        float *s1 = calloc(nc, 4), *s2 = calloc(nc, 4);
        gemv_plain(n, s1, x, y, nc); gemv_r4(n, s2, xr, y, nc);
        double e1 = 0, e2 = 0;
        for (int r = 0; r < (nc < 512 ? nc : 512); r++) { double rf = ref_dot(n, x + (size_t) r * nb, y), sc = 1e-3 + (rf < 0 ? -rf : rf);
            double a = (s1[r] - rf) / sc, c = (s2[r] - rf) / sc; if (a < 0) a = -a; if (c < 0) c = -c; if (a > e1) e1 = a; if (c > e2) e2 = c; }
        int reps = k == 0 ? 20000 : 10;
        printf("rows %5d (%6.0f KB)  max rel err vs scalar: plain %.1e  r4 %.1e\n", nc, wb / 1024.0, e1, e2);
        double p1 = bench(gemv_plain, n, nc, 1, (const uint8_t *) x, y, s1, reps), p3 = bench(gemv_plain, n, nc, 3, (const uint8_t *) x, y, s1, reps);
        double r1 = bench(gemv_r4, n, nc, 1, (const uint8_t *) xr, y, s2, reps), r3 = bench(gemv_r4, n, nc, 3, (const uint8_t *) xr, y, s2, reps);
        double gbs = sizeof(block_q2_K) / 256.0;
        printf("   plain 1T %5.2f  3T %5.2f G weights/s (3T %4.2f GB/s)   r4 1T %5.2f  3T %5.2f G weights/s (3T %4.2f GB/s)\n",
               p1, p3, p3 * gbs, r1, r3, r3 * gbs);
        if (k == 1) { int pf[] = { 672, 1344, 2688, 5376 }; for (int i = 0; i < 4; i++) { g_pfd = pf[i];
            printf("   r4 prefetch %4d B: 3T %5.2f  4T %5.2f G weights/s\n", pf[i], bench(gemv_r4, n, nc, 3, (const uint8_t *) xr, y, s2, reps), bench(gemv_r4, n, nc, 4, (const uint8_t *) xr, y, s2, reps)); }
            g_pfd = 0; printf("   r4 no prefetch 4T %5.2f   plain 4T %5.2f\n", bench(gemv_r4, n, nc, 4, (const uint8_t *) xr, y, s2, reps), bench(gemv_plain, n, nc, 4, (const uint8_t *) x, y, s1, reps)); }
        free(x); free(xr); free(s1); free(s2);
    }
    return 0;
}
