// kbench: micro-benchmark of the Q4_K x Q8_K mat-vec kernel used for token generation on this box
// (llama.cpp's ggml_gemv_q4_K_8x4_q8_K: weights repacked 8 rows at a time, ARM dotprod).
// "ref" is llama.cpp's kernel (copied, logic unchanged); "v2" is a variant tuned for the in-order Cortex-A55.
// Measures GB/s of weight bytes, 1 and 3 threads, for a cache-resident matrix and a DRAM-sized one,
// and checks v2 against ref.   usage: kbench
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define QK_K 256
typedef struct { __fp16 d[8]; __fp16 dmin[8]; uint8_t scales[96]; uint8_t qs[1024]; } block_q4_Kx8;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } block_q8_K;

static inline void decode_scales(const uint8_t *in, int16x8_t *out_mins, int8_t *out_scales) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t sm[3]; memcpy(sm, in, 12);
    const uint32_t mins_0_3 = sm[1] & kmask1;
    const uint32_t mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
    const uint32x2_t mins_u32 = { mins_0_3, mins_4_7 };
    *out_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins_u32)));
    uint32_t sc[2]; sc[0] = sm[0] & kmask1; sc[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    memcpy(out_scales, sc, 8);
}

// ---- reference (llama.cpp v0.8.0 ggml_gemv_q4_K_8x4_q8_K, NEON dotprod path) ----
static int g_pfd = 0;   // software prefetch distance in bytes (0 = off), for the variant sweep
static void gemv_ref(int n, float *s, const void *vx, const void *vy, int nc) {
    const int nb = n / QK_K; const uint8x16_t m4b = vdupq_n_u8(0x0f); const int pfd = g_pfd;
    float32x4_t acc_f32[2];
    const block_q8_K *q8_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_q4_Kx8 *q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);
        acc_f32[0] = acc_f32[1] = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            float32x4_t q4_d_0 = vcvt_f32_f16(vld1_f16(q4_ptr[b].d)), q4_d_1 = vcvt_f32_f16(vld1_f16(q4_ptr[b].d + 4));
            float32x4_t q8_d = vdupq_n_f32(q8_ptr[b].d);
            float32x4_t sb_scale_0123 = vmulq_f32(q4_d_0, q8_d), sb_scale_4567 = vmulq_f32(q4_d_1, q8_d);
            float32x4_t sb_min_0123 = vmulq_f32(vcvt_f32_f16(vld1_f16(q4_ptr[b].dmin)), q8_d);
            float32x4_t sb_min_4567 = vmulq_f32(vcvt_f32_f16(vld1_f16(q4_ptr[b].dmin + 4)), q8_d);
            int32x4_t bias_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
            int32x4_t acc_lo[2], acc_hi[2];
            const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums), vld1q_s16(q8_ptr[b].bsums + 8));
            int16_t bsums_arr[8]; vst1q_s16(bsums_arr, bsums);
            for (int sb = 0; sb < QK_K / 64; sb++) {
                if (pfd) { const char *p = (const char *) &q4_ptr[b] + sb * 288 + pfd;
                           __builtin_prefetch(p); __builtin_prefetch(p + 64); __builtin_prefetch(p + 128); __builtin_prefetch(p + 192); __builtin_prefetch(p + 256); }
                acc_lo[0] = acc_lo[1] = acc_hi[0] = acc_hi[1] = vdupq_n_s32(0);
                int16x8_t q4sb_mins[2], q4sb_scales[2];
                for (int i = 0; i < 2; i++) { int8_t aux[8]; decode_scales(&q4_ptr[b].scales[sb * 24 + i * 12], &q4sb_mins[i], aux); q4sb_scales[i] = vmovl_s8(vld1_s8(aux)); }
                int8x16_t q8_qs[4];
                for (int i = 0; i < 4; i++) q8_qs[i] = vld1q_s8(q8_ptr[b].qs + sb * 64 + i * 16);
                for (int c = 0; c < 2; c++) {
                    uint8x16_t q4_cols[8];
                    for (int i = 0; i < 8; i++) q4_cols[i] = vld1q_u8(q4_ptr[b].qs + sb * QK_K + i * 32 + 16 * c);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b)), q8_qs[0], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b)), q8_qs[0], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b)), q8_qs[0], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b)), q8_qs[0], 3);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b)), q8_qs[1], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b)), q8_qs[1], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b)), q8_qs[1], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b)), q8_qs[1], 3);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4)), q8_qs[2], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4)), q8_qs[2], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4)), q8_qs[2], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4)), q8_qs[2], 3);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4)), q8_qs[3], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4)), q8_qs[3], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4)), q8_qs[3], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4)), q8_qs[3], 3);
                }
                const float32x4_t sumf_0123 = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_low_s16(q4sb_scales[0])), acc_lo[0]), vmulq_s32(vmovl_s16(vget_low_s16(q4sb_scales[1])), acc_hi[0])));
                acc_f32[0] = vfmaq_f32(acc_f32[0], sb_scale_0123, sumf_0123);
                const float32x4_t sumf_4567 = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(vget_high_s16(q4sb_scales[0])), acc_lo[1]), vmulq_s32(vmovl_s16(vget_high_s16(q4sb_scales[1])), acc_hi[1])));
                acc_f32[1] = vfmaq_f32(acc_f32[1], sb_scale_4567, sumf_4567);
                const int16x4_t blo = vdup_n_s16(bsums_arr[2 * sb]), bhi = vdup_n_s16(bsums_arr[2 * sb + 1]);
                bias_acc[0] = vmlal_s16(bias_acc[0], blo, vget_low_s16(q4sb_mins[0]));
                bias_acc[0] = vmlal_s16(bias_acc[0], bhi, vget_low_s16(q4sb_mins[1]));
                bias_acc[1] = vmlal_s16(bias_acc[1], blo, vget_high_s16(q4sb_mins[0]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bhi, vget_high_s16(q4sb_mins[1]));
            }
            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_4567);
        }
        vst1q_f32(s + x * 8, acc_f32[0]); vst1q_f32(s + x * 8 + 4, acc_f32[1]);
    }
}

// ---- v2: same math, scheduled for an in-order core ----
// * all 16 loads of a sub-block issued first, masks/shifts next, then the 32 dot products spread over
//   8 independent accumulators (no back-to-back dependent SDOTs)
// * software prefetch of the next super-block (1152 B) ahead
// * the mins/bias correction is folded per super-block with one widening multiply chain
static void gemv_v2(int n, float *s, const void *vx, const void *vy, int nc) {
    const int nb = n / QK_K; const uint8x16_t m4b = vdupq_n_u8(0x0f);
    const block_q8_K *q8_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / 8; x++) {
        const block_q4_Kx8 *q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);
        float32x4_t f0 = vdupq_n_f32(0), f1 = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            const float32x4_t q8_d = vdupq_n_f32(q8_ptr[b].d);
            const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums), vld1q_s16(q8_ptr[b].bsums + 8));
            int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0), bias0 = vdupq_n_s32(0), bias1 = vdupq_n_s32(0);
            const uint8_t *qs = q4_ptr[b].qs;
            for (int sb = 0; sb < 4; sb++) {
                if (g_pfd) { const char *pp = (const char *) &q4_ptr[b] + sb * 288 + g_pfd;
                             __builtin_prefetch(pp); __builtin_prefetch(pp + 64); __builtin_prefetch(pp + 128); __builtin_prefetch(pp + 192); __builtin_prefetch(pp + 256); }
                int16x8_t mn0, mn1; int8_t a0[8], a1[8];
                decode_scales(&q4_ptr[b].scales[sb * 24], &mn0, a0);
                decode_scales(&q4_ptr[b].scales[sb * 24 + 12], &mn1, a1);
                const int8x16_t y0 = vld1q_s8(q8_ptr[b].qs + sb * 64), y1 = vld1q_s8(q8_ptr[b].qs + sb * 64 + 16);
                const int8x16_t y2 = vld1q_s8(q8_ptr[b].qs + sb * 64 + 32), y3 = vld1q_s8(q8_ptr[b].qs + sb * 64 + 48);
                const uint8_t *p = qs + sb * QK_K;
                uint8x16_t w[16];
                for (int i = 0; i < 16; i++) w[i] = vld1q_u8(p + i * 16);   // w[2i] = cols 0-3, w[2i+1] = cols 4-7
                int32x4_t l0a = vdupq_n_s32(0), l0b = l0a, l1a = l0a, l1b = l0a, h0a = l0a, h0b = l0a, h1a = l0a, h1b = l0a;
#define LO(v) vreinterpretq_s8_u8(vandq_u8(v, m4b))
#define HI(v) vreinterpretq_s8_u8(vshrq_n_u8(v, 4))
                l0a = vdotq_laneq_s32(l0a, LO(w[0]), y0, 0);  l1a = vdotq_laneq_s32(l1a, LO(w[1]), y0, 0);
                h0a = vdotq_laneq_s32(h0a, HI(w[0]), y2, 0);  h1a = vdotq_laneq_s32(h1a, HI(w[1]), y2, 0);
                l0b = vdotq_laneq_s32(l0b, LO(w[2]), y0, 1);  l1b = vdotq_laneq_s32(l1b, LO(w[3]), y0, 1);
                h0b = vdotq_laneq_s32(h0b, HI(w[2]), y2, 1);  h1b = vdotq_laneq_s32(h1b, HI(w[3]), y2, 1);
                l0a = vdotq_laneq_s32(l0a, LO(w[4]), y0, 2);  l1a = vdotq_laneq_s32(l1a, LO(w[5]), y0, 2);
                h0a = vdotq_laneq_s32(h0a, HI(w[4]), y2, 2);  h1a = vdotq_laneq_s32(h1a, HI(w[5]), y2, 2);
                l0b = vdotq_laneq_s32(l0b, LO(w[6]), y0, 3);  l1b = vdotq_laneq_s32(l1b, LO(w[7]), y0, 3);
                h0b = vdotq_laneq_s32(h0b, HI(w[6]), y2, 3);  h1b = vdotq_laneq_s32(h1b, HI(w[7]), y2, 3);
                l0a = vdotq_laneq_s32(l0a, LO(w[8]), y1, 0);  l1a = vdotq_laneq_s32(l1a, LO(w[9]), y1, 0);
                h0a = vdotq_laneq_s32(h0a, HI(w[8]), y3, 0);  h1a = vdotq_laneq_s32(h1a, HI(w[9]), y3, 0);
                l0b = vdotq_laneq_s32(l0b, LO(w[10]), y1, 1); l1b = vdotq_laneq_s32(l1b, LO(w[11]), y1, 1);
                h0b = vdotq_laneq_s32(h0b, HI(w[10]), y3, 1); h1b = vdotq_laneq_s32(h1b, HI(w[11]), y3, 1);
                l0a = vdotq_laneq_s32(l0a, LO(w[12]), y1, 2); l1a = vdotq_laneq_s32(l1a, LO(w[13]), y1, 2);
                h0a = vdotq_laneq_s32(h0a, HI(w[12]), y3, 2); h1a = vdotq_laneq_s32(h1a, HI(w[13]), y3, 2);
                l0b = vdotq_laneq_s32(l0b, LO(w[14]), y1, 3); l1b = vdotq_laneq_s32(l1b, LO(w[15]), y1, 3);
                h0b = vdotq_laneq_s32(h0b, HI(w[14]), y3, 3); h1b = vdotq_laneq_s32(h1b, HI(w[15]), y3, 3);
                const int16x8_t sc0 = vmovl_s8(vld1_s8(a0)), sc1 = vmovl_s8(vld1_s8(a1));
                sum0 = vmlaq_s32(sum0, vmovl_s16(vget_low_s16(sc0)),  vaddq_s32(l0a, l0b));
                sum0 = vmlaq_s32(sum0, vmovl_s16(vget_low_s16(sc1)),  vaddq_s32(h0a, h0b));
                sum1 = vmlaq_s32(sum1, vmovl_s16(vget_high_s16(sc0)), vaddq_s32(l1a, l1b));
                sum1 = vmlaq_s32(sum1, vmovl_s16(vget_high_s16(sc1)), vaddq_s32(h1a, h1b));
                const int16x4_t blo = vdup_laneq_s16(bsums, 0), bhi = vdup_laneq_s16(bsums, 1);
                (void) blo; (void) bhi;
                const int16_t bl = vgetq_lane_s16(bsums, 0); (void) bl;
                int16_t ba[8]; vst1q_s16(ba, bsums);
                bias0 = vmlal_n_s16(bias0, vget_low_s16(mn0),  ba[2 * sb]); bias0 = vmlal_n_s16(bias0, vget_low_s16(mn1),  ba[2 * sb + 1]);
                bias1 = vmlal_n_s16(bias1, vget_high_s16(mn0), ba[2 * sb]); bias1 = vmlal_n_s16(bias1, vget_high_s16(mn1), ba[2 * sb + 1]);
            }
            const float32x4_t d0 = vmulq_f32(vcvt_f32_f16(vld1_f16(q4_ptr[b].d)), q8_d), d1 = vmulq_f32(vcvt_f32_f16(vld1_f16(q4_ptr[b].d + 4)), q8_d);
            const float32x4_t m0 = vmulq_f32(vcvt_f32_f16(vld1_f16(q4_ptr[b].dmin)), q8_d), m1 = vmulq_f32(vcvt_f32_f16(vld1_f16(q4_ptr[b].dmin + 4)), q8_d);
            f0 = vfmaq_f32(f0, d0, vcvtq_f32_s32(sum0)); f1 = vfmaq_f32(f1, d1, vcvtq_f32_s32(sum1));
            f0 = vmlsq_f32(f0, vcvtq_f32_s32(bias0), m0); f1 = vmlsq_f32(f1, vcvtq_f32_s32(bias1), m1);
        }
        vst1q_f32(s + x * 8, f0); vst1q_f32(s + x * 8 + 4, f1);
    }
}

typedef void (*kfn)(int, float *, const void *, const void *, int);
typedef struct { kfn f; int n, nc; const uint8_t *w; const void *y; float *s; int reps; } job;
static void *run(void *a) { job *j = a; for (int r = 0; r < j->reps; r++) j->f(j->n, j->s, j->w, j->y, j->nc); return 0; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static double bench(kfn f, int n, int nc, int nth, const uint8_t *w, const void *y, float *s, int reps) {
    pthread_t th[4]; job j[4]; int per = nc / nth;
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        for (int i = 0; i < nth; i++) {
            j[i] = (job) { f, n, per, w + (size_t) i * per / 8 * (n / QK_K) * sizeof(block_q4_Kx8), y, s + i * per, reps };
            pthread_create(&th[i], 0, run, &j[i]);
        }
        for (int i = 0; i < nth; i++) pthread_join(th[i], 0);
        double t = now() - t0; if (t < best) best = t;
    }
    return (double) nc / 8 * (n / QK_K) * sizeof(block_q4_Kx8) * reps / best / 1e9;
}

int main(void) {
    const int n = 2048;
    const int nb = n / QK_K;
    srand(1);
    block_q8_K *y = aligned_alloc(64, (sizeof(block_q8_K) * nb + 63) / 64 * 64);
    for (int b = 0; b < nb; b++) { y[b].d = 0.01f; for (int i = 0; i < QK_K; i++) y[b].qs[i] = rand() % 255 - 127;
        for (int i = 0; i < 16; i++) { int t = 0; for (int k = 0; k < 16; k++) t += y[b].qs[i * 16 + k]; y[b].bsums[i] = t; } }
    int ncs[2] = { 48, 24576 };             // 48 rows = 54 KB (cache-resident), 24576 rows = 27 MB (DRAM, like an FFN matrix x4)
    for (int k = 0; k < 2; k++) {
        int nc = ncs[k];
        size_t wb = (size_t) nc / 8 * nb * sizeof(block_q4_Kx8);
        uint8_t *w = aligned_alloc(64, wb + 65536);
        for (size_t i = 0; i < wb + 65536; i++) w[i] = rand();
        block_q4_Kx8 *bw = (block_q4_Kx8 *) w;
        for (size_t i = 0; i < wb / sizeof(block_q4_Kx8); i++) for (int c = 0; c < 8; c++) { bw[i].d[c] = 0.01f; bw[i].dmin[c] = 0.002f; }
        float *s1 = calloc(nc, 4), *s2 = calloc(nc, 4);
        gemv_ref(n, s1, w, y, nc); gemv_v2(n, s2, w, y, nc);
        double maxd = 0; for (int i = 0; i < nc; i++) { double d = s1[i] - s2[i]; if (d < 0) d = -d; double r = d / (1e-6 + (s1[i] < 0 ? -s1[i] : s1[i])); if (r > maxd) maxd = r; }
        int reps = k == 0 ? 20000 : 8;
        printf("rows %5d (%7.0f KB)  max rel diff v2/ref %.2e\n", nc, wb / 1024.0, maxd);
        int pfds[] = { 0, 512, 1024, 2048, 4096, 8192 };
        for (int i = 0; i < 6; i++) {
            g_pfd = pfds[i];
            double r1 = bench(gemv_ref, n, nc, 1, w, y, s1, reps), r3 = bench(gemv_ref, n, nc, 3, w, y, s1, reps);
            double v1 = bench(gemv_v2,  n, nc, 1, w, y, s2, reps), v3 = bench(gemv_v2,  n, nc, 3, w, y, s2, reps);
            printf("   prefetch %5d B   ref 1T %5.2f  3T %5.2f GB/s   v2 1T %5.2f  3T %5.2f GB/s\n", pfds[i], r1, r3, v1, v3);
        }
        free(w); free(s1); free(s2);
    }
    return 0;
}
