// q1_lut: Four-Russians / T-MAC style Q1_0 kernel vs the bit-plane kernel (repack-q1-tv.inc), kernel only.
// Layout under test: groups of 32 rows; one 16-byte vector holds 32 weight nibbles (4 columns): low nibble rows 0-15, high
// nibble rows 16-31. One 16-entry table per 4 activations. T[n] = round(sum(+-x)/4) + 128, antisymmetric rounding.
// Prints error vs the exact dot and G weights/s, cache-resident and DRAM-sized, 1-4 threads. PRFM distance via argv[1].
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <algorithm>
#define QK_K 256
#define GGML_RESTRICT __restrict__
#define GGML_UNUSED(x) (void)(x)
#define GGML_ASSERT(x) do { if (!(x)) { printf("assert %s\n", #x); abort(); } } while (0)
#define GGML_ABORT(...) abort()
typedef uint16_t ggml_half;
enum { GGML_TYPE_Q1_0 = 41 };
struct ggml_tensor { int type; int64_t ne[4]; void * data; };
static int64_t ggml_nrows(const ggml_tensor * t) { return t->ne[1] * t->ne[2] * t->ne[3]; }
struct block_q1_0 { ggml_half d; uint8_t qs[16]; };
static inline float GGML_CPU_FP16_TO_FP32(ggml_half v) { __fp16 x; memcpy(&x, &v, 2); return (float) x; }
static inline ggml_half GGML_CPU_FP32_TO_FP16(float f) { __fp16 x = (__fp16) f; ggml_half r; memcpy(&r, &x, 2); return r; }
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; };
struct block_q8_Kx4 { float d[4]; int8_t qs[QK_K * 4]; int16_t bsums[QK_K / 4]; };
#include "../../third_party/llama.cpp/ggml/src/ggml-cpu/repack-q1-tv.inc"
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float hf(ggml_half v) { __fp16 x; memcpy(&x, &v, 2); return (float) x; }
static ggml_half h(float f) { __fp16 x = (__fp16) f; ggml_half r; memcpy(&r, &x, 2); return r; }
static void quant_row(const float * x, block_q8_K * y, int k) {
    for (int i = 0; i < k / QK_K; i++) {
        float amax = 0, mx = 0;
        for (int j = 0; j < QK_K; j++) { float a = fabsf(x[i * QK_K + j]); if (a > amax) { amax = a; mx = x[i * QK_K + j]; } }
        const float is = amax ? -127.f / mx : 0;
        for (int j = 0; j < QK_K; j++) { int v = (int) nearbyintf(is * x[i * QK_K + j]); y[i].qs[j] = v > 127 ? 127 : v; }
        for (int j = 0; j < 16; j++) { int s = 0; for (int t = 0; t < 16; t++) s += y[i].qs[j * 16 + t]; y[i].bsums[j] = s; }
        y[i].d = amax ? 1 / is : 0;
    }
}
// ---- LUT layout
static inline size_t lut_group_bytes(int n) { return 128 + (size_t)(n / 128) * (32 + 512); }   // dmin[32], dstep[32] fp16; per 128 cols: 32 codes + 32 vectors
static void lut_repack(uint8_t * dst, const block_q1_0 * src, int n, int nrow) {
    const int nb128 = n / 128;
    for (int r = 0; r < nrow; r += 32) {
        ggml_half * hdr = (ggml_half *) dst; float mn[32], st[32];
        for (int j = 0; j < 32; j++) {
            float lo = INFINITY, hi = -INFINITY;
            for (int b = 0; b < nb128; b++) { float d = hf(src[(size_t)(r + j) * nb128 + b].d); lo = std::min(lo, d); hi = std::max(hi, d); }
            hdr[j] = h(lo); hdr[32 + j] = h((hi - lo) / 255.f); mn[j] = hf(hdr[j]); st[j] = hf(hdr[32 + j]);
        }
        uint8_t * o = dst + 128;
        for (int b = 0; b < nb128; b++, o += 544) {
            memset(o, 0, 544);
            for (int j = 0; j < 32; j++) {
                const block_q1_0 * in = &src[(size_t)(r + j) * nb128 + b];
                int q = st[j] > 0 ? (int) lroundf((hf(in->d) - mn[j]) / st[j]) : 0; o[j] = (uint8_t) std::max(0, std::min(255, q));
                for (int m = 0; m < 128; m++) if ((in->qs[m / 8] >> (m % 8)) & 1) {
                    const int tile = m / 4, k = m % 4;
                    o[32 + tile * 16 + (j & 15)] |= (uint8_t)(1u << (k + (j >= 16 ? 4 : 0)));
                }
            }
        }
        dst += lut_group_bytes(n);
    }
}
// tables: 16 bytes per 4 columns. Only 8 entries are computed; T[15-n] = 256 - T[n] mirrors them (exactly antisymmetric).
static void lut_tables(uint8_t * T, const block_q8_K * y, int n) {
    static const int16_t P0[8] = {-1, 1,-1, 1,-1, 1,-1, 1}, P1[8] = {-1,-1, 1, 1,-1,-1, 1, 1}, P2[8] = {-1,-1,-1,-1, 1, 1, 1, 1};
    const int16x8_t p0 = vld1q_s16(P0), p1 = vld1q_s16(P1), p2 = vld1q_s16(P2);
    const uint8x8_t rev = vcreate_u8(0x0001020304050607ull);
    for (int b = 0; b < n / QK_K; b++) {
        const int8_t * q = y[b].qs;
        for (int t = 0; t < QK_K / 4; t++, q += 4, T += 16) {
            int16x8_t s = vmulq_n_s16(p0, q[0]);
            s = vmlaq_n_s16(s, p1, q[1]); s = vmlaq_n_s16(s, p2, q[2]);
            s = vsubq_s16(s, vdupq_n_s16(q[3]));                    // entries 0-7: bit 3 clear
            const int8x8_t lo = vrshrn_n_s16(s, 2);                 // round(s / 4), |.| <= 127
            const uint8x8_t ulo = vreinterpret_u8_s8(vadd_s8(lo, vdup_n_s8((int8_t) 128)));
            const uint8x8_t uhi = vtbl1_u8(vreinterpret_u8_s8(vsub_s8(vdup_n_s8((int8_t) 128), lo)), rev);
            vst1q_u8(T, vcombine_u8(ulo, uhi));
        }
    }
}
template <int PRF>
static void lut_gemv(int n, float * s, const uint8_t * vx, const uint8_t * T, const block_q8_K * y, int nc) {
    const int nb128 = n / 128;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const uint16x8_t off = vdupq_n_u16(32 * 128);
    for (int x = 0; x < nc / 32; x++) {
        const uint8_t * grp = vx + (size_t) x * lut_group_bytes(n);
        const __fp16 * hdr = (const __fp16 *) grp;
        float32x4_t mn[8], st[8], f[8];
        for (int i = 0; i < 8; i++) { mn[i] = vcvt_f32_f16(vld1_f16(hdr + i * 4)); st[i] = vcvt_f32_f16(vld1_f16(hdr + 32 + i * 4)); f[i] = vdupq_n_f32(0); }
        const uint8_t * o = grp + 128; const uint8_t * tb = T;
        for (int g = 0; g < nb128; g++, o += 544) {
            uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
            const uint8_t * w = o + 32;
            if (PRF) __builtin_prefetch(o + 544 + PRF, 0, 0);
            for (int t = 0; t < 32; t++, w += 16, tb += 16) {
                const uint8x16_t v = vld1q_u8(w), tv = vld1q_u8(tb);
                const uint8x16_t r0 = vqtbl1q_u8(tv, vandq_u8(v, m0f));
                const uint8x16_t r1 = vqtbl1q_u8(tv, vshrq_n_u8(v, 4));
                a0 = vaddw_u8(a0, vget_low_u8(r0)); a2 = vaddw_u8(a2, vget_low_u8(r1));
                a1 = vaddw_high_u8(a1, r0);          a3 = vaddw_high_u8(a3, r1);
            }
            const float yd = 4.0f * y[g / 2].d;
            const uint16x8_t A[4] = { a0, a1, a2, a3 };
            for (int i = 0; i < 4; i++) {
                const int16x8_t si = vreinterpretq_s16_u16(vsubq_u16(A[i], off));
                const uint16x8_t c16 = vmovl_u8(vld1_u8(o + i * 8));
                const float32x4_t d0 = vfmaq_f32(mn[i * 2], vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16))), st[i * 2]);
                const float32x4_t d1 = vfmaq_f32(mn[i * 2 + 1], vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16))), st[i * 2 + 1]);
                f[i * 2]     = vfmaq_f32(f[i * 2],     vmulq_n_f32(d0, yd), vcvtq_f32_s32(vmovl_s16(vget_low_s16(si))));
                f[i * 2 + 1] = vfmaq_f32(f[i * 2 + 1], vmulq_n_f32(d1, yd), vcvtq_f32_s32(vmovl_s16(vget_high_s16(si))));
            }
        }
        for (int i = 0; i < 8; i++) vst1q_f32(s + x * 32 + i * 4, f[i]);
    }
}
struct job { int mode, n, nc, reps, prf; const void * w; const uint8_t * T; const block_q8_K * y; float * s; };
static void * run(void * p) {
    job * j = (job *) p;
    for (int r = 0; r < j->reps; r++) {
        if (j->mode == 0) ggml_gemv_q1_0_8x4_q8_K_tv(j->n, j->s, 0, j->w, j->y, 1, j->nc);
        else if (j->prf == 0) lut_gemv<0>(j->n, j->s, (const uint8_t *) j->w, j->T, j->y, j->nc);
        else lut_gemv<1024>(j->n, j->s, (const uint8_t *) j->w, j->T, j->y, j->nc);
    }
    return nullptr;
}
int main(int argc, char ** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 4096, nc = 512, nb128 = n / 128, nb = n / QK_K;
    srand(7);
    block_q1_0 * w = (block_q1_0 *) malloc(sizeof(block_q1_0) * nc * nb128);
    for (int c = 0; c < nc; c++) { const float rm = 0.002f * (1 + rand() % 9); for (int b = 0; b < nb128; b++) { block_q1_0 & B = w[c * nb128 + b]; for (int t = 0; t < 16; t++) B.qs[t] = rand(); B.d = h(rm * (0.45f + 0.55f * rand() / (float) RAND_MAX)); } }
    void * wr = malloc((size_t) nc / 8 * q1tv_group_bytes(n));
    ggml_tensor t = { GGML_TYPE_Q1_0, { n, nc, 1, 1 }, wr };
    GGML_ASSERT(repack_q1_0_to_q1x8_tv(&t, w, sizeof(block_q1_0) * nc * nb128) == 0);
    uint8_t * wl = (uint8_t *) malloc((size_t) nc / 32 * lut_group_bytes(n)); lut_repack(wl, w, n, nc);
    // Gaussian-ish activations with outliers, like a real hidden state
    float * x = (float *) malloc(4 * n);
    for (int i = 0; i < n; i++) { float a = 0; for (int k = 0; k < 6; k++) a += rand() / (float) RAND_MAX - 0.5f; x[i] = a * (i % 97 == 0 ? 6 : 1); }
    block_q8_K * q = (block_q8_K *) malloc(sizeof(block_q8_K) * nb); quant_row(x, q, n);
    uint8_t * T = (uint8_t *) malloc(n * 4); double t0 = now(); for (int r = 0; r < 100; r++) lut_tables(T, q, n); double tt = (now() - t0) / 100;
    printf("table build: %.1f us for %d columns (%.1f cycles per table at 2.5 GHz)\n", tt * 1e6, n, tt * 2.5e9 / (n / 4));
    float * sv = (float *) malloc(4 * nc), * sl = (float *) malloc(4 * nc);
    ggml_gemv_q1_0_8x4_q8_K_tv(n, sv, 0, wr, q, 1, nc); lut_gemv<0>(n, sl, wl, T, q, nc);
    double se = 0, ss = 0, ex = 0;
    for (int c = 0; c < nc; c++) {   // exact float dot as the reference
        double r = 0; for (int e = 0; e < n; e++) { const block_q1_0 & b = w[c * nb128 + e / 128]; const int m = e % 128; r += hf(b.d) * (((b.qs[m / 8] >> (m % 8)) & 1) ? 1 : -1) * x[e]; }
        ss += r * r; se += (sl[c] - r) * (sl[c] - r); ex += (sv[c] - r) * (sv[c] - r);
    }
    printf("NMSE vs exact float: bit-plane q8 %.3e   LUT %.3e\n", ex / ss, se / ss);
    int reps = 400; double a0 = now(); for (int r = 0; r < reps; r++) ggml_gemv_q1_0_8x4_q8_K_tv(n, sv, 0, wr, q, 1, nc); double a1 = now(); for (int r = 0; r < reps; r++) lut_gemv<0>(n, sl, wl, T, q, nc); double a2 = now();
    printf("cache-resident %d x %d, 1 thread: bit-plane %.2f  LUT %.2f G weights/s\n", nc, n, (double) n * nc * reps / (a1 - a0) / 1e9, (double) n * nc * reps / (a2 - a1) / 1e9);
    const int NC = 4096 * 16;  // ~36 MB, beyond the caches
    size_t b0 = (size_t) NC / 8 * q1tv_group_bytes(n), b1 = (size_t) NC / 32 * lut_group_bytes(n);
    uint8_t * big = (uint8_t *) malloc(b0), * bigl = (uint8_t *) malloc(b1); for (size_t i = 0; i < b0; i++) big[i] = (uint8_t)(i * 2654435761u >> 11); for (size_t i = 0; i < b1; i++) bigl[i] = (uint8_t)(i * 2654435761u >> 11);
    for (int mode = 0; mode < 3; mode++) for (int th = 1; th <= 4; th++) {
        pthread_t p[4]; job j[4]; float * out = (float *) malloc(NC * 4); reps = 6; double s0 = now();
        for (int i = 0; i < th; i++) {
            int c0 = NC / th * i / 32 * 32, c1 = (i + 1 == th ? NC : NC / th * (i + 1) / 32 * 32);
            j[i] = job{ mode, n, c1 - c0, reps, mode == 2, mode == 0 ? (const void *)(big + (size_t) c0 / 8 * q1tv_group_bytes(n)) : (const void *)(bigl + (size_t) c0 / 32 * lut_group_bytes(n)), T, q, out + c0 };
            pthread_create(&p[i], nullptr, run, &j[i]);
        }
        for (int i = 0; i < th; i++) pthread_join(p[i], nullptr);
        double e = now() - s0;
        printf("DRAM %s %d thr: %.2f G weights/s (%.2f GB/s)\n", mode == 0 ? "bit-plane " : mode == 1 ? "LUT       " : "LUT+PRFM  ", th, (double) n * NC * reps / e / 1e9, (mode == 0 ? b0 : b1) * reps / e / 1e9);
        free(out);
    }
    return 0;
}
