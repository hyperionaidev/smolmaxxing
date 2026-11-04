// q2k_check: exactness test of the tvbox Q2_K kernels exactly as compiled into llama.cpp
// (#includes ggml-cpu/repack-q2k-tv.inc through a tiny shim of the ggml types).
// Random Q2_K weights, 4 activation rows quantized to Q8_K (per row, as the gemv path does) and to the
// interleaved Q8_Kx4 layout (copy of ggml_quantize_mat_q8_K_4x4_generic, as the gemm path does).
// Reference = the same integer quants dequantized in double. Also times gemm vs 4x gemv.
// usage: q2k_check
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define QK_K 256
#define GGML_RESTRICT __restrict
#define GGML_UNUSED(x) (void)(x)
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "assert %s\n", #x); abort(); } } while (0)
#define GGML_ABORT(...) abort()
typedef uint16_t ggml_half;
enum { GGML_TYPE_Q2_K = 10 };
struct ggml_tensor { int type; int64_t ne[4]; void * data; };
static int64_t ggml_nrows(const ggml_tensor * t) { return t->ne[1] * t->ne[2] * t->ne[3]; }
#define GGML_COMMON_AGGR_U u
#define GGML_COMMON_AGGR_S s
struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; union { struct { ggml_half d; ggml_half dmin; } s; uint32_t dm; } u; };
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; };
struct block_q2_Kx8 { ggml_half d[8]; ggml_half dmin[8]; uint8_t scales[128]; uint8_t qs[512]; };
struct block_q8_Kx4 { float d[4]; int8_t qs[QK_K * 4]; int16_t bsums[QK_K / 4]; };
#include "../../third_party/llama.cpp/ggml/src/ggml-cpu/repack-q2k-tv.inc"

static ggml_half h(float f) { __fp16 x = (__fp16) f; ggml_half r; memcpy(&r, &x, 2); return r; }
static float hf(ggml_half v) { __fp16 x; memcpy(&x, &v, 2); return (float) x; }

static void quant_row(const float * x, block_q8_K * y, int k) {   // same math as quantize_row_q8_K_ref
    for (int i = 0; i < k / QK_K; i++) {
        float amax = 0, mx = 0;
        for (int j = 0; j < QK_K; j++) { float a = fabsf(x[i * QK_K + j]); if (a > amax) { amax = a; mx = x[i * QK_K + j]; } }
        const float is = amax ? -127.f / mx : 0;
        for (int j = 0; j < QK_K; j++) { int v = (int) nearbyintf(is * x[i * QK_K + j]); y[i].qs[j] = v > 127 ? 127 : v; }
        for (int j = 0; j < 16; j++) { int s = 0; for (int t = 0; t < 16; t++) s += y[i].qs[j * 16 + t]; y[i].bsums[j] = s; }
        y[i].d = amax ? 1 / is : 0;
    }
}
// interleave 4 already-quantized rows into the Q8_Kx4 layout of ggml_quantize_mat_q8_K_4x4_generic
static void interleave4(const block_q8_K * r[4], block_q8_Kx4 * y, int nb) {
    for (int i = 0; i < nb; i++) {
        for (int m = 0; m < 4; m++) y[i].d[m] = r[m][i].d;
        memset(y[i].bsums, 0, sizeof y[i].bsums);
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / 16) * 4 + (j % 4), src_id = (j % 16) / 4;
            int index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
            y[i].qs[j] = r[src_id][i].qs[src_offset];
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}
static double ref_dot(int n, const block_q2_K * x, const block_q8_K * y) {
    double s = 0;
    for (int b = 0; b < n / QK_K; b++) {
        const double d = (double) hf(x[b].u.s.d) * y[b].d, dm = (double) hf(x[b].u.s.dmin) * y[b].d;
        for (int e = 0; e < QK_K; e++) {
            const int hh = e / 128, r = e % 128, sh = r / 32, l = r % 32, sb = e / 16;
            const int q = (x[b].qs[hh * 32 + l] >> (2 * sh)) & 3;
            s += (d * (x[b].scales[sb] & 15) * q - dm * (x[b].scales[sb] >> 4)) * y[b].qs[e];
        }
    }
    return s;
}
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main() {
    const int n = 2560, nc = 512, nb = n / QK_K;
    srand(7);
    block_q2_K * w = (block_q2_K *) malloc(sizeof(block_q2_K) * nc * nb);
    for (int i = 0; i < nc * nb; i++) { for (int t = 0; t < 16; t++) w[i].scales[t] = rand(); for (int t = 0; t < 64; t++) w[i].qs[t] = rand();
        w[i].u.s.d = h(0.001f * (1 + rand() % 9)); w[i].u.s.dmin = h(0.0007f * (1 + rand() % 5)); }
    block_q2_Kx8 * wr = (block_q2_Kx8 *) malloc(sizeof(block_q2_K) * nc * nb);
    ggml_tensor t = { GGML_TYPE_Q2_K, { n, nc, 1, 1 }, wr };
    GGML_ASSERT(repack_q2_K_to_q2_K_8x4_tv(&t, w, sizeof(block_q2_K) * nc * nb) == 0);
    float * x = (float *) malloc(4 * n * 4);
    for (int i = 0; i < 4 * n; i++) x[i] = (rand() / (float) RAND_MAX - 0.5f) * (i % 97 == 0 ? 20 : 1);
    block_q8_K * q[4]; for (int m = 0; m < 4; m++) { q[m] = (block_q8_K *) malloc(sizeof(block_q8_K) * nb); quant_row(x + m * n, q[m], n); }
    block_q8_Kx4 * q4 = (block_q8_Kx4 *) malloc(sizeof(block_q8_Kx4) * nb);
    const block_q8_K * qr[4] = { q[0], q[1], q[2], q[3] };
    interleave4(qr, q4, nb);
    float * sv = (float *) malloc(4 * nc * 4), * sg = (float *) malloc(4 * nc * 4);
    for (int m = 0; m < 4; m++) ggml_gemv_q2_K_8x4_q8_K_tv(n, sv + m * nc, 0, wr, q[m], 1, nc);
    ggml_gemm_q2_K_8x4_q8_K_tv(n, sg, nc, wr, q4, 4, nc);
    double ev = 0, eg = 0, mag = 0;
    for (int m = 0; m < 4; m++) for (int c = 0; c < nc; c++) {
        const double r = ref_dot(n, w + (size_t) c * nb, q[m]);
        mag = fmax(mag, fabs(r)); ev = fmax(ev, fabs(sv[m * nc + c] - r)); eg = fmax(eg, fabs(sg[m * nc + c] - r));
    }
    printf("max |ref| %.3f   max abs err: gemv %.2e  gemm %.2e   (%s)\n", mag, ev, eg, (ev < 1e-4 * mag && eg < 1e-4 * mag) ? "PASS" : "FAIL");
    const int reps = 200; double t0 = now();
    for (int r = 0; r < reps; r++) for (int m = 0; m < 4; m++) ggml_gemv_q2_K_8x4_q8_K_tv(n, sv + m * nc, 0, wr, q[m], 1, nc);
    double t1 = now();
    for (int r = 0; r < reps; r++) ggml_gemm_q2_K_8x4_q8_K_tv(n, sg, nc, wr, q4, 4, nc);
    double t2 = now();
    printf("4 rows, 1 thread, cache-resident: 4x gemv %.2f  gemm %.2f G weight-rows/s\n", 4.0 * n * nc * reps / (t1 - t0) / 1e9, 4.0 * n * nc * reps / (t2 - t1) / 1e9);
    return 0;
}
