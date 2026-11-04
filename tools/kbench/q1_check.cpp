// q1_check: exactness + speed of the tvbox Q1_0 (1-bit, group 128) bit-plane kernels exactly as compiled into
// llama.cpp (#includes ggml-cpu/repack-q1-tv.inc through a shim of the ggml types).
// Random Q1_0 weights; activations quantized to Q8_K (gemv) and interleaved Q8_Kx4 (gemm).
// Reference = the same integer quants in double. Then G weights/s: 1 thread cache-resident, and a
// DRAM-sized matrix (12288 x 4096, ~7 MB of bits... x NMAT copies) with 1..4 threads.
// usage: q1_check
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#define QK_K 256
#define GGML_RESTRICT __restrict
#define GGML_UNUSED(x) (void)(x)
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "assert %s\n", #x); abort(); } } while (0)
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
static void interleave4(const block_q8_K * r[4], block_q8_Kx4 * y, int nb) {   // ggml_quantize_mat_q8_K_4x4 layout
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
static double ref_dot(int n, const block_q1_0 * x, const block_q8_K * y) {
    double s = 0;
    for (int e = 0; e < n; e++) {
        const block_q1_0 & b = x[e / 128];
        const int m = e % 128, sg = ((b.qs[m / 8] >> (m % 8)) & 1) ? 1 : -1;
        s += (double) hf(b.d) * sg * y[e / QK_K].qs[e % QK_K] * y[e / QK_K].d;
    }
    return s;
}
// scale of row c, 128-group g, as stored in the compact layout
static double compact_d(const uint8_t * wr, int n, int c, int g) {
    const uint8_t * grp = wr + (size_t) (c / 8) * q1tv_group_bytes(n);
    const ggml_half * hdr = (const ggml_half *) grp;
    const block_q1_0x8c * w = (const block_q1_0x8c *) (grp + 32);
    const int j = c % 8;
    return (double) hf(hdr[j]) + w[g / 2].d[(g % 2) * 8 + j] * (double) hf(hdr[8 + j]);
}
static double ref_dot_c(int n, const block_q1_0 * x, const block_q8_K * y, const uint8_t * wr, int c) {
    double s = 0;
    for (int e = 0; e < n; e++) {
        const block_q1_0 & b = x[e / 128];
        const int m = e % 128, sg = ((b.qs[m / 8] >> (m % 8)) & 1) ? 1 : -1;
        s += compact_d(wr, n, c, e / 128) * sg * y[e / QK_K].qs[e % QK_K] * y[e / QK_K].d;
    }
    return s;
}
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

struct job { int n, nc, reps; const void * w; const block_q8_K * y; float * s; };
static void * run(void * p) {
    job * j = (job *) p;
    for (int r = 0; r < j->reps; r++) ggml_gemv_q1_0_8x4_q8_K_tv(j->n, j->s, 0, j->w, j->y, 1, j->nc);
    return nullptr;
}

int main() {
    const int n = 4096, nc = 512, nb128 = n / 128, nb = n / QK_K;
    srand(7);
    block_q1_0 * w = (block_q1_0 *) malloc(sizeof(block_q1_0) * nc * nb128);
    for (int i = 0; i < nc * nb128; i++) { for (int t = 0; t < 16; t++) w[i].qs[t] = rand(); w[i].d = h(0.001f * (1 + rand() % 9)); }
    // Bonsai-like scales: positive, 0.45-1.0x of a per-row maximum
    for (int c = 0; c < nc; c++) { const float rm = 0.002f * (1 + rand() % 9); for (int b = 0; b < nb128; b++) w[c * nb128 + b].d = h(rm * (0.45f + 0.55f * rand() / (float) RAND_MAX)); }
    void * wr = malloc((size_t) nc / 8 * q1tv_group_bytes(n));
    ggml_tensor t = { GGML_TYPE_Q1_0, { n, nc, 1, 1 }, wr };
    GGML_ASSERT(repack_q1_0_to_q1x8_tv(&t, w, sizeof(block_q1_0) * nc * nb128) == 0);
    float * x = (float *) malloc(4 * n * 4);
    for (int i = 0; i < 4 * n; i++) x[i] = (rand() / (float) RAND_MAX - 0.5f) * (i % 97 == 0 ? 20 : 1);
    block_q8_K * q[4]; for (int m = 0; m < 4; m++) { q[m] = (block_q8_K *) malloc(sizeof(block_q8_K) * nb); quant_row(x + m * n, q[m], n); }
    block_q8_Kx4 * q4 = (block_q8_Kx4 *) malloc(sizeof(block_q8_Kx4) * nb);
    const block_q8_K * qr[4] = { q[0], q[1], q[2], q[3] };
    interleave4(qr, q4, nb);
    float * sv = (float *) malloc(4 * nc * 4), * sg = (float *) malloc(4 * nc * 4);
    for (int m = 0; m < 4; m++) ggml_gemv_q1_0_8x4_q8_K_tv(n, sv + m * nc, 0, wr, q[m], 1, nc);
    ggml_gemm_q1_0_8x4_q8_K_tv(n, sg, nc, wr, q4, 4, nc);
    double ev = 0, eg = 0, mag = 0, eo = 0;
    for (int m = 0; m < 4; m++) for (int c = 0; c < nc; c++) {
        const double r = ref_dot_c(n, w + (size_t) c * nb128, q[m], (const uint8_t *) wr, c);
        const double ro = ref_dot(n, w + (size_t) c * nb128, q[m]);
        mag = fmax(mag, fabs(r)); ev = fmax(ev, fabs(sv[m * nc + c] - r)); eg = fmax(eg, fabs(sg[m * nc + c] - r)); eo = fmax(eo, fabs(r - ro));
    }
    printf("max |ref| %.3f   max abs err vs compact-scale reference: gemv %.2e  gemm %.2e   (%s)\n", mag, ev, eg, (ev < 1e-4 * mag && eg < 1e-4 * mag) ? "PASS" : "FAIL");
    double sd = 0, sm = 0;
    for (int c = 0; c < nc; c++) for (int g = 0; g < nb128; g++) { const double o = hf(w[c * nb128 + g].d); sd = fmax(sd, fabs(compact_d((const uint8_t *) wr, n, c, g) - o) / o); sm += fabs(compact_d((const uint8_t *) wr, n, c, g) - o) / o; }
    printf("8-bit scales vs fp16: max rel err %.4f, mean %.5f; dot products max abs diff %.2e (%.2e of max)\n", sd, sm / (nc * nb128), eo, eo / mag);
    int reps = 400; double t0 = now();
    for (int r = 0; r < reps; r++) for (int m = 0; m < 4; m++) ggml_gemv_q1_0_8x4_q8_K_tv(n, sv + m * nc, 0, wr, q[m], 1, nc);
    double t1 = now();
    for (int r = 0; r < reps; r++) ggml_gemm_q1_0_8x4_q8_K_tv(n, sg, nc, wr, q4, 4, nc);
    double t2 = now();
    printf("cache-resident (%d x %d), 1 thread: gemv %.2f  gemm(4 rows) %.2f G weight-rows/s\n", nc, n,
           4.0 * n * nc * reps / (t1 - t0) / 1e9, 4.0 * n * nc * reps / (t2 - t1) / 1e9);
    // DRAM: 4 threads each own a slice of a 12288 x 4096 x 4 matrix (~28 MB of repacked bits, > caches)
    const int NC = 12288 * 4;
    void * big = malloc((size_t) NC / 8 * q1tv_group_bytes(n));
    memset(big, 0x5a, (size_t) NC / 8 * q1tv_group_bytes(n));
    for (int th = 1; th <= 4; th++) {
        pthread_t p[4]; job j[4]; float * out = (float *) malloc(NC * 4);
        reps = 6; double a = now();
        for (int i = 0; i < th; i++) {
            const int c0 = NC / th * i, c1 = NC / th * (i + 1);
            j[i] = job{ n, c1 - c0, reps, (const char *) big + (size_t) c0 / 8 * q1tv_group_bytes(n), q[0], out + c0 };
            pthread_create(&p[i], nullptr, run, &j[i]);
        }
        for (int i = 0; i < th; i++) pthread_join(p[i], nullptr);
        double e = now() - a;
        printf("DRAM %d x %d (%.1f MB), %d thread(s): gemv %.2f G weights/s (%.2f GB/s)\n", NC, n,
               (double) NC / 8 * q1tv_group_bytes(n) / 1e6, th, (double) n * NC * reps / e / 1e9,
               (double) NC / 8 * q1tv_group_bytes(n) * reps / e / 1e9);
        free(out);
    }
    return 0;
}
