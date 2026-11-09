// Compares two NEON kernels for the Q1_0 x Q8_0 dot product on CPUs without dotprod:
// the stock llama.cpp path, and a multiplication-free one (kernel_new, also in the patch).
// Checks they return bit-identical results, including extreme inputs, then times them.

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK1_0 128

typedef struct { uint16_t d; uint8_t qs[QK1_0 / 8]; } block_q1_0;
typedef struct { uint16_t d; int8_t  qs[32]; } block_q8_0;

static inline float h2f(uint16_t h) {
    __fp16 v;
    memcpy(&v, &h, sizeof(v));
    return (float) v;
}

static uint64_t table_b2b_0[256];

static void init_table(void) {
    for (int b = 0; b < 256; b++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) {
            if (b & (1 << j)) {
                v |= (uint64_t) 0x10 << (8 * j);
            }
        }
        table_b2b_0[b] = v;
    }
}

// llama.cpp's fallback for vdotq_s32
static inline int32x4_t vdot_emulated(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8(a),  vget_low_s8(b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

__attribute__((noinline))
static float kernel_old(int nb, const block_q1_0 * x, const block_q8_0 * y) {
    float32x4_t sumv = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++) {
        const float d0 = h2f(x[i].d);
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i * 4 + k];
            const float d1 = h2f(yb->d);
            const uint8_t * bits = &x[i].qs[k * 4];
            const int8x16_t y0 = vld1q_s8(yb->qs);
            const int8x16_t y1 = vld1q_s8(yb->qs + 16);
            int8x8_t s0 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(table_b2b_0[bits[0]]), 4));
            int8x8_t s1 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(table_b2b_0[bits[1]]), 4));
            int8x8_t s2 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(table_b2b_0[bits[2]]), 4));
            int8x8_t s3 = vreinterpret_s8_u8(vshr_n_u8(vcreate_u8(table_b2b_0[bits[3]]), 4));
            const int8x8_t one = vdup_n_s8(1);
            s0 = vsub_s8(vadd_s8(s0, s0), one);
            s1 = vsub_s8(vadd_s8(s1, s1), one);
            s2 = vsub_s8(vadd_s8(s2, s2), one);
            s3 = vsub_s8(vadd_s8(s3, s3), one);
            const int32x4_t p0 = vdot_emulated(vdupq_n_s32(0), vcombine_s8(s0, s1), y0);
            const int32x4_t p1 = vdot_emulated(p0, vcombine_s8(s2, s3), y1);
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p1), d0 * d1);
        }
    }
    return vaddvq_f32(sumv);
}

// The weights are +1 or -1, so sum(w * y) = 2 * sum(y where the bit is set) - sum(y).
// No multiplication is needed. The per-lane sums match kernel_old exactly, so the
// floating-point result is identical too.
__attribute__((noinline))
static float kernel_new(int nb, const block_q1_0 * x, const block_q8_0 * y) {
    static const uint8_t k_bit[16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    static const uint8_t k_lo[16]  = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
    static const uint8_t k_hi[16]  = { 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 };
    const uint8x16_t bit = vld1q_u8(k_bit);
    const uint8x16_t lo  = vld1q_u8(k_lo);
    const uint8x16_t hi  = vld1q_u8(k_hi);

    float32x4_t sumv = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++) {
        const float d0 = h2f(x[i].d);
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * yb = &y[i * 4 + k];
            const float d1 = h2f(yb->d);
            uint32_t w;
            memcpy(&w, &x[i].qs[k * 4], sizeof(w));
            const uint8x16_t b  = vreinterpretq_u8_u32(vdupq_n_u32(w));
            const int8x16_t  m0 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(b, lo), bit));
            const int8x16_t  m1 = vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(b, hi), bit));
            const int8x16_t  y0 = vld1q_s8(yb->qs);
            const int8x16_t  y1 = vld1q_s8(yb->qs + 16);

            const int16x8_t all = vpadalq_s8(vpaddlq_s8(y0), y1);
            const int16x8_t set = vpadalq_s8(vpaddlq_s8(vandq_s8(y0, m0)), vandq_s8(y1, m1));
            const int16x8_t q   = vsubq_s16(vshlq_n_s16(set, 1), all);
            const int32x4_t p   = vaddl_s16(vget_low_s16(q), vget_high_s16(q));

            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), d0 * d1);
        }
    }
    return vaddvq_f32(sumv);
}

static uint32_t rnd_state = 0x12345678;
static uint32_t rnd(void) {
    rnd_state ^= rnd_state << 13;
    rnd_state ^= rnd_state >> 17;
    rnd_state ^= rnd_state << 5;
    return rnd_state;
}

static uint16_t rnd_half(void) {
    __fp16 v = (__fp16) ((float) (1 + (rnd() & 1023)) / 512.0f);
    uint16_t h;
    memcpy(&h, &v, sizeof(h));
    return h;
}

static void fill(int nb, block_q1_0 * x, block_q8_0 * y, int mode) {
    for (int i = 0; i < nb; i++) {
        x[i].d = rnd_half();
        for (int j = 0; j < QK1_0 / 8; j++) {
            x[i].qs[j] = mode == 1 ? 0xff : mode == 2 ? 0x00 : (uint8_t) rnd();
        }
    }
    for (int i = 0; i < nb * 4; i++) {
        y[i].d = rnd_half();
        for (int j = 0; j < 32; j++) {
            y[i].qs[j] = mode == 1 ? -128 : mode == 2 ? 127 : mode == 3 ? (rnd() & 1 ? -128 : 127) : (int8_t) rnd();
        }
    }
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    init_table();

    enum { NB = 8, ROUNDS = 200000 };
    block_q1_0 * x = malloc(sizeof(block_q1_0) * NB);
    block_q8_0 * y = malloc(sizeof(block_q8_0) * NB * 4);

    long checked = 0, diff = 0;
    for (int mode = 0; mode < 4; mode++) {
        for (int r = 0; r < ROUNDS; r++) {
            fill(NB, x, y, mode);
            const float a = kernel_old(NB, x, y);
            const float b = kernel_new(NB, x, y);
            if (memcmp(&a, &b, sizeof(a)) != 0) {
                diff++;
            }
            checked++;
            if (mode != 0 && r > 2000) {
                break;
            }
        }
    }
    printf("checked %ld vectors, %ld differ\n", checked, diff);

    fill(NB, x, y, 0);
    enum { REPS = 2000000 };
    volatile float sink = 0;
    // the barrier stops the compiler from treating the calls as repeatable and skipping them
#define BARRIER() __asm__ volatile("" : : "r"(x), "r"(y) : "memory")
    for (int pass = 0; pass < 3; pass++) {
        double t0 = now();
        for (int r = 0; r < REPS; r++) { BARRIER(); sink += kernel_old(NB, x, y); }
        double t1 = now();
        for (int r = 0; r < REPS; r++) { BARRIER(); sink += kernel_new(NB, x, y); }
        double t2 = now();
        const double weights = (double) REPS * NB * QK1_0;
        printf("old %.2f Gw/s   new %.2f Gw/s   x%.2f\n",
               weights / (t1 - t0) / 1e9, weights / (t2 - t1) / 1e9, (t1 - t0) / (t2 - t1));
    }
    (void) sink;
    free(x);
    free(y);
    return diff != 0;
}
