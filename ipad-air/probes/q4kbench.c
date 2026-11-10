// Raw Q4_K x Q8_K dot throughput on this CPU, 2 threads, weights from the model file (mapped), activations random int8.
#include <arm_neon.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } bq4;     // block_q4_K, 144 bytes
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } bq8;                // block_q8_K
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
static float h2f(uint16_t h){__fp16 x;memcpy(&x,&h,2);return x;}
static float dot_new(int nb, const bq4 * x, const bq8 * y) {
    const uint8x16_t m4b = vdupq_n_u8(0xf); float sumf = 0; uint32_t utmp[4];
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d), dmin = y[i].d * h2f(x[i].dmin);
        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));
        memcpy(utmp, x[i].scales, 12);
        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4); utmp[0] &= kmask1;
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        sumf -= dmin * vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)), vmull_high_s16(q8sums, mins)));
        const uint8_t * scales = (const uint8_t *) utmp; const uint8_t * q4 = x[i].qs; const int8_t * q8 = y[i].qs;
#define J(j, pl, ph) int16x8_t pl, ph; { \
        const uint8x16_t b0 = vld1q_u8(q4 + 32*(j)), b1 = vld1q_u8(q4 + 32*(j) + 16); \
        const int8x16_t y0 = vld1q_s8(q8 + 64*(j)), y1 = vld1q_s8(q8 + 64*(j) + 16), y2 = vld1q_s8(q8 + 64*(j) + 32), y3 = vld1q_s8(q8 + 64*(j) + 48); \
        const int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(b0, m4b)), l1 = vreinterpretq_s8_u8(vandq_u8(b1, m4b)); \
        const int8x16_t h0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)); \
        pl = vmull_s8(vget_low_s8(l0), vget_low_s8(y0)); ph = vmull_s8(vget_low_s8(h0), vget_low_s8(y2)); \
        pl = vmlal_high_s8(pl, l0, y0); ph = vmlal_high_s8(ph, h0, y2); \
        pl = vmlal_s8(pl, vget_low_s8(l1), vget_low_s8(y1)); ph = vmlal_s8(ph, vget_low_s8(h1), vget_low_s8(y3)); \
        pl = vmlal_high_s8(pl, l1, y1); ph = vmlal_high_s8(ph, h1, y3); }
        J(0, pl0, ph0) J(1, pl1, ph1) J(2, pl2, ph2) J(3, pl3, ph3)
        int32x4_t a0 = vmull_n_s16(vget_low_s16(pl0), scales[0]), a1 = vmull_high_n_s16(pl0, scales[0]);
        int32x4_t a2 = vmull_n_s16(vget_low_s16(ph0), scales[1]), a3 = vmull_high_n_s16(ph0, scales[1]);
        a0 = vmlal_n_s16(a0, vget_low_s16(pl1), scales[2]); a1 = vmlal_high_n_s16(a1, pl1, scales[2]);
        a2 = vmlal_n_s16(a2, vget_low_s16(ph1), scales[3]); a3 = vmlal_high_n_s16(a3, ph1, scales[3]);
        a0 = vmlal_n_s16(a0, vget_low_s16(pl2), scales[4]); a1 = vmlal_high_n_s16(a1, pl2, scales[4]);
        a2 = vmlal_n_s16(a2, vget_low_s16(ph2), scales[5]); a3 = vmlal_high_n_s16(a3, ph2, scales[5]);
        a0 = vmlal_n_s16(a0, vget_low_s16(pl3), scales[6]); a1 = vmlal_high_n_s16(a1, pl3, scales[6]);
        a2 = vmlal_n_s16(a2, vget_low_s16(ph3), scales[7]); a3 = vmlal_high_n_s16(a3, ph3, scales[7]);
        sumf += d * vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
    }
    return sumf;
}
struct job { const bq4 * w; long rows; int nb; const bq8 * y; float * out; };
static void * run(void * v) { struct job * j = v; for (long r = 0; r < j->rows; r++) j->out[r] = dot_new(j->nb, j->w + r * j->nb, j->y); return 0; }
int main(int c, char ** v) {
    int fd = open(v[1], O_RDONLY); long off = atol(v[2]); size_t mb = atoi(v[3]);
    const int nb = 6;   // 1536 columns
    long rows = (long) (mb << 20) / (nb * 144);
    long pg = off & ~16383L; char * m = mmap(0, (mb << 20) + 16384 * 2, PROT_READ, MAP_SHARED, fd, pg);
    const bq4 * w = (const bq4 *) (m + (off - pg));
    bq8 y[6]; for (int i = 0; i < 6; i++) { y[i].d = 0.01f; for (int k = 0; k < 256; k++) y[i].qs[k] = (int8_t) (rand() % 255 - 127); for (int k = 0; k < 16; k++) { int s = 0; for (int q = 0; q < 16; q++) s += y[i].qs[16*k+q]; y[i].bsums[k] = s; } }
    float * out = malloc(rows * sizeof(float)); volatile float sink = 0;
    for (int pass = 0; pass < 4; pass++) {
        pthread_t t[2]; struct job j[2]; double t0 = now();
        for (int k = 0; k < 2; k++) { j[k] = (struct job){ w + (rows / 2) * k * nb, rows / 2, nb, y, out + (rows / 2) * k }; pthread_create(&t[k], 0, run, &j[k]); }
        for (int k = 0; k < 2; k++) pthread_join(t[k], 0);
        double dt = now() - t0; sink += out[0];
        printf("pass %d: %.2f G weights/s, %.2f GB/s\n", pass, rows * 1536.0 / dt / 1e9, rows * nb * 144.0 / dt / 1e9);
    }
    return 0;
}
