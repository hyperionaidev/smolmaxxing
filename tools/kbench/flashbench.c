// flashbench: sequential read throughput of a file on the TV's eMMC, for sizing weight streaming.
// Modes: O_DIRECT pread (bypasses the page cache) with block size B and T threads reading interleaved
// blocks of one region; buffered pread after POSIX_FADV_DONTNEED (cold page cache) for comparison.
// Each run reads a different 128 MB region so nothing is cached.
// usage: flashbench <file>
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
typedef struct { int fd; off_t base; size_t len, bs; int tid, nt; } job;
static void * rd(void * p) {
    job * j = p; void * buf; posix_memalign(&buf, 4096, j->bs);
    for (size_t o = (size_t) j->tid * j->bs; o < j->len; o += (size_t) j->nt * j->bs) {
        if (pread(j->fd, buf, j->bs, j->base + (off_t) o) <= 0) { perror("pread"); break; }
    }
    free(buf); return NULL;
}
int main(int argc, char ** argv) {
    struct stat st; stat(argv[1], &st);
    const size_t REG = 128 << 20; off_t base = 0;
    const size_t bss[] = { 128 << 10, 512 << 10, 2 << 20, 8 << 20 };
    const int nts[] = { 1, 2, 4 };
    for (int direct = 1; direct >= 0; direct--) {
        for (int b = 0; b < 4; b++) for (int t = 0; t < 3; t++) {
            if (!direct && t > 0) continue;
            if (base + (off_t) REG > st.st_size) base = 0;
            int fd = open(argv[1], O_RDONLY | (direct ? O_DIRECT : 0));
            if (!direct) posix_fadvise(fd, base, REG, POSIX_FADV_DONTNEED);
            pthread_t th[4]; job j[4]; double a = now();
            for (int i = 0; i < nts[t]; i++) { j[i] = (job){ fd, base, REG, bss[b], i, nts[t] }; pthread_create(&th[i], NULL, rd, &j[i]); }
            for (int i = 0; i < nts[t]; i++) pthread_join(th[i], NULL);
            double e = now() - a;
            printf("%-8s bs %5zu KB threads %d: %6.1f MB/s\n", direct ? "O_DIRECT" : "buffered", bss[b] >> 10, nts[t], REG / e / 1e6);
            fflush(stdout);
            close(fd); base += REG;
        }
    }
    return 0;
}
