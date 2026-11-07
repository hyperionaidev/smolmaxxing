// gguf-punch: free the flash under byte ranges of a file without changing its size (FALLOC_FL_PUNCH_HOLE). Once the repack cache holds the
// experts, the GGUF's own copies are never read, and on this box flash space is what limits how large a model can be.
// usage: gguf-punch <file> <ranges file: "offset length" per line>
#define _GNU_SOURCE
#include <fcntl.h>
#include <linux/falloc.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
int main(int c, char ** v) {
    if (c < 3) { fprintf(stderr, "usage\n"); return 2; }
    int fd = open(v[1], O_RDWR); if (fd < 0) { perror(v[1]); return 1; }
    FILE * f = fopen(v[2], "r"); if (!f) { perror(v[2]); return 1; }
    struct stat a, b; fstat(fd, &a);
    long long off, len, total = 0;
    while (fscanf(f, "%lld %lld", &off, &len) == 2) {
        if (fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, off, len) != 0) { perror("fallocate"); return 1; }
        total += len;
    }
    fsync(fd); fstat(fd, &b);
    printf("punched %.1f MB; size %lld, blocks on flash %.1f -> %.1f MB\n", total / 1e6, (long long) b.st_size, a.st_blocks * 512 / 1e6, b.st_blocks * 512 / 1e6);
    return 0;
}
