// rangecat FILE OFFSET LENGTH: write LENGTH bytes of FILE starting at OFFSET to stdout (Android's toybox dd can't skip by bytes).
// Used by tools/gguf_split_adb.py to cut a GGUF on the phone into llama.cpp split shards on the Mac.
#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int c, char ** v) {
    if (c != 4) { fprintf(stderr, "usage: rangecat FILE OFFSET LENGTH\n"); return 2; }
    int fd = open(v[1], O_RDONLY); if (fd < 0) { perror(v[1]); return 1; }
    long long off = atoll(v[2]), left = atoll(v[3]);
    static char buf[4 << 20];
    while (left > 0) {
        ssize_t r = pread(fd, buf, left < (long long) sizeof buf ? (size_t) left : sizeof buf, off);
        if (r <= 0) { perror("pread"); return 1; }
        for (ssize_t w = 0; w < r; ) { ssize_t k = write(1, buf + w, r - w); if (k <= 0) { perror("write"); return 1; } w += k; }
        off += r; left -= r;
    }
    return 0;
}
