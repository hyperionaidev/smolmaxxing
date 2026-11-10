// Largest single mapping of a file and of anonymous memory (reserved, never touched), in 64 MB steps.
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
int main(int c, char ** v) {
    int fd = open(v[1], O_RDONLY); struct stat st; fstat(fd, &st);
    size_t step = 64ull << 20, best = 0;
    for (size_t n = step; n <= (size_t) st.st_size; n += step) {
        void * p = mmap(NULL, n, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) break;
        munmap(p, n); best = n;
    }
    printf("file %s: %lld MB, largest file mapping %zu MB\n", v[1], (long long) st.st_size >> 20, best >> 20);
    best = 0;
    for (size_t n = step; n <= (16ull << 30); n += step) {
        void * p = mmap(NULL, n, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (p == MAP_FAILED) break;
        munmap(p, n); best = n;
    }
    printf("largest anonymous reservation %zu MB\n", best >> 20);
    // two mappings at once: does the limit apply to the total?
    size_t a = 1200ull << 20;
    void * p1 = mmap(NULL, a, PROT_READ, MAP_SHARED, fd, 0);
    size_t more = 0;
    for (size_t n = step; n <= (16ull << 30); n += step) {
        void * p = mmap(NULL, n, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (p == MAP_FAILED) break;
        munmap(p, n); more = n;
    }
    printf("with a 1200 MB file mapping held (%s), largest extra reservation %zu MB\n", p1 == MAP_FAILED ? "failed" : "ok", more >> 20);
    return 0;
}
