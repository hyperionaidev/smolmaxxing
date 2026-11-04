#define _GNU_SOURCE
// tvfetch: minimal HTTP/1.0 downloader for the TV box (static aarch64), because adb over Wi-Fi wedges on
// multi-GB transfers. Resumes: if <out> exists, asks for "Range: bytes=<size>-" and appends. Retries forever
// on connection errors (5 s backoff) until the expected length is reached.
// usage: tvfetch <host> <port> </path> <out>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static long long fsize(const char *p) { struct stat st; return stat(p, &st) == 0 ? (long long) st.st_size : 0; }

static int attempt(const char *host, const char *port, const char *path, const char *out, long long *total) {
    struct addrinfo hints = {0}, *ai;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &ai)) return -1;
    int s = socket(ai->ai_family, SOCK_STREAM, 0);
    if (s < 0 || connect(s, ai->ai_addr, ai->ai_addrlen)) { freeaddrinfo(ai); if (s >= 0) close(s); return -1; }
    freeaddrinfo(ai);
    long long have = fsize(out);
    char req[1024];
    snprintf(req, sizeof req, "GET %s HTTP/1.0\r\nHost: %s\r\nRange: bytes=%lld-\r\n\r\n", path, host, have);
    if (write(s, req, strlen(req)) < 0) { close(s); return -1; }
    // read headers
    static char buf[1 << 20];
    int n = 0, hdr_end = -1;
    while (hdr_end < 0) {
        int r = read(s, buf + n, sizeof buf - 1 - n);
        if (r <= 0) { close(s); return -1; }
        n += r; buf[n] = 0;
        char *e = strstr(buf, "\r\n\r\n");
        if (e) hdr_end = (int) (e - buf) + 4;
    }
    int code = atoi(buf + 9);
    if (code == 416) { close(s); return 0; }  // already complete
    if (code != 206 && code != 200) { fprintf(stderr, "HTTP %d\n", code); close(s); return -2; }
    char *cr = strcasestr(buf, "Content-Range:");
    if (cr && (cr = strchr(cr, '/'))) *total = atoll(cr + 1);
    else { char *cl = strcasestr(buf, "Content-Length:"); if (cl) *total = have + atoll(cl + 15); }
    if (code == 200) have = 0;
    FILE *f = fopen(out, code == 200 ? "wb" : "ab");
    if (!f) { close(s); return -2; }
    fwrite(buf + hdr_end, 1, n - hdr_end, f);
    long long got = have + (n - hdr_end), last = got;
    time_t t0 = time(0), tl = t0;
    for (;;) {
        int r = read(s, buf, sizeof buf);
        if (r <= 0) break;
        fwrite(buf, 1, r, f); got += r;
        time_t t = time(0);
        if (t - tl >= 10) { fprintf(stderr, "%lld / %lld MB  %.1f MB/s\n", got >> 20, *total >> 20, (got - last) / 1e6 / (t - tl)); tl = t; last = got; }
    }
    fclose(f); close(s);
    return got >= *total ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: tvfetch <host> <port> </path> <out>\n"); return 1; }
    long long total = -1;
    for (int tries = 0; tries < 1000; tries++) {
        int r = attempt(argv[1], argv[2], argv[3], argv[4], &total);
        if (r == 0) { printf("done: %s %lld bytes\n", argv[4], fsize(argv[4])); return 0; }
        if (r == -2) return 2;
        fprintf(stderr, "retry (%s), have %lld\n", strerror(errno), fsize(argv[4]));
        sleep(5);
    }
    return 1;
}
