// Tiny LLM web server for the Xiaomi TV box.
// Loads a llama2.c model once, serves index.html, and streams generations over HTTP.
//   usage: llm-server <model.bin> <tokenizer.bin> <index.html> [port=8080]
// Routes: GET /         -> index.html (re-read on every request)
//         GET /info     -> model config as JSON
//         GET /gen?prompt=..&t=0.8&delay=35&n=512&seed=..
//              -> text stream: each token piece followed by 0x1d (prompt) or 0x1f (generated),
//                 then 0x1e + stats JSON at the end.
// Single-threaded: one request at a time. A new /gen from the page aborts the old one,
// the old socket's send fails, and generation stops.
#define TESTING
#include "run.c"   // llama2.c inference code (karpathy/llama2.c @ 350e04f), main() excluded by TESTING
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <errno.h>
#include <sys/time.h>

static Transformer T;
static Tokenizer TK;
static const char *html_path;
static long long n_params;

static long long now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000LL + t.tv_nsec / 1000;
}

static int send_all(int fd, const char *b, size_t n) {
    while (n) {
        ssize_t w = send(fd, b, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        b += w; n -= (size_t)w;
    }
    return 0;
}
static int send_str(int fd, const char *s) { return send_all(fd, s, strlen(s)); }

static void send_simple(int fd, const char *status, const char *ctype, const char *body, size_t len) {
    char h[256];
    snprintf(h, sizeof h, "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
             "Cache-Control: no-store\r\nConnection: close\r\n\r\n", status, ctype, len);
    if (send_str(fd, h) == 0 && len) send_all(fd, body, len);
}

static int hexv(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Finds key=value in a query string and URL-decodes the value into out. Returns 1 if found.
static int qparam(const char *q, const char *key, char *out, size_t outsz) {
    size_t kl = strlen(key);
    for (const char *p = q; p && *p; p = strchr(p, '&') ? strchr(p, '&') + 1 : NULL) {
        if (strncmp(p, key, kl) != 0 || p[kl] != '=') continue;
        p += kl + 1;
        size_t o = 0;
        while (*p && *p != '&' && o + 1 < outsz) {
            if (*p == '+') { out[o++] = ' '; p++; }
            else if (*p == '%' && hexv(p[1]) >= 0 && hexv(p[2]) >= 0) {
                out[o++] = (char)(hexv(p[1]) * 16 + hexv(p[2])); p += 3;
            } else out[o++] = *p++;
        }
        out[o] = '\0';
        return 1;
    }
    return 0;
}

static void handle_index(int fd) {
    FILE *f = fopen(html_path, "rb");
    if (!f) { send_simple(fd, "500 Internal Server Error", "text/plain", "index.html missing\n", 19); return; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(n > 0 ? n : 1);
    size_t got = fread(b, 1, n, f); fclose(f);
    send_simple(fd, "200 OK", "text/html; charset=utf-8", b, got);
    free(b);
}

static void handle_info(int fd) {
    Config *c = &T.config;
    char j[320];
    int n = snprintf(j, sizeof j,
        "{\"dim\":%d,\"hidden_dim\":%d,\"n_layers\":%d,\"n_heads\":%d,\"n_kv_heads\":%d,"
        "\"vocab_size\":%d,\"seq_len\":%d,\"params\":%lld,\"file_bytes\":%lld}",
        c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads, c->vocab_size,
        c->seq_len, n_params, (long long)T.file_size);
    send_simple(fd, "200 OK", "application/json", j, (size_t)n);
}

static void handle_gen(int fd, const char *q) {
    char prompt[257] = "", buf[32];
    qparam(q, "prompt", prompt, sizeof prompt);
    for (char *c = prompt; *c; c++) if ((unsigned char)*c < 32) *c = ' ';

    float temp = qparam(q, "t", buf, sizeof buf) ? (float)atof(buf) : 0.8f;
    if (!(temp >= 0.0f)) temp = 0.0f;
    if (temp > 2.0f) temp = 2.0f;
    int steps = qparam(q, "n", buf, sizeof buf) ? atoi(buf) : T.config.seq_len;
    if (steps < 1 || steps > T.config.seq_len) steps = T.config.seq_len;
    int delay = qparam(q, "delay", buf, sizeof buf) ? atoi(buf) : 0;
    if (delay < 0) delay = 0;
    if (delay > 500) delay = 500;
    unsigned long long seed = qparam(q, "seed", buf, sizeof buf) ? strtoull(buf, NULL, 10) : 0;
    if (seed == 0) seed = (unsigned long long)now_us() ^ (unsigned long long)time(NULL);
    if (seed == 0) seed = 1;

    Sampler S;
    build_sampler(&S, T.config.vocab_size, temp, 0.9f, seed);
    int *toks = malloc((strlen(prompt) + 3) * sizeof(int));
    int np = 0;
    encode(&TK, prompt, 1, 0, toks, &np);

    int ok = send_str(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n"
                          "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
                          "Connection: close\r\n\r\n") == 0;
    long long compute_us = 0;
    int pos = 0, gen = 0, token = toks[0], next;
    const char *stop = "length";
    while (ok && pos < steps) {
        long long t0 = now_us();
        float *logits = forward(&T, token, pos);
        int forced = pos < np - 1;
        next = forced ? toks[pos + 1] : sample(&S, logits);
        compute_us += now_us() - t0;
        pos++;
        if (next == 1) { stop = "end"; break; }  // BOS = the model ended its story
        char *piece = decode(&TK, token, next);
        // same filter as run.c's safe_printf: drop lone non-printable bytes
        int printable = piece && piece[0] &&
            (piece[1] || isprint((unsigned char)piece[0]) || isspace((unsigned char)piece[0]));
        if (printable && send_str(fd, piece) != 0) { ok = 0; break; }
        if (send_all(fd, forced ? "\x1d" : "\x1f", 1) != 0) { ok = 0; break; }
        if (!forced) gen++;
        token = next;
        if (!forced && delay) usleep((useconds_t)delay * 1000);
    }
    if (ok) {
        char j[256];
        double secs = compute_us / 1e6;
        snprintf(j, sizeof j, "\x1e{\"tokens\":%d,\"prompt_tokens\":%d,\"forward_passes\":%d,"
                 "\"compute_ms\":%.1f,\"tok_s\":%.1f,\"seed\":\"%llu\",\"t\":%.2f,\"stop\":\"%s\"}",
                 gen, np - 1, pos, compute_us / 1000.0, secs > 0 ? pos / secs : 0.0, seed, temp, stop);
        send_str(fd, j);
    }
    fprintf(stderr, "gen prompt=\"%s\" t=%.2f seed=%llu tokens=%d stop=%s%s\n",
            prompt, temp, seed, gen, stop, ok ? "" : " (client left)");
    free(toks);
    free_sampler(&S);
}

static void handle(int fd) {
    char req[4096];
    size_t n = 0;
    while (n < sizeof req - 1) {
        ssize_t r = recv(fd, req + n, sizeof req - 1 - n, 0);
        if (r <= 0) break;
        n += (size_t)r; req[n] = '\0';
        if (strstr(req, "\r\n\r\n")) break;
    }
    req[n] = '\0';
    if (strncmp(req, "GET ", 4) != 0) { send_simple(fd, "405 Method Not Allowed", "text/plain", "GET only\n", 9); return; }
    char *path = req + 4, *end = strchr(path, ' ');
    if (!end) return;
    *end = '\0';
    char *q = strchr(path, '?');
    if (q) *q++ = '\0'; else q = "";

    if (!strcmp(path, "/") || !strcmp(path, "/index.html")) handle_index(fd);
    else if (!strcmp(path, "/info")) handle_info(fd);
    else if (!strcmp(path, "/gen")) handle_gen(fd, q);
    else send_simple(fd, "404 Not Found", "text/plain", "not found\n", 10);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin> <index.html> [port]\n", argv[0]); return 1; }
    int port = argc > 4 ? atoi(argv[4]) : 8080;
    html_path = argv[3];
    signal(SIGPIPE, SIG_IGN);

    build_transformer(&T, argv[1]);
    build_tokenizer(&TK, argv[2], T.config.vocab_size);
    Config *c = &T.config;
    long long kv = (long long)c->dim * c->n_kv_heads / c->n_heads;
    n_params = (long long)c->vocab_size * c->dim
             + (long long)c->n_layers * (2LL * c->dim + 2LL * c->dim * c->dim + 2LL * c->dim * kv
                                         + 3LL * c->dim * c->hidden_dim)
             + c->dim;  // assumes tied output head, true for stories260K

    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port), .sin_addr.s_addr = htonl(INADDR_ANY) };
    if (bind(s, (struct sockaddr *)&a, sizeof a) < 0 || listen(s, 8) < 0) { perror("bind/listen"); return 1; }
    fprintf(stderr, "tiny-llm: %lld params, listening on :%d\n", n_params, port);

    for (;;) {
        int fd = accept(s, NULL, NULL);
        if (fd < 0) continue;
        struct timeval tv = { 5, 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        tv.tv_sec = 10;
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);  // flush each token right away
        handle(fd);
        close(fd);
    }
}
