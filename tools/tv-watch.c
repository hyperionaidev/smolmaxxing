// tv-watch: memory guard + box stats for the TV, in one small static binary that never forks. The shell versions (tv-guard.sh,
// boxstats.sh) start about 25 processes a second, each able to preempt a compute thread and stall a barrier: boxstats alone cost
// 10 % of Bonsai-8B's speed (4.67 vs 5.16 tok/s, EXPERIMENT.md section 19).
// usage: tv-watch <log> <command...>     command's output goes to <log>, samples to <log>.mem (time, MemAvailable MB, SwapFree MB,
// RSS MB), same format as tv-guard.sh. Environment: LIMIT_MB (default 180): kill the command below this MemAvailable.
// GUARD_INTERVAL seconds (default 1), BOX_JSON=<file> also writes the web page's stats (same fields as boxstats.sh).
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
static int rd(const char * p, char * b, int n) { int f = open(p, O_RDONLY | O_CLOEXEC); if (f < 0) return 0; int r = read(f, b, n - 1); close(f); if (r < 0) r = 0; b[r] = 0; return r; }
static long field(const char * b, const char * k) { const char * s = strstr(b, k); if (!s) return 0; s += strlen(k); while (*s == ' ' || *s == ':' || *s == '\t') s++; return atol(s); }
int main(int c, char ** v) {
    if (c < 3) { fprintf(stderr, "usage: tv-watch <log> <command...>\n"); return 2; }
    long limit = getenv("LIMIT_MB") ? atol(getenv("LIMIT_MB")) : 180;
    double iv = getenv("GUARD_INTERVAL") ? atof(getenv("GUARD_INTERVAL")) : 1.0;
    const char * box = getenv("BOX_JSON");
    int lf = open(v[1], O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (lf < 0) { perror(v[1]); return 2; }
    char mp[512], bt[512]; snprintf(mp, sizeof mp, "%s.mem", v[1]);
    FILE * mf = fopen(mp, "w");
    if (box) snprintf(bt, sizeof bt, "%s.tmp", box);
    pid_t p = fork();
    if (p == 0) { dup2(lf, 1); dup2(lf, 2); execvp(v[2], v + 2); perror("exec"); _exit(127); }
    char st[64], b[8192], o[512]; snprintf(st, sizeof st, "/proc/%d/status", p);
    long minav = 1L << 40, maxrss = 0, prevpg = -1; int status = 0, killed = 0;
    for (;;) {
        pid_t w = waitpid(p, &status, WNOHANG);
        if (w == p) break;
        rd("/proc/meminfo", b, sizeof b);
        long av = field(b, "MemAvailable") / 1024, sw = field(b, "SwapFree") / 1024, tot = field(b, "MemTotal") / 1024;
        rd(st, b, sizeof b);
        long rss = field(b, "VmRSS") / 1024;
        if (av < minav) minav = av;
        if (rss > maxrss) maxrss = rss;
        time_t t = time(NULL); struct tm tm; localtime_r(&t, &tm);
        if (mf) { fprintf(mf, "%02d:%02d:%02d %ld %ld %ld\n", tm.tm_hour, tm.tm_min, tm.tm_sec, av, sw, rss); fflush(mf); }
        if (box) {
            rd("/proc/vmstat", b, sizeof b); long pg = field(b, "\npgpgin ");
            long fl = prevpg < 0 ? 0 : (long) ((pg - prevpg) / 1024 / iv); prevpg = pg;
            long f0 = 0, f3 = 0; if (rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", b, 64)) f0 = atol(b) / 1000; if (rd("/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq", b, 64)) f3 = atol(b) / 1000;
            char load[32] = "0"; if (rd("/proc/loadavg", b, 64)) { sscanf(b, "%31s", load); }
            int n = snprintf(o, sizeof o, "{\"t\":%ld,\"memtotal_mb\":%ld,\"memavail_mb\":%ld,\"swapfree_mb\":%ld,\"server_rss_mb\":%ld,\"flash_read_mb_s\":%ld,\"cpu0_mhz\":%ld,\"cpu3_mhz\":%ld,\"load1\":%s}\n", (long) t, tot, av, sw, rss, fl, f0, f3, load);
            int bf = open(bt, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
            if (bf >= 0) { if (write(bf, o, n) == n) { close(bf); rename(bt, box); } else close(bf); }
        }
        if (av < limit) {
            kill(p, SIGKILL); killed = 1;
            dprintf(lf, "tv-guard: killed, MemAvailable %ld MB < %ld MB\n", av, limit);
            waitpid(p, &status, 0);
            break;
        }
        struct timespec ts = { (time_t) iv, (long) ((iv - (time_t) iv) * 1e9) };
        nanosleep(&ts, NULL);
    }
    int rc = killed ? 137 : WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    dprintf(lf, "tv-guard: exit %d, min MemAvailable %ld MB, max RSS %ld MB\n", rc, minav, maxrss);
    if (mf) fclose(mf);
    fsync(lf); close(lf);
    return rc;
}
