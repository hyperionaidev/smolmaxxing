#!/usr/bin/env python3
"""Replay a GGML_MOE_TRACE routing log through expert caches. usage: moe_sim.py <trace> [MB_per_expert=1.33] [slots...]
Reports the reuse rate between consecutive tokens and the miss rate of LRU and Belady (offline optimal) for a pool of N experts."""
import struct, sys, collections
tr = open(sys.argv[1], 'rb').read(); o = 0
mb = float(sys.argv[2]) if len(sys.argv) > 2 else 1.33
sizes = [int(x) for x in sys.argv[3:]] or [200, 300, 400, 525, 700, 900, 1200]
layers = collections.defaultdict(list)   # il -> list of tokens' id tuples, in execution order
order = []
while o < len(tr):
    il, n, t = struct.unpack_from('<3i', tr, o); o += 12
    ids = struct.unpack_from('<%di' % (n * t), tr, o); o += 4 * n * t
    for i in range(t):
        tok = tuple(sorted(x for x in ids[i * n:i * n + n] if x >= 0))
        layers[il].append(tok); order.append((il, len(layers[il]) - 1))
L = sorted(layers); T = min(len(layers[l]) for l in L)
print("layers %d, tokens %d, experts per token per layer %.2f" % (len(L), T, sum(len(x) for x in layers[L[0]][:T]) / T))
same = sum(len(set(layers[l][t]) & set(layers[l][t - 1])) for l in L for t in range(1, T)); tot = sum(len(layers[l][t]) for l in L for t in range(1, T))
print("reuse from the previous token: %.1f %%" % (100 * same / tot))
uniq = {l: len(set(e for t in layers[l][:T] for e in t)) for l in L}
print("distinct experts used per layer: min %d max %d of 128" % (min(uniq.values()), max(uniq.values())))
# the sequence of accesses per token, layer by layer, as in a real forward pass
seq = [(l, e) for t in range(T) for l in L for e in layers[l][t]]
n_per_tok = len(seq) / T
warm = len(seq) // 5
def lru(C):
    c = collections.OrderedDict(); miss = 0
    for i, k in enumerate(seq):
        if k in c: c.move_to_end(k)
        else:
            if i >= warm: miss += 1
            c[k] = 1
            if len(c) > C: c.popitem(last=False)
    return miss / (len(seq) - warm)
def belady(C):
    nxt = [0] * len(seq); last = {}
    for i in range(len(seq) - 1, -1, -1): nxt[i] = last.get(seq[i], 1 << 60); last[seq[i]] = i
    import heapq
    c = {}; h = []; miss = 0
    for i, k in enumerate(seq):
        if k not in c:
            if i >= warm: miss += 1
            while len(c) >= C:
                n_, kk = heapq.heappop(h)     # farthest next use first
                if c.get(kk) == -n_: del c[kk]
        c[k] = nxt[i]; heapq.heappush(h, (-nxt[i], k))
    return miss / (len(seq) - warm)
print("experts per token %.1f = %.0f MB touched at %.2f MB per expert" % (n_per_tok, n_per_tok * mb, mb))
for C in sizes:
    a, b = lru(C), belady(C)
    print("pool %5d experts (%5.0f MB, %4.1f %% of all): LRU miss %5.1f %% = %5.1f MB/token   Belady %5.1f %% = %5.1f MB/token" % (C, C * mb, 100 * C / (len(L) * 128), 100 * a, a * n_per_tok * mb, 100 * b, b * n_per_tok * mb))

# frequency-aware policies. Calibrate on the first half of the trace, test on the second half (no peeking).
half = T // 2
freq = collections.Counter(k for t in range(half) for l in L for k in ((l, e) for e in layers[l][t]))
test = [(l, e) for t in range(half, T) for l in L for e in layers[l][t]]
n_warm = len(test) // 5
def static_pin(C):
    keep = set(k for k, _ in freq.most_common(C))
    return sum(1 for k in test[n_warm:] if k not in keep) / (len(test) - n_warm)
def pin_plus_lru(C, frac):
    keep = set(k for k, _ in freq.most_common(int(C * frac))); c = collections.OrderedDict(); room = C - len(keep); miss = 0
    for i, k in enumerate(test):
        if k in keep: continue
        if k in c: c.move_to_end(k)
        else:
            if i >= n_warm: miss += 1
            c[k] = 1
            if len(c) > room: c.popitem(last=False)
    return miss / (len(test) - n_warm)
def lru2(C):   # admit to cache only on second touch within a ghost window (scan resistance, what tiny LFU does)
    c = collections.OrderedDict(); ghost = collections.OrderedDict(); miss = 0
    for i, k in enumerate(test):
        if k in c: c.move_to_end(k); continue
        if i >= n_warm: miss += 1
        if k in ghost or len(c) < C:
            ghost.pop(k, None); c[k] = 1
            if len(c) > C: c.popitem(last=False)
        else:
            ghost[k] = 1
            if len(ghost) > C: ghost.popitem(last=False)
    return miss / (len(test) - n_warm)
print("\nsecond half only (calibrated on the first half):")
for C in sizes:
    a, b, c_, d = static_pin(C), pin_plus_lru(C, .5), pin_plus_lru(C, .75), lru2(C)
    print("pool %5d (%5.0f MB): static pin %5.1f %% | pin 50 %%+LRU %5.1f %% | pin 75 %%+LRU %5.1f %% | LRU-2 admit %5.1f %% -> best %5.1f MB/token" % (C, C * mb, 100 * a, 100 * b, 100 * c_, 100 * d, min(a, b, c_, d) * n_per_tok * mb))
