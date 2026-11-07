#!/usr/bin/env python3
"""Per-layer expert pools sized by marginal benefit: LRU miss curve of each layer on a calibration trace, then greedy allocation of a fixed
total. usage: moe_alloc.py <cal trace> <test trace> <total slots> <used> <out>   (out lines: il pool)  Reports misses on the test trace."""
import struct, sys, collections
def load(path):
    tr = open(path, 'rb').read(); o = 0; L = collections.defaultdict(list)
    while o < len(tr):
        il, n, t = struct.unpack_from('<3i', tr, o); o += 12
        ids = struct.unpack_from('<%di' % (n * t), tr, o); o += 4 * n * t
        for i in range(t): L[il].append(ids[i * n:i * n + n])
    return L
def curve(seq, used, maxc=128):
    # stack distances: misses for every LRU size at once
    stack = []; hist = [0] * (maxc + 2); total = 0
    for tok in seq:
        for e in tok[:used]:
            total += 1
            if e in stack:
                d = len(stack) - stack.index(e); stack.remove(e)
                hist[min(d, maxc + 1)] += 1
            else:
                hist[maxc + 1] += 1
            stack.append(e)
    misses = []; acc = sum(hist[1:])
    for c in range(0, maxc + 1):
        if c > 0: acc -= hist[c]
        misses.append(acc)
    return misses, total
cal, test = load(sys.argv[1]), load(sys.argv[2]); tot = int(sys.argv[3]); used = int(sys.argv[4])
layers = sorted(cal)
C = {l: curve(cal[l], used)[0] for l in layers}
alloc = {l: used for l in layers}; left = tot - used * len(layers)
while left > 0:
    best = max(layers, key=lambda l: C[l][alloc[l]] - C[l][min(alloc[l] + 1, 128)])
    alloc[best] += 1; left -= 1
uni = tot // len(layers)
T = {l: curve(test[l], used) for l in layers}
n_tok = len(test[layers[0]])
mu = sum(T[l][0][uni] for l in layers); ma = sum(T[l][0][alloc[l]] for l in layers); all_ = sum(T[l][1] for l in layers)
print("test trace, no bias, top-%d, %d slots: uniform %d per layer: %.2f %% misses (%.2f per token); by benefit: %.2f %% (%.2f per token), %+.1f %%" % (used, tot, uni, 100 * mu / all_, mu / n_tok, 100 * ma / all_, ma / n_tok, 100 * (ma - mu) / mu))
print("alloc:", ' '.join("%d:%d" % (l, alloc[l]) for l in layers))
open(sys.argv[5], 'w').write(''.join("%d %d\n" % (l, alloc[l]) for l in layers))
