#!/usr/bin/env python3
"""Per-layer lists of the most used experts, from a GGML_MOE_TRACE log of calibration text (never the eval text).
usage: moe_keep.py <trace> <K> <out>     K experts per layer, most used first. K=N:M also tops layers up by their share of uses."""
import struct, sys, collections
tr = open(sys.argv[1], 'rb').read(); o = 0; K = int(sys.argv[2])
c = collections.defaultdict(collections.Counter)
while o < len(tr):
    il, n, t = struct.unpack_from('<3i', tr, o); o += 12
    ids = struct.unpack_from('<%di' % (n * t), tr, o); o += 4 * n * t
    for x in ids:
        if x >= 0: c[il][x] += 1
with open(sys.argv[3], 'w') as f:
    for il in sorted(c):
        top = c[il].most_common(128)
        # top up with unseen experts so a layer always has K
        have = [e for e, _ in top][:K]
        f.write(str(il) + ' ' + ' '.join(map(str, have)) + '\n')
        s = sum(c[il].values()); print('layer %2d: top %d carry %.1f %% of uses' % (il, K, 100 * sum(v for _, v in top[:K]) / s))
