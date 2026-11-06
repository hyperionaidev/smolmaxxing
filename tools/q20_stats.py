#!/usr/bin/env python3
"""Stats of a Q2_0 (PrismML group 64) GGUF: is it ternary, do adjacent 64-groups share a scale, how many weights per tensor?
Each block_q2_0 is 18 bytes: fp16 scale + 16 bytes of 2-bit codes. usage: q20_stats.py <gguf>"""
import sys, numpy as np
sys.path.insert(0, "third_party/llama.cpp/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1])
tot = same128 = same256 = blocks = 0; rows = []
for t in r.tensors:
    if t.tensor_type.name != "Q2_0" : continue
    n = int(np.prod([int(x) for x in t.shape])); ne0 = int(t.shape[0]); nr = n // ne0
    d = np.array(t.data, copy=False).reshape(-1, 18)
    sc = np.frombuffer(d[:, :2].tobytes(), dtype=np.uint16).reshape(nr, -1)
    g = sc.shape[1]
    s128 = (sc[:, 0::2] == sc[:, 1::2]).sum() if g % 2 == 0 else 0
    s256 = ((sc[:, 0::4] == sc[:, 1::4]) & (sc[:, 0::4] == sc[:, 2::4]) & (sc[:, 0::4] == sc[:, 3::4])).sum() if g % 4 == 0 else 0
    tern = 0
    codes = d[:, 2:]
    bad = sum(int(((codes >> s) & 3 == 3).sum()) for s in (0, 2, 4, 6))
    rows.append((n, t.name, ne0, nr, g, s128 / max(1, sc[:, 0::2].size), s256 / max(1, sc[:, 0::4].size), bad, ne0 % 256))
    tot += n
rows.sort(reverse=True)
print("Q2_0 tensors", len(rows), "weights %.3f B" % (tot / 1e9))
for x in rows[:12]: print("%11d %-28s ne0 %5d rows %6d groups/row %3d same128 %.4f same256 %.4f code3 %d ne0%%256 %d" % x)
small = [x for x in rows if x[8] != 0]
print("not divisible by 256:", len(small), sum(x[0] for x in small))
print("min same128 %.4f, min same256 %.4f, weighted same128 %.4f" % (min(x[5] for x in rows), min(x[6] for x in rows), sum(x[0]*x[5] for x in rows)/tot))
print("total code3", sum(x[7] for x in rows))
acc = 0
for x in rows:
    acc += x[0]
print("biggest 10 share %.3f" % (sum(x[0] for x in rows[:10]) / tot))
