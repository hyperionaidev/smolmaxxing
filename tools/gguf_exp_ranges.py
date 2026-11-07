#!/usr/bin/env python3
"""Byte ranges of the *_exps tensors of a GGUF, one "offset length" per line, for tools/gguf-punch.c. usage: gguf_exp_ranges.py <gguf> <out>"""
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_info.py")); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
r = g.R(open(sys.argv[1], 'rb').read(64 << 20))
assert r.take(4) == b"GGUF"
ver, nt, nkv = r.u32(), r.u64(), r.u64()
meta = {}
for _ in range(nkv):
    k = r.s(); t = r.u32(); meta[k] = r.val(t)
rows = []
for _ in range(nt):
    name = r.s(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; t = r.u32(); off = r.u64()
    n = 1
    for d in dims: n *= d
    tn, bs, bb = g.GGML_TYPES.get(t, (f"type{t}", 1, 0))
    rows.append((name, n // bs * bb, off))
al = meta.get("general.alignment", 32)
start = (r.o + al - 1) // al * al
tot = 0
with open(sys.argv[2], 'w') as f:
    for name, nb, off in rows:
        if "_exps.weight" in name:
            f.write("%d %d\n" % (start + off, nb)); tot += nb
print("data starts at %d, %d expert bytes" % (start, tot))
