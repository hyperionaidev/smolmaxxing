#!/usr/bin/env python3
"""Minimal GGUF header reader: metadata + tensor table, from a local file or the first N MB of a URL.
usage: python3 -I gguf_info.py <file-or-url> [--meta] [--tensors]
Only reads the header; never needs the full file.
"""
import struct, sys, urllib.request, collections

GGML_TYPES = {  # type id: (name, block size, bytes per block)
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}

class R:
    def __init__(self, data): self.d, self.o = data, 0
    def take(self, n):
        if self.o + n > len(self.d): raise EOFError
        b = self.d[self.o:self.o + n]; self.o += n; return b
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")
    def val(self, t):
        f = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in f:
            n = struct.calcsize(f[t]); return struct.unpack(f[t], self.take(n))[0]
        if t == 8: return self.s()
        if t == 9:
            et, n = self.u32(), self.u64()
            return [self.val(et) for _ in range(n)]
        raise ValueError(f"bad type {t}")

def load(src, mb):
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"Range": f"bytes=0-{mb * 1048576 - 1}", "User-Agent": "gguf-info"})
        return urllib.request.urlopen(req).read()
    with open(src, "rb") as f: return f.read(mb * 1048576)

def parse(data):
    r = R(data)
    assert r.take(4) == b"GGUF", "not a GGUF file"
    ver, nt, nkv = r.u32(), r.u64(), r.u64()
    meta = {}
    for _ in range(nkv):
        k = r.s(); t = r.u32(); meta[k] = r.val(t)
    tensors = []
    for _ in range(nt):
        name = r.s(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; t = r.u32(); off = r.u64()
        n = 1
        for d in dims: n *= d
        tn, bs, bb = GGML_TYPES.get(t, (f"type{t}", 1, 0))
        tensors.append((name, dims, tn, n, n // bs * bb, off))
    return ver, meta, tensors

if __name__ == "__main__":
    src = sys.argv[1]
    for mb in (8, 32, 64):
        try:
            ver, meta, tensors = parse(load(src, mb)); break
        except EOFError:
            continue
    if "--meta" in sys.argv:
        for k, v in meta.items():
            if isinstance(v, list): v = f"[{len(v)} items] {v[:3]}"
            print(f"{k} = {v}")
    if "--tensors" in sys.argv:
        for name, dims, tn, n, nb, off in tensors:
            print(f"{nb/1e6:9.2f} MB  {tn:7s} {str(dims):22s} {name}")
    # summary
    by_type, by_group = collections.Counter(), collections.Counter()
    total = 0
    for name, dims, tn, n, nb, off in tensors:
        by_type[tn] += nb; total += nb
        grp = name.split(".")[-2] if name.startswith("blk.") else name.rsplit(".", 1)[0]
        by_group[grp] += nb
    nparams = sum(t[3] for t in tensors)
    print(f"GGUF v{ver}: {len(tensors)} tensors, {nparams/1e9:.3f} B params, {total/1e6:.1f} MB of weights")
    print("by type:  " + ", ".join(f"{k} {v/1e6:.0f} MB" for k, v in by_type.most_common()))
    print("by group: " + ", ".join(f"{k} {v/1e6:.0f} MB" for k, v in by_group.most_common(14)))
