#!/usr/bin/env python3
"""Add a pruned LM head to a tied-embedding GGUF (Qwen3.5 etc.).

Writes a copy of <in.gguf> with two extra tensors:
  output.weight      [n_embd, n_keep]  rows of token_embd for the kept vocab ids (raw bytes, same quant type)
  output_ids.weight  [n_keep] I32      the vocab id of each kept row
token_embd stays complete, so any token can still be read as input. Only the output matmul shrinks.
Needs the matching llama.cpp patch (LLM_TENSOR_OUTPUT_IDS) to be used; stock llama.cpp rejects the shape.

Kept ids = every id < K (byte-level BPE ranks frequent tokens first; ids 0..255 are raw bytes, so any
text stays writable) + every control / user-defined token (type 3/4), minus unused [PAD] (type 5).

usage: python3 -I gguf_prune_head.py in.gguf out.gguf K
"""
import os, struct, sys

ALIGN_KEY = "general.alignment"
TYPE_SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
              10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 26: (1, 4), 30: (1, 2), 34: (256, 54), 35: (256, 66)}
GGML_TYPE_I32 = 26
GGUF_TYPE_UINT32, GGUF_TYPE_STRING, GGUF_TYPE_ARRAY = 4, 8, 9


class Reader:
    def __init__(self, f): self.f = f
    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n: raise EOFError
        return b
    def u32(self): return struct.unpack("<I", self.raw(4))[0]
    def u64(self): return struct.unpack("<Q", self.raw(8))[0]
    def s(self): return self.raw(self.u64()).decode("utf-8", "replace")
    def val(self, t):
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fmt: return struct.unpack(fmt[t], self.raw(struct.calcsize(fmt[t])))[0]
        if t == GGUF_TYPE_STRING: return self.s()
        if t == GGUF_TYPE_ARRAY:
            et, n = self.u32(), self.u64()
            return [self.val(et) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {t}")


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def main():
    src, dst, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(src, "rb") as f:
        r = Reader(f)
        assert r.raw(4) == b"GGUF"
        ver, n_t, n_kv = r.u32(), r.u64(), r.u64()
        kv_start = f.tell()
        meta = {}
        for _ in range(n_kv):
            k = r.s(); t = r.u32(); meta[k] = r.val(t)
        kv_end = f.tell()
        f.seek(kv_start); kv_raw = f.read(kv_end - kv_start)
        tensors = []
        for _ in range(n_t):
            name = r.s(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; t = r.u32(); off = r.u64()
            tensors.append([name, dims, t, off])
        align = meta.get(ALIGN_KEY, 32)
        data_start = (f.tell() + align - 1) // align * align

    names = [t[0] for t in tensors]
    assert "output.weight" not in names, "model already has an output tensor (not tied)"
    emb = tensors[names.index("token_embd.weight")]
    n_embd, n_vocab = emb[1]
    bs, bb = TYPE_SIZES[emb[2]]
    assert n_embd % bs == 0
    row_bytes = n_embd // bs * bb

    types = meta["tokenizer.ggml.token_type"]
    keep = [i for i in range(n_vocab) if i < K or types[i] in (3, 4)]
    print(f"vocab {n_vocab}, keeping {len(keep)} rows ({len(keep) / n_vocab:.1%}), "
          f"head {len(keep) * row_bytes / 1e6:.1f} MB instead of {n_vocab * row_bytes / 1e6:.1f} MB")

    # sizes of existing tensors from offset deltas (data is laid out in table order, aligned)
    order = sorted(tensors, key=lambda t: t[3])
    total = os.path.getsize(src) - data_start
    for i, t in enumerate(order):
        nxt = order[i + 1][3] if i + 1 < len(order) else total
        t.append(nxt - t[3])  # padded size

    new = [["output.weight", [n_embd, len(keep)], emb[2], None, len(keep) * row_bytes],
           ["output_ids.weight", [len(keep)], GGML_TYPE_I32, None, len(keep) * 4]]
    all_t = tensors + new
    off = 0
    for t in all_t:
        t[3] = off
        off += (t[4] + align - 1) // align * align

    with open(src, "rb") as fi, open(dst, "wb") as fo:
        hdr = bytearray(b"GGUF" + struct.pack("<IQQ", ver, len(all_t), n_kv + 1))
        hdr += kv_raw
        hdr += gguf_str("tvbox.pruned_head.keep_below") + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", K)
        for name, dims, t, o, _ in all_t:
            hdr += gguf_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            hdr += struct.pack("<IQ", t, o)
        hdr += b"\0" * ((-len(hdr)) % align)
        fo.write(hdr)
        base = fo.tell()
        # original offsets (tensors[] now holds the new layout), re-read from the source header
        orig = {}
        with open(src, "rb") as f2:
            r2 = Reader(f2)
            r2.raw(4); r2.u32(); nt2 = r2.u64(); nkv2 = r2.u64()
            for _ in range(nkv2):
                r2.s(); r2.val(r2.u32())
            for _ in range(nt2):
                nm = r2.s(); nd = r2.u32(); [r2.u64() for _ in range(nd)]; r2.u32(); orig[nm] = r2.u64()
        for name, dims, t, o, size in tensors:
            fo.seek(base + o)
            fi.seek(data_start + orig[name])
            left = size
            while left:
                chunk = fi.read(min(left, 64 << 20)); fo.write(chunk); left -= len(chunk)
        # pruned head: gather rows of token_embd
        fo.seek(base + new[0][3])
        emb_off = data_start + orig["token_embd.weight"]
        for i in keep:
            fi.seek(emb_off + i * row_bytes)
            fo.write(fi.read(row_bytes))
        fo.seek(base + new[1][3])
        fo.write(struct.pack(f"<{len(keep)}i", *keep))
        end = base + off
        fo.truncate(end)
    print(f"wrote {dst}: {end / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
