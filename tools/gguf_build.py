#!/usr/bin/env python3
"""Streaming GGUF rewriter: copy a GGUF and add tensors / metadata in one sequential pass.

Built so a 3.8 GB BF16 model can go TV -> Mac -> TV through pipes without ever being stored on the
Mac (which has ~4 GB free):

  adb exec-out cat in.gguf | python3 -I gguf_build.py - - [options] | adb exec-in 'cat > out.gguf'

Options:
  --head-ids FILE   add a pruned LM head: output.weight = the token_embd rows listed in FILE (one id per
                    line, raw bytes, same type as token_embd) + output_ids.weight (I32). Needs the llama.cpp
                    pruned-head patch (patches/llama.cpp-v0.8.0-tvbox.patch). For a model with an untied
                    output.weight (e.g. Qwen3-8B / Bonsai-8B) the rows are cut from that tensor instead.
  --replace-head    drop an existing pruned head (output.weight + output_ids.weight) first; with --head-ids
                    this re-cuts the head from token_embd (for a row-wise quant like Q4_K the new rows are
                    byte-identical to quantizing those rows from BF16)
  --add DIR         append the tensors listed in DIR/manifest.json (written by fetch_mtp.py)
  --set KEY=TYPE:V  set / add a metadata key; TYPE is u32, i32, f32, bool or str (repeatable)

The input is read strictly in order (header, then tensor data in offset order), so it can be a pipe.
New tensors go after the existing ones; the existing tensors keep their order and alignment.
"""
import json, os, struct, sys

GGUF_U32, GGUF_I32, GGUF_F32, GGUF_BOOL, GGUF_STR, GGUF_ARR = 4, 5, 6, 7, 8, 9
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 20: (32, 18),
         23: (256, 136), 26: (1, 4), 30: (1, 2), 34: (256, 54), 35: (256, 66), 41: (128, 18), 42: (64, 18),
         16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 21: (256, 110), 22: (256, 82), 29: (256, 56)}
I32 = 26


class Stream:
    """Sequential reader over a file or pipe that counts the bytes consumed."""
    def __init__(self, f): self.f, self.pos = f, 0
    def read(self, n):
        out = bytearray()
        while len(out) < n:
            b = self.f.read(n - len(out))
            if not b: raise EOFError(f"input ended at byte {self.pos + len(out)}")
            out += b
        self.pos += n
        return bytes(out)
    def u32(self): return struct.unpack("<I", self.read(4))[0]
    def u64(self): return struct.unpack("<Q", self.read(8))[0]
    def s(self): return self.read(self.u64()).decode("utf-8")
    def skip_to(self, pos):
        while self.pos < pos: self.read(min(pos - self.pos, 1 << 20))


def read_value_raw(r, t):
    """Returns (python value or None for big arrays, raw bytes)."""
    if t in SCALAR:
        b = r.read(SCALAR[t]); return b, b
    if t == GGUF_STR:
        n = r.read(8); b = r.read(struct.unpack("<Q", n)[0]); return b.decode("utf-8", "replace"), n + b
    if t == GGUF_ARR:
        hdr = r.read(12); et, n = struct.unpack("<IQ", hdr)
        parts, vals = [hdr], []
        for _ in range(n):
            v, raw = read_value_raw(r, et); parts.append(raw); vals.append(v)
        return vals, b"".join(parts)
    raise ValueError(f"unknown gguf type {t}")


def pack_str(s):
    b = s.encode(); return struct.pack("<Q", len(b)) + b


def pack_value(typ, v):
    if typ == "u32": return GGUF_U32, struct.pack("<I", int(v))
    if typ == "i32": return GGUF_I32, struct.pack("<i", int(v))
    if typ == "f32": return GGUF_F32, struct.pack("<f", float(v))
    if typ == "bool": return GGUF_BOOL, struct.pack("<?", v in ("1", "true", "True"))
    if typ == "str": return GGUF_STR, pack_str(v)
    raise ValueError(typ)


def nbytes(dims, t):
    n = 1
    for d in dims: n *= d
    bs, bb = BLOCK[t]
    return n // bs * bb


def main():
    a = sys.argv[1:]
    src, dst = a[0], a[1]
    head_ids, add_dir, sets, replace = None, None, [], False
    i = 2
    while i < len(a):
        if a[i] == "--head-ids": head_ids = a[i + 1]; i += 2
        elif a[i] == "--add": add_dir = a[i + 1]; i += 2
        elif a[i] == "--set": sets.append(a[i + 1]); i += 2
        elif a[i] == "--replace-head": replace = True; i += 1
        else: sys.exit(f"unknown option {a[i]}")
    log = lambda *x: print(*x, file=sys.stderr, flush=True)

    fi = sys.stdin.buffer if src == "-" else open(src, "rb")
    r = Stream(fi)
    assert r.read(4) == b"GGUF"
    ver, n_t, n_kv = r.u32(), r.u64(), r.u64()
    kvs = []  # [key, type, raw]
    meta = {}
    for _ in range(n_kv):
        k = r.s(); t = r.u32(); v, raw = read_value_raw(r, t)
        kvs.append([k, t, raw])
        if t != GGUF_ARR or k == "tokenizer.ggml.token_type": meta[k] = v
    tensors = []
    for _ in range(n_t):
        name = r.s(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; t = r.u32(); off = r.u64()
        tensors.append({"name": name, "dims": dims, "type": t, "off": off, "size": nbytes(dims, t)})
    align = struct.unpack("<I", meta["general.alignment"])[0] if "general.alignment" in meta else 32
    data_start = (r.pos + align - 1) // align * align
    arch = meta["general.architecture"].decode() if isinstance(meta["general.architecture"], bytes) else meta["general.architecture"]

    # metadata edits
    for sv in sets:
        key, rest = sv.split("=", 1); typ, val = rest.split(":", 1)
        key = key.replace("{arch}", arch)
        t, raw = pack_value(typ, val)
        for kv in kvs:
            if kv[0] == key: kv[1], kv[2] = t, raw; break
        else: kvs.append([key, t, raw])
        log(f"set {key} = {val} ({typ})")

    dropped = []
    if replace:
        dropped = [t for t in tensors if t["name"] in ("output.weight", "output_ids.weight")]
        tensors = [t for t in tensors if t["name"] not in ("output.weight", "output_ids.weight")]
        kvs = [kv for kv in kvs if not kv[0].startswith("tvbox.pruned_head")]
        log(f"dropping old head: {[t['name'] for t in dropped]}")
    new = []
    names = [t["name"] for t in tensors]
    head_src, untied = "token_embd.weight", None
    if head_ids:
        if "output.weight" in names:   # untied head: cut the rows from output.weight, which is dropped
            head_src = "output.weight"
            untied = tensors.pop(names.index("output.weight")); names.remove("output.weight")
            log("untied output.weight: the pruned head is cut from it")
        emb = untied or tensors[names.index("token_embd.weight")]
        n_embd, n_vocab = emb["dims"]
        row = nbytes([n_embd], emb["type"])
        keep = sorted({int(x) for x in open(head_ids).read().split()})
        assert keep[-1] < n_vocab
        new.append({"name": "output.weight", "dims": [n_embd, len(keep)], "type": emb["type"], "size": row * len(keep), "src": "head"})
        new.append({"name": "output_ids.weight", "dims": [len(keep)], "type": I32, "size": 4 * len(keep),
                    "data": struct.pack(f"<{len(keep)}i", *keep)})
        t, raw = pack_value("u32", len(keep))
        kvs.append(["tvbox.pruned_head.n_keep", t, raw])
        log(f"pruned head: {len(keep)} of {n_vocab} rows, {row * len(keep) / 1e6:.1f} MB ({emb['type']=})")
    if add_dir:
        for m in json.load(open(os.path.join(add_dir, "manifest.json"))):
            assert m["name"] not in names, m["name"]
            new.append({"name": m["name"], "dims": m["dims"], "type": m["type"], "size": m["bytes"],
                        "file": os.path.join(add_dir, m["file"])})
            assert nbytes(m["dims"], m["type"]) == m["bytes"], m["name"]
        log(f"adding {len(new) - (2 if head_ids else 0)} tensors from {add_dir}")

    # new layout: existing tensors in their original data order, then the new ones
    order = sorted(tensors, key=lambda t: t["off"])
    off = 0
    for t in order + new:
        t["new_off"] = off
        off += (t["size"] + align - 1) // align * align
    total = off

    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", ver, len(tensors) + len(new), len(kvs)))
    for k, t, raw in kvs:
        hdr += pack_str(k) + struct.pack("<I", t) + raw
    for t in tensors + new:
        hdr += pack_str(t["name"]) + struct.pack("<I", len(t["dims"])) + b"".join(struct.pack("<Q", d) for d in t["dims"])
        hdr += struct.pack("<IQ", t["type"], t["new_off"])
    hdr += b"\0" * ((-len(hdr)) % align)

    fo = sys.stdout.buffer if dst == "-" else open(dst, "wb")
    fo.write(hdr)
    written = 0
    head_rows = {}
    keep_set = set(keep) if head_ids else set()
    for t in sorted(order + ([untied] if untied else []), key=lambda t: t["off"]):
        r.skip_to(data_start + t["off"])
        if t is not untied:
            fo.write(b"\0" * (t["new_off"] - written)); written = t["new_off"]
        left = t["size"]
        if head_ids and t["name"] == head_src:
            idx = 0
            while left:
                n = min(left, row * 4096)
                chunk = r.read(n)
                if t is not untied: fo.write(chunk)
                first = idx
                for j in range(n // row):
                    if first + j in keep_set: head_rows[first + j] = chunk[j * row:(j + 1) * row]
                idx += n // row; left -= n
        else:
            while left:
                n = min(left, 8 << 20); fo.write(r.read(n)); left -= n
        if t is not untied: written += t["size"]
        log(f"  {t['name']:40s} {t['size'] / 1e6:9.2f} MB") if t["size"] > 50e6 else None
    for t in new:
        fo.write(b"\0" * (t["new_off"] - written)); written = t["new_off"]
        if t.get("src") == "head":
            data = b"".join(head_rows[k] for k in keep)
        elif "data" in t:
            data = t["data"]
        else:
            data = open(t["file"], "rb").read()
        assert len(data) == t["size"], t["name"]
        fo.write(data); written += len(data)
    fo.write(b"\0" * (total - written))
    fo.flush()
    log(f"wrote {len(hdr) + total:,} bytes ({len(tensors)} + {len(new)} tensors, {len(kvs)} keys)")


if __name__ == "__main__":
    main()
