#!/usr/bin/env python3
"""Assemble an experts-last MoE GGUF for the iPad's expert pager from two sources, writing strictly sequentially so the output can
be piped straight to the device (nothing big is stored on the Mac):
  - base.gguf: a local GGUF with every non-expert tensor and the metadata (gguf_fetch_subset.py -> llama-quantize -> gguf_build.py)
  - URL: a remote GGUF whose *_exps.* tensors are taken as they are, fetched with parallel HTTP range requests
With --q2lut the Q2_K expert tensors are repacked for the lookup-table kernel on the fly (same layout as gguf_q2lut.py, which also
gets the key ipad.expert_layout = "q2lut"); other expert types are copied unchanged.
usage: gguf_moe_assemble.py base.gguf URL out.gguf|- [--q2lut] [--jobs N]"""
import argparse, concurrent.futures as cf, hashlib, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_remote import Header, remote_header, fetch, info_bytes, nbytes, kv_str

ap = argparse.ArgumentParser()
ap.add_argument('base'); ap.add_argument('url'); ap.add_argument('out')
ap.add_argument('--q2lut', action='store_true'); ap.add_argument('--jobs', type=int, default=8)
ap.add_argument('--chunk', type=int, default=16 << 20)
a = ap.parse_args()
Q2_K = 10


def repack(raw, ne):   # identical to tools/gguf_q2lut.py
    ne00, ne01, n_exp = ne
    nsb, G = ne00 // 256, ne01 // 16
    blk = raw.reshape(n_exp, G, 16, nsb, 84)
    scales, qs = blk[..., 0:16], blk[..., 16:80]
    d, dmin = blk[..., 80:82], blk[..., 82:84]
    q = qs.reshape(n_exp, G, 16, nsb, 2, 1, 32)
    codes = (q >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 1, 1, 4, 1)) & 3
    codes = codes.reshape(n_exp, G, 16, nsb, 16, 8, 2)
    nib = codes[..., 0] | (codes[..., 1] << 2)
    nib = nib.reshape(n_exp, G, 16, nsb, 16, 4, 2)
    byte = nib[..., 0] | (nib[..., 1] << 4)
    out = np.empty((n_exp, G, nsb, 1344), dtype=np.uint8)
    out[..., 0:32] = d.transpose(0, 1, 3, 2, 4).reshape(n_exp, G, nsb, 32)
    out[..., 32:64] = dmin.transpose(0, 1, 3, 2, 4).reshape(n_exp, G, nsb, 32)
    out[..., 64:320] = scales.transpose(0, 1, 3, 4, 2).reshape(n_exp, G, nsb, 256)
    out[..., 320:1344] = byte.transpose(0, 1, 3, 4, 5, 2).reshape(n_exp, G, nsb, 1024)
    return out.reshape(-1)


bm = np.memmap(a.base, dtype=np.uint8, mode='r')
base = Header(bm, len(bm))
assert not any('_exps.' in x[0] for x in base.infos), 'base already has expert tensors'
rh, rsize = remote_header(a.url)
exps = [i for i in rh.order if '_exps.' in rh.infos[i][0]]
for k in ('general.architecture',):
    assert base.keys.get(k) == rh.keys.get(k), f'{k} differs between base and expert source'

align = base.align
tensors = [('b', i, base.infos[i]) for i in base.order] + [('r', i, rh.infos[i]) for i in exps]
lay, cur = [], 0
for src, i, (name, ne, ty, off, psz) in tensors:
    lay.append(cur); cur = (cur + nbytes(ne, ty) + align - 1) // align * align
total_data = cur

extra = kv_str('ipad.expert_layout', 'q2lut') if a.q2lut else b''
hdr = bytearray(struct.pack('<4sIQQ', b'GGUF', base.version, len(tensors), base.n_kv + (1 if a.q2lut else 0)))
hdr += base.kv_bytes() + extra
for (src, i, (name, ne, ty, off, psz)), o in zip(tensors, lay):
    hdr += info_bytes(name, ne, ty, o)
hdr += b'\0' * ((len(hdr) + align - 1) // align * align - len(hdr))

out = sys.stdout.buffer if a.out == '-' else open(a.out, 'wb')
md5 = hashlib.md5(); written = 0
def w(buf):
    global written
    out.write(buf); md5.update(buf); written += len(buf)
def pad_to(o):
    if written - len(hdr) < o: w(b'\0' * (o - (written - len(hdr))))

w(bytes(hdr))
for (src, i, (name, ne, ty, off, psz)), o in zip(tensors, lay):
    if src != 'b': break
    pad_to(o); w(bm[base.data0 + off: base.data0 + off + nbytes(ne, ty)].tobytes())

# experts: chunks of every tensor fetched by a pool, consumed in order
jobs = []
for k, (src, i, (name, ne, ty, off, psz)) in enumerate(tensors):
    if src != 'r': continue
    n = nbytes(ne, ty)
    jobs.append((k, [(rh.data0 + off + s, min(a.chunk, n - s)) for s in range(0, n, a.chunk)]))
t0 = time.time(); n_rep = 0; fetched = 0
with cf.ThreadPoolExecutor(a.jobs) as ex:
    flat = [(k, s, n) for k, ch in jobs for s, n in ch]
    futs = {}; nxt = 0
    def submit_upto(m):
        global nxt
        while nxt < len(flat) and nxt < m:
            k, s, n = flat[nxt]; futs[nxt] = ex.submit(fetch, a.url, s, n); nxt += 1
    submit_upto(a.jobs * 3)
    pos = 0
    for k, ch in jobs:
        parts = []
        for _ in ch:
            submit_upto(pos + a.jobs * 3)
            parts.append(futs.pop(pos).result()); pos += 1
        raw = b''.join(parts); fetched += len(raw)
        src, i, (name, ne, ty, off, psz) = tensors[k]
        if a.q2lut and ty == Q2_K and len(ne) == 3 and ne[0] % 256 == 0 and ne[1] % 16 == 0:
            raw = repack(np.frombuffer(raw, dtype=np.uint8), ne).tobytes(); n_rep += 1
        pad_to(lay[k]); w(raw)
        el = time.time() - t0
        print(f'  {name:28s} {nbytes(ne, ty) / 1e6:6.1f} MB  {fetched / 1e9:5.2f} GB in {el:4.0f} s ({fetched / 1e6 / max(el, 1e-3):4.1f} MB/s)', file=sys.stderr, flush=True)
pad_to(total_data)
out.flush()
if out is not sys.stdout.buffer: out.close()
print(f'{len(tensors)} tensors ({len(exps)} expert tensors last, {n_rep} repacked q2lut), {written} bytes, md5 {md5.hexdigest()}', file=sys.stderr)
