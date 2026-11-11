#!/usr/bin/env python3
"""Download only some tensors of a remote GGUF (HTTP range requests) into a local GGUF with the same metadata.
Built for Ling-mini-2.0 (EXPERIMENT.md section 21.11): the Mac has ~3 GB free, so instead of the 9.9 GB Q4_K_M we take only its
non-expert tensors (~0.7 GB), requantize those, and stream the experts from another file later (gguf_moe_assemble.py).
usage: gguf_fetch_subset.py URL out.gguf [--exclude SUBSTR] [--include SUBSTR]"""
import argparse, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_remote import remote_header, fetch, info_bytes, nbytes

ap = argparse.ArgumentParser()
ap.add_argument('url'); ap.add_argument('out')
ap.add_argument('--exclude', default='_exps.'); ap.add_argument('--include', default=None)
a = ap.parse_args()

h, size = remote_header(a.url)
keep = [i for i in h.order if a.exclude not in h.infos[i][0] and (a.include is None or a.include in h.infos[i][0])]
lay, cur = {}, 0
for i in keep:
    lay[i] = cur; cur = (cur + nbytes(h.infos[i][1], h.infos[i][2]) + h.align - 1) // h.align * h.align
with open(a.out, 'wb') as f:
    f.write(struct.pack('<4sIQQ', b'GGUF', h.version, len(keep), h.n_kv)); f.write(h.kv_bytes())
    for i in keep:
        name, ne, ty, off, _ = h.infos[i]; f.write(info_bytes(name, ne, ty, lay[i]))
    pos = f.tell(); f.write(b'\0' * ((pos + h.align - 1) // h.align * h.align - pos)); base = f.tell()
    # merge neighbouring tensors into one request when they are contiguous in the source
    runs = []
    for i in keep:
        name, ne, ty, off, psz = h.infos[i]
        if runs and runs[-1][1] == off: runs[-1][1] = off + psz; runs[-1][2].append(i)
        else: runs.append([off, off + psz, [i]])
    got = 0
    for s, e, ids in runs:
        buf = fetch(a.url, h.data0 + s, e - s)
        for i in ids:
            name, ne, ty, off, psz = h.infos[i]
            f.seek(base + lay[i]); f.write(buf[off - s: off - s + nbytes(ne, ty)])
        got += e - s
    f.truncate(base + cur)
print(f'{len(keep)} of {h.n_t} tensors, {got / 1e6:.0f} MB fetched; wrote {a.out}')
