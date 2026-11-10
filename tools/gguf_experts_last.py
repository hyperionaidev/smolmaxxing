#!/usr/bin/env python3
"""Rewrite a GGUF so every expert tensor (*_exps.*) comes after all other tensors. Nothing else changes.
With the expert pager (GGML_MOE_PAGER=1, ipad-air patch) llama.cpp then maps only the leading non-expert
part of the file and reads experts with pread, so a model bigger than the per-process address-space limit
(1,728 MB on the iPad Air, EXPERIMENT.md E136) can still load.
usage: gguf_experts_last.py in.gguf out.gguf"""
import struct, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
b = np.memmap(src, dtype=np.uint8, mode='r'); o = 0
def rd(fmt):
    global o
    v = struct.unpack_from(fmt, b, o); o += struct.calcsize(fmt); return v
def rstr():
    global o
    (n,) = rd('<Q'); s = bytes(b[o:o + n]); o += n; return s
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def skip(t):
    global o
    if t == 8: rstr()
    elif t == 9:
        (et,) = rd('<I'); (n,) = rd('<Q')
        if et in SZ: o += SZ[et] * n
        else:
            for _ in range(n): skip(et)
    else: o += SZ[t]

magic, ver = rd('<4sI'); assert magic == b'GGUF'
n_t, n_kv = rd('<QQ'); align = 32
for _ in range(n_kv):
    k = rstr(); (t,) = rd('<I')
    if k == b'general.alignment': align = struct.unpack_from('<I', b, o)[0]
    skip(t)
kv_end = o
infos = []
for _ in range(n_t):
    name = rstr().decode(); (nd,) = rd('<I'); ne = list(rd('<%dQ' % nd)); (ty,) = rd('<I'); (off,) = rd('<Q')
    infos.append([name, ne, ty, off])
data0 = (o + align - 1) // align * align
order = sorted(range(n_t), key=lambda i: infos[i][3])
for j, i in enumerate(order):   # padded byte sizes from offsets; the last runs to the end of the file
    nxt = infos[order[j + 1]][3] if j + 1 < len(order) else len(b) - data0
    infos[i].append(nxt - infos[i][3])

is_exp = lambda n: '_exps.' in n
# file order otherwise kept: non-experts first, then experts (tensor infos follow, gguf wants them in data order)
new_order = [i for i in order if not is_exp(infos[i][0])] + [i for i in order if is_exp(infos[i][0])]
new_off = {}; cur = 0
for i in new_order:
    new_off[i] = cur; cur = (cur + infos[i][4] + align - 1) // align * align

with open(dst, 'wb') as f:
    f.write(bytes(b[:kv_end]))
    for i in new_order:   # gguf requires the infos in data order
        name, ne, ty, off, size = infos[i]
        nb = name.encode()
        f.write(struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(ne)) + struct.pack('<%dQ' % len(ne), *ne) + struct.pack('<IQ', ty, new_off[i]))
    pos = f.tell(); f.write(b'\0' * ((pos + align - 1) // align * align - pos))
    base = f.tell()
    first_exp = None
    for i in new_order:
        name, ne, ty, off, size = infos[i]
        if is_exp(name) and first_exp is None: first_exp = base + new_off[i]
        f.seek(base + new_off[i]); f.write(b[data0 + off: data0 + off + size])
    f.truncate(base + cur)
n_e = sum(1 for x in infos if is_exp(x[0]))
print(f'{n_t} tensors, {n_e} expert tensors moved last; non-expert part ends at {first_exp / 1e6:.1f} MB of {(base + cur) / 1e6:.1f} MB; wrote {dst}')
