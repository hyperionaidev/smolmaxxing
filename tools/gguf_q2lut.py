#!/usr/bin/env python3
"""Repack the Q2_K expert tensors of an experts-last GGUF (tools/gguf_experts_last.py) for the iPad's lookup-table kernel
(ipad-air patch, ggml-cpu/moe-pager-ios.inc, EXPERIMENT.md section 21.9). Every byte stays inside its expert, and each expert keeps
its size, so offsets don't change; the file gets the key ipad.expert_layout = "q2lut". Only the expert pager can read these experts:
to stock llama.cpp they are scrambled Q2_K.

Q2_K block (256 weights): scales[16] (low nibble scale, high nibble min, per 16 weights), qs[64] (2-bit codes), d, dmin (fp16).
Repacked, per expert, per group of 16 rows, per 256 columns (1,344 bytes = 16 blocks):
  d[16 rows] fp16 | dmin[16 rows] fp16 | sc[16 sub-blocks][16 rows] | qs[16 sub-blocks][4][16 rows]
where qs byte (j, v, r) = nib(r, 2v) | nib(r, 2v+1) << 4 and nib(r, k) = q(r, 16j+2k) | q(r, 16j+2k+1) << 2: one 4-bit index per row
and activation pair, so a 16-byte vector indexes 16 rows' entries of a 16-entry pair table.
usage: gguf_q2lut.py in-xl.gguf out.gguf"""
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
n_t, n_kv = rd('<QQ'); kv0 = o; align = 32
for _ in range(n_kv):
    k = rstr(); (t,) = rd('<I')
    assert k != b'ipad.expert_layout', 'already repacked'
    if k == b'general.alignment': align = struct.unpack_from('<I', b, o)[0]
    skip(t)
kv1 = o
infos = []
for _ in range(n_t):
    name = rstr().decode(); (nd,) = rd('<I'); ne = list(rd('<%dQ' % nd)); (ty,) = rd('<I'); (off,) = rd('<Q')
    infos.append((name, ne, ty, off))
ti1 = o
data0 = (o + align - 1) // align * align
Q2_K = 10

def repack(raw, ne):
    ne00, ne01, n_exp = ne
    nsb, G = ne00 // 256, ne01 // 16
    blk = raw.reshape(n_exp, G, 16, nsb, 84)                       # expert, row group, row, super-block
    scales, qs = blk[..., 0:16], blk[..., 16:80]
    d, dmin = blk[..., 80:82], blk[..., 82:84]
    q = qs.reshape(n_exp, G, 16, nsb, 2, 1, 32)
    codes = (q >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 1, 1, 4, 1)) & 3   # (.., half, j, l): position half*128+j*32+l
    codes = codes.reshape(n_exp, G, 16, nsb, 16, 8, 2)            # sub-block, pair, member
    nib = codes[..., 0] | (codes[..., 1] << 2)                     # (E, G, 16r, nsb, 16j, 8k)
    nib = nib.reshape(n_exp, G, 16, nsb, 16, 4, 2)
    byte = nib[..., 0] | (nib[..., 1] << 4)                        # (E, G, 16r, nsb, 16j, 4v)
    out = np.empty((n_exp, G, nsb, 1344), dtype=np.uint8)
    out[..., 0:32] = d.transpose(0, 1, 3, 2, 4).reshape(n_exp, G, nsb, 32)
    out[..., 32:64] = dmin.transpose(0, 1, 3, 2, 4).reshape(n_exp, G, nsb, 32)
    out[..., 64:320] = scales.transpose(0, 1, 3, 4, 2).reshape(n_exp, G, nsb, 256)
    out[..., 320:1344] = byte.transpose(0, 1, 3, 4, 5, 2).reshape(n_exp, G, nsb, 1024)
    return out.reshape(-1)

key = b'ipad.expert_layout'; val = b'q2lut'
kv = struct.pack('<Q', len(key)) + key + struct.pack('<I', 8) + struct.pack('<Q', len(val)) + val
with open(dst, 'wb') as f:
    f.write(b[:kv0 - 16].tobytes()); f.write(struct.pack('<QQ', n_t, n_kv + 1))
    f.write(b[kv0:kv1].tobytes()); f.write(kv); f.write(b[kv1:ti1].tobytes())
    pos = f.tell(); f.write(b'\0' * ((pos + align - 1) // align * align - pos))
    base = f.tell()
    order = sorted(range(n_t), key=lambda i: infos[i][3])
    n_rep = 0
    for j, i in enumerate(order):
        name, ne, ty, off = infos[i]
        end = infos[order[j + 1]][3] if j + 1 < len(order) else len(b) - data0
        raw = b[data0 + off: data0 + end]
        f.seek(base + off)
        if '_exps.' in name and ty == Q2_K and len(ne) == 3 and ne[0] % 256 == 0 and ne[1] % 16 == 0:
            n = ne[0] // 256 * 84 * ne[1] * ne[2]
            f.write(repack(np.asarray(raw[:n]), ne).tobytes()); f.write(raw[n:].tobytes()); n_rep += 1
        else:
            f.write(raw.tobytes())
    f.truncate(base + (len(b) - data0))
print(f'repacked {n_rep} expert tensors for the q2lut kernel; wrote {dst}')
