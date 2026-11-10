#!/usr/bin/env python3
"""Write a copy of a MoE GGUF that keeps PER experts in each of the 8 router groups, chosen by usage in a
GGML_MOE_TRACE calibration log (moe_keep.py's format). Expert tensors, router rows and router biases are sliced;
expert_count is set to the new total. Group order is kept, so group-limited routing still works.
usage: moe_slice.py in.gguf trace PER out.gguf"""
import collections, struct, sys
import numpy as np
src, trace, per, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
GROUPS = 8

tr = open(trace, 'rb').read(); o = 0; cnt = collections.defaultdict(collections.Counter)
while o < len(tr):
    il, n, t = struct.unpack_from('<3i', tr, o); o += 12
    for x in struct.unpack_from('<%di' % (n * t), tr, o):
        if x >= 0: cnt[il][x] += 1
    o += 4 * n * t

m = np.memmap(src, dtype=np.uint8, mode='r'); b = m; o = 0
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

magic, ver = rd('<4sI'); n_t, n_kv = rd('<QQ')
kv_start = o; align = 32; arch = None; patch_at = None; n_exp = None
for _ in range(n_kv):
    k = rstr(); (t,) = rd('<I')
    if k == b'general.alignment': align = struct.unpack_from('<I', b, o)[0]
    if k == b'general.architecture': arch = bytes(b[o + 8:o + 8 + struct.unpack_from('<Q', b, o)[0]]).decode()
    if k.endswith(b'.expert_count') and t == 4: patch_at = o; n_exp = struct.unpack_from('<I', b, o)[0]
    skip(t)
kv_end = o
assert patch_at and n_exp % GROUPS == 0
gsz = n_exp // GROUPS; K = per * GROUPS
infos = []
for _ in range(n_t):
    name = rstr().decode(); (nd,) = rd('<I'); ne = list(rd('<%dQ' % nd)); (ty,) = rd('<I'); (off,) = rd('<Q')
    infos.append([name, ne, ty, off])
data0 = (o + align - 1) // align * align
order = sorted(range(n_t), key=lambda i: infos[i][3])
for j, i in enumerate(order):   # byte sizes from offsets (the last one runs to the end of the file)
    nxt = infos[order[j + 1]][3] if j + 1 < len(order) else len(b) - data0
    infos[i].append(nxt - infos[i][3])

keep = {}
def keep_for(layer):
    if layer not in keep:
        c = cnt.get(layer, collections.Counter()); ks = []
        for g in range(GROUPS):
            grp = sorted(range(g * gsz, g * gsz + gsz), key=lambda e: (-c[e], e))[:per]
            ks += sorted(grp)
        keep[layer] = ks
    return keep[layer]

def sliced(name, ne):
    return ne and ne[-1] == n_exp and name.startswith('blk.') and any(s in name for s in ('_exps.', 'ffn_gate_inp.', 'exp_probs_b.'))

out = []
for name, ne, ty, off, size in infos:
    if sliced(name, ne):
        layer = int(name.split('.')[1]); slab = size // n_exp
        # trailing padding: the slab is the exact tensor size / n_exp when sizes are aligned
        exact = None
        out.append((name, ne[:-1] + [K], ty, ('slice', off, slab, keep_for(layer))))
    else:
        out.append((name, ne, ty, ('copy', off, size)))

def tsize(spec):
    return spec[2] * len(spec[3]) if spec[0] == 'slice' else spec[2]

with open(dst, 'wb') as f:
    hdr = bytearray(b[:kv_end]); struct.pack_into('<I', hdr, patch_at, K); f.write(hdr)
    cur = 0; offs = []
    for name, ne, ty, spec in out:
        offs.append(cur); cur = (cur + tsize(spec) + align - 1) // align * align
    for (name, ne, ty, spec), off in zip(out, offs):
        nb = name.encode(); f.write(struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(ne)) + struct.pack('<%dQ' % len(ne), *ne) + struct.pack('<IQ', ty, off))
    pos = f.tell(); f.write(b'\0' * ((pos + align - 1) // align * align - pos))
    base = f.tell()
    for (name, ne, ty, spec), off in zip(out, offs):
        f.seek(base + off)
        if spec[0] == 'copy':
            f.write(b[data0 + spec[1]: data0 + spec[1] + spec[2]])
        else:
            _, so, slab, ks = spec
            for e in ks: f.write(b[data0 + so + e * slab: data0 + so + (e + 1) * slab])
    f.truncate(base + cur)
print(f'{arch}: {n_exp} -> {K} experts ({per} per group), {len(keep)} MoE layers, wrote {dst}')
