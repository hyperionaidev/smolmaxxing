#!/usr/bin/env python3
"""Cut a GGUF that lives on the phone into llama.cpp split shards on the Mac, one shard at a time, so a file bigger than the Mac's free
disk (or the phone's) can be published. Same layout as llama-gguf-split: shard 1 has all the metadata, every shard has split.no,
split.count and split.tensors.count, and a shard's tensors are a contiguous run of the original file's, byte for byte, with their
offsets rebased. Bytes are read over adb with the phone-side `rangecat` (tools/rangecat).
usage: gguf_split_adb.py PHONE_PATH N_SHARDS OUT_PREFIX K      writes OUT_PREFIX-0000K-of-0000N.gguf; the shard's data is checked
       against a sha256 of the same byte range taken on the phone"""
import hashlib, os, struct, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_remote import Header

path, n, prefix, k = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
S = os.environ.get('ADB_SERIAL') or subprocess.run(['adb', 'get-serialno'], capture_output=True, text=True).stdout.strip()
RC = os.path.join(os.path.dirname(path), 'rangecat')


def adb_range(off, ln):   # small reads only (the header)
    return subprocess.Popen(['adb', '-s', S, 'exec-out', RC, path, str(off), str(ln)], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)


size = int(subprocess.run(['adb', '-s', S, 'shell', 'stat', '-c', '%s', path], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout)
buf = b''; m = 16 << 20
while True:
    buf = adb_range(0, min(m, size)).communicate()[0]
    try:
        h = Header(buf, size); break
    except struct.error:
        m *= 2
order = h.order                                  # tensor indices by offset
data_total = size - h.data0
# split at tensor boundaries into n runs of about equal data
cuts, acc, target = [0], 0, data_total / n
for j, i in enumerate(order):
    if len(cuts) < n and acc >= target * len(cuts):
        cuts.append(j)
    acc += h.infos[i][4]
cuts.append(len(order))
run = order[cuts[k - 1]:cuts[k]]
off0 = h.infos[run[0]][3]
dlen = sum(h.infos[i][4] for i in run)


def kv(key, ty, fmt, val):
    kb = key.encode()
    return struct.pack('<Q', len(kb)) + kb + struct.pack('<I', ty) + struct.pack(fmt, val)


split_kv = kv('split.no', 2, '<H', k - 1) + kv('split.count', 2, '<H', n) + kv('split.tensors.count', 5, '<i', h.n_t)
kvb = (bytes(buf[h.kv0:h.kv1]) + split_kv) if k == 1 else split_kv
n_kv = (h.n_kv + 3) if k == 1 else 3
infos = b''
for i in run:
    name, ne, ty, off = h.infos[i][:4]
    nb = name.encode()
    infos += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(ne)) + struct.pack('<%dQ' % len(ne), *ne) + struct.pack('<IQ', ty, off - off0)
hdr = b'GGUF' + struct.pack('<IQQ', 3, len(run), n_kv) + kvb + infos
hdr += b'\0' * ((-len(hdr)) % h.align)
out = '%s-%05d-of-%05d.gguf' % (prefix, k, n)
# The shard is built on the phone (header pushed, data appended by rangecat) and then pulled: a 14 GB `adb exec-out` stream stalled
# halfway with rangecat blocked in n_tty_write (E165), while `adb pull` of a file is robust. Needs the shard's size free on the phone.
tmp = os.path.join(os.path.dirname(path), 'shard.tmp'); hfile = out + '.hdr'
open(hfile, 'wb').write(hdr)
adb = ['adb', '-s', S]
subprocess.run(adb + ['push', hfile, tmp + '.hdr'], check=True, stdin=subprocess.DEVNULL, capture_output=True)
r = subprocess.run(adb + ['shell', 'cat %s.hdr > %s && %s %s %d %d | tee -a %s | sha256sum && rm %s.hdr' % (tmp, tmp, RC, path, h.data0 + off0, dlen, tmp, tmp)],
                   check=True, stdin=subprocess.DEVNULL, capture_output=True, text=True)
phone_sha = r.stdout.split()[0]
subprocess.run(adb + ['pull', tmp, out], check=True, stdin=subprocess.DEVNULL, capture_output=True)
subprocess.run(adb + ['shell', 'rm', tmp], check=True, stdin=subprocess.DEVNULL)
os.remove(hfile)
sha = hashlib.sha256()
with open(out, 'rb') as f:
    assert f.read(len(hdr)) == hdr
    got = 0
    while True:
        b = f.read(8 << 20)
        if not b: break
        sha.update(b); got += len(b)
assert got == dlen, 'short: %d of %d' % (got, dlen)
assert sha.hexdigest() == phone_sha, 'data sha mismatch: %s vs phone %s' % (sha.hexdigest(), phone_sha)
print('%s: %d tensors (%s .. %s), header %d bytes, data %d bytes, data sha256 %s (matches the phone)' % (out, len(run), h.infos[run[0]][0], h.infos[run[-1]][0], len(hdr), dlen, phone_sha))
