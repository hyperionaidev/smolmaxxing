#!/usr/bin/env python3
"""Shared helpers for reading a GGUF over HTTP with range requests (Hugging Face resolve URLs), so a model can be cut and
reassembled without downloading or storing the whole file. Used by gguf_fetch_subset.py and gguf_moe_assemble.py."""
import struct, subprocess, sys, time

SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# ggml type -> (block size, bytes per block)
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110),
         12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
         21: (256, 110), 22: (256, 82), 23: (256, 136), 26: (1, 4), 29: (256, 56), 30: (1, 2)}


def nbytes(ne, ty):
    bs, bb = BLOCK[ty]
    n = 1
    for x in ne: n *= x
    assert n % bs == 0
    return n // bs * bb


def fetch(url, start, n, tries=6):
    """Bytes [start, start + n) of url (curl follows the Hugging Face redirect)."""
    for t in range(tries):
        p = subprocess.run(['curl', '-sfL', '--max-time', '600', '-r', f'{start}-{start + n - 1}', url], capture_output=True)
        if p.returncode == 0 and len(p.stdout) == n:
            return p.stdout
        print(f'  range {start}+{n}: curl rc {p.returncode}, got {len(p.stdout)} bytes, retry {t + 1}', file=sys.stderr)
        time.sleep(2 + 3 * t)
    raise IOError(f'could not fetch {url} [{start}, {start + n})')


def remote_size(url):
    p = subprocess.run(['curl', '-sIL', url], capture_output=True, text=True, check=True)
    sizes = [int(l.split(':', 1)[1]) for l in p.stdout.splitlines() if l.lower().startswith('content-length:')]
    return sizes[-1]


class Header:
    """Parsed GGUF header: raw KV bytes, tensor infos [name, ne, type, offset, padded size], data start, alignment."""
    def __init__(self, buf, file_size):
        self.b, self.o = buf, 0
        magic, self.version = self.rd('<4sI'); assert magic == b'GGUF', 'not a GGUF'
        self.n_t, self.n_kv = self.rd('<QQ')
        self.kv0 = self.o; self.align = 32; self.keys = {}
        for _ in range(self.n_kv):
            k = self.rstr(); (t,) = self.rd('<I'); v0 = self.o
            self.skip(t)
            if t in SCALAR or t == 8: self.keys[k.decode()] = (t, bytes(buf[v0:self.o]))
            if k == b'general.alignment': self.align = struct.unpack_from('<I', buf, v0)[0]
        self.kv1 = self.o
        self.infos = []
        for _ in range(self.n_t):
            name = self.rstr().decode(); (nd,) = self.rd('<I'); ne = list(self.rd('<%dQ' % nd)); (ty,) = self.rd('<I'); (off,) = self.rd('<Q')
            self.infos.append([name, ne, ty, off])
        self.data0 = (self.o + self.align - 1) // self.align * self.align
        order = sorted(range(self.n_t), key=lambda i: self.infos[i][3])
        for j, i in enumerate(order):
            nxt = self.infos[order[j + 1]][3] if j + 1 < len(order) else file_size - self.data0
            self.infos[i].append(nxt - self.infos[i][3])
        self.order = order

    def rd(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.o); self.o += struct.calcsize(fmt); return v

    def rstr(self):
        (n,) = self.rd('<Q'); s = bytes(self.b[self.o:self.o + n]); self.o += n; return s

    def skip(self, t):
        if t == 8: self.rstr()
        elif t == 9:
            (et,) = self.rd('<I'); (n,) = self.rd('<Q')
            if et in SCALAR: self.o += SCALAR[et] * n
            else:
                for _ in range(n): self.skip(et)
        else: self.o += SCALAR[t]

    def kv_bytes(self):
        return bytes(self.b[self.kv0:self.kv1])


def remote_header(url, guess=16 << 20):
    size = remote_size(url)
    n = guess
    while True:
        buf = fetch(url, 0, min(n, size))
        try:
            return Header(buf, size), size
        except struct.error:
            n *= 2


def info_bytes(name, ne, ty, off):
    nb = name.encode()
    return struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(ne)) + struct.pack('<%dQ' % len(ne), *ne) + struct.pack('<IQ', ty, off)


def kv_str(key, val):
    k, v = key.encode(), val.encode()
    return struct.pack('<Q', len(k)) + k + struct.pack('<I', 8) + struct.pack('<Q', len(v)) + v
