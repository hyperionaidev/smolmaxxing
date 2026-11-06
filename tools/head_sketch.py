#!/usr/bin/env python3
"""Low-bit sketch of the LM head (round 5): score every kept row with a 1- or 2-bit copy of the head, re-score the top K
exactly. Unlike the low-rank attempt of E60 this keeps all dimensions. Inputs as tools/head_approx.py.
usage: head_sketch.py <q8 gguf with pruned head> <h.f32> <ids json>"""
import sys, json, numpy as np
sys.path.insert(0, "third_party/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize
r = gguf.GGUFReader(sys.argv[1]); T = {t.name: t for t in r.tensors}
def deq(name):
    t = T[name]; return dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0])).astype(np.float32)
W = deq("output.weight"); print("head", W.shape)
keep = np.array(sorted(set(int(x) for x in open(sys.argv[4] if len(sys.argv) > 4 else "data/head-ids-ml60k.txt").read().split())), dtype=np.int64); assert len(keep) == W.shape[0]
H = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, W.shape[1])
ids = json.load(open(sys.argv[3]))
C = 128; F = C // 2; nch = len(H) // F; H = H[-nch * F:]
pos, tgt = [], []
for c in range(nch):
    for j in range(F, C - 1):
        pos.append(c * F + (j - F)); tgt.append(ids[c * C + j + 1])
Hs = H[pos]; tgt = np.array(tgt)
inv = {int(t): i for i, t in enumerate(keep)}
ok = np.array([t in inv for t in tgt]); ti = np.array([inv.get(int(t), 0) for t in tgt])
Z = Hs @ W.T
def nll(Z):
    m = Z.max(1, keepdims=True); lse = (m[:, 0] + np.log(np.exp(Z - m).sum(1)))
    return lse[ok] - Z[ok, ti[ok]]
print("exact: PPL %.4f on %d positions (%d misses)" % (np.exp(nll(Z).mean()), ok.sum(), (~ok).sum()))
top1 = Z.argmax(1); rows = np.arange(len(Z))[:, None]
top10 = np.argpartition(-Z, 10, axis=1)[:, :10]
full_bytes = W.size * 4.5 / 8
norm = np.linalg.norm(W, axis=1, keepdims=True)
sd = W.std(); 
sketches = {
 "sign": (np.sign(W), 1.0),
 "sign*rownorm": (np.sign(W) * norm, 1.0 + 8 * W.shape[0] / W.size),
 "2-bit": (np.round(np.clip(W / (W.std(1, keepdims=True) * 1.0), -2, 1.0)).clip(-2, 1) * W.std(1, keepdims=True), 2.0 + 8 * W.shape[0] / W.size),
}
for name, (A, bits) in sketches.items():
    Za = Hs @ A.T
    # shift/scale so the unre-scored rows stay comparable with the exact logits (least squares per position)
    for K in (16, 64, 256, 1024):
        cand = np.argpartition(-Za, K, axis=1)[:, :K]
        hit = (cand == top1[:, None]).any(1).mean()
        hit10 = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top10, cand)])
        Z2 = np.full_like(Z, -1e9); Z2[rows, cand] = Z[rows, cand]
        tin = (cand == ti[:, None]).any(1) & ok
        nl = nll(np.where(Z2 > -1e8, Z2, Z.min(1, keepdims=True) - 5))
        mb = (W.size * bits / 8 + K * W.shape[1] * 4.5 / 8) / 1e6
        print("%-13s K %4d: top-1 %.4f top-10 share %.4f target kept %.4f PPL %.3f, %.1f MB read vs %.1f" % (name, K, hit, hit10, tin[ok].mean(), np.exp(nl.mean()), mb, full_bytes / 1e6))
