#!/usr/bin/env python3
"""Approximate LM-head experiments (EXPERIMENT.md §15): can a cheap low-rank / clustered score find the tokens the
exact head would pick? Inputs: a Q8_0 GGUF (tied token_embd = head), hidden states dumped by LLAMA_DUMP_HIDDEN
(llama-perplexity, ctx 128), and the eval token ids. usage: head_approx.py <gguf> <h.f32> <ids.txt> [vocab=full|h65]"""
import sys, json, numpy as np
sys.path.insert(0, "third_party/llama.cpp/gguf-py")
import gguf
from gguf.quants import dequantize
r = gguf.GGUFReader(sys.argv[1])
T = {t.name: t for t in r.tensors}
def deq(name):
    t = T[name]; return dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0])).astype(np.float32)
E = deq("token_embd.weight")                    # [248320, 1024]
H = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, E.shape[1])
ids = json.load(open(sys.argv[3]))
# llama-perplexity only outputs the second half of each 128-token chunk (positions 64..127): 64 rows per chunk
C = 128; F = C // 2; nch = len(H) // F; H = H[-nch * F:]   # drop any warm-up rows at the start
print("vocab", E.shape, "hidden rows", H.shape, "chunks", nch)
if len(sys.argv) > 4 and sys.argv[4] == "h65":
    keep = np.array(sorted(set(int(x) for x in open("data/head-ids-ml60k.txt").read().split())), dtype=np.int64)
else:
    keep = np.arange(E.shape[0])
W = E[keep]
# scored positions: second half of every chunk (as llama-perplexity), target = next token
pos, tgt = [], []
for c in range(nch):
    for j in range(F, C - 1):
        pos.append(c * F + (j - F)); tgt.append(ids[c * C + j + 1])
Hs = H[pos]; tgt = np.array(tgt)
inv = {int(t): i for i, t in enumerate(keep)}
ok = np.array([t in inv for t in tgt]); ti = np.array([inv.get(int(t), 0) for t in tgt])
Z = Hs @ W.T                                   # exact logits
def nll(Z):
    m = Z.max(1, keepdims=True); lse = (m[:, 0] + np.log(np.exp(Z - m).sum(1)))
    return lse[ok] - Z[ok, ti[ok]]
print("exact: PPL %.4f on %d positions (%d misses)" % (np.exp(nll(Z).mean()), ok.sum(), (~ok).sum()))
top1 = Z.argmax(1)
U, S, Vt = np.linalg.svd(W, full_matrices=False)
for rank in (64, 128, 192, 256, 384):
    A = (Hs @ Vt[:rank].T) @ (U[:, :rank] * S[:rank]).T     # low-rank scores
    for K in (16, 64, 256):
        cand = np.argpartition(-A, K, axis=1)[:, :K]
        hit = (cand == top1[:, None]).any(1).mean()
        # logits: exact for candidates, low-rank elsewhere
        Z2 = A.copy(); rows = np.arange(len(Z))[:, None]; Z2[rows, cand] = Z[rows, cand]
        g = (Z2.argmax(1) == top1).mean()
        print("rank %3d K %3d: top-1 in candidates %.4f, greedy agreement %.4f, PPL %.4f, cost %.1f M weights" %
              (rank, K, hit, g, np.exp(nll(Z2).mean()), (rank * (W.shape[0] + W.shape[1]) + K * W.shape[1]) / 1e6))
