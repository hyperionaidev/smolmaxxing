#!/usr/bin/env python3
"""Summarize a GGML_OPPROF=2 trace (one graph, per-node per-thread compute us). usage: optrace.py <log> [n_rows]"""
import sys, collections
L = [l for l in open(sys.argv[1], errors="replace") if l.startswith("[optrace]")]
print(L[0].strip())
rows = []
for l in L[1:]:
    p = l.split("|"); a, b = p[0], p[1]; f = a.split(); us = list(map(float, b.split()))   # later columns: start skew, t offset
    rows.append((int(f[1]), f[2], " ".join(f[3:-3]), f[-3], int(f[-2]), int(f[-1]), us))
mx = sum(max(r[6]) for r in rows); mean = sum(sum(r[6]) / len(r[6]) for r in rows)
print("nodes %d | critical path (sum of slowest thread per node) %.0f us | sum of mean %.0f us | imbalance %.0f us" % (len(rows), mx, mean, mx - mean))
agg = collections.defaultdict(lambda: [0, 0, 0.0])
for r in rows:
    a = agg[r[1] + ":" + r[3]]; a[0] += 1; a[1] += max(r[6]); a[2] += sum(r[6]) / len(r[6])
for k, v in sorted(agg.items(), key=lambda x: -x[1][1])[:24]:
    print("%-28s n=%4d  slowest-thread sum %8.0f us  mean %8.0f us  per node %6.1f us" % (k, v[0], v[1], v[2], v[1] / v[0]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for r in rows[:n]:
    print("%4d %-14s %-34s %-5s %6d %5d  %s" % (r[0], r[1], r[2][:34], r[3], r[4], r[5], " ".join("%6.1f" % x for x in r[6])))
