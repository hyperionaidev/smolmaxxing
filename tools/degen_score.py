#!/usr/bin/env python3
"""Degeneration that a repeated-6-gram count misses (DRY turns exact loops into paraphrased ones: "The abacus was traded. The abacus was
valued. ..."). Per config: zlib compression ratio of the second half of each answer (lower = more repetitive; normal prose is ~0.40-0.45),
share of sentences whose first three words already opened an earlier sentence, and distinct-2 (unique bigrams / bigrams).
A run counts as degenerate if compression < 0.30 or repeated openings > 40 %. usage: degen_score.py <texts.txt>..."""
import collections, re, statistics as st, sys, zlib
by = collections.OrderedDict()
for path in sys.argv[1:]:
    for sec in open(path, errors='replace').read().split('=== ')[1:]:
        head, _, t = sec.partition('\n')
        label = re.sub(r' seed \d+.*', '', head).strip()
        label = label.replace('{"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":1024}', '+DRY0.8').replace('{"dry_multiplier":1.2,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":1024}', '+DRY1.2').replace('{"repeat_penalty":1.15,"repeat_last_n":256}', '+rep1.15').replace('{"presence_penalty":0.6}', '+pres0.6')
        w = t.split()
        if len(w) < 120: continue
        half = ' '.join(w[len(w) // 2:]).encode()
        comp = len(zlib.compress(half, 9)) / max(1, len(half))
        sents = [s.split()[:3] for s in re.split(r'[.!?\n]+', t) if len(s.split()) >= 3]
        seen = set(); rep = 0
        for s in sents:
            k = ' '.join(x.lower() for x in s)
            if k in seen: rep += 1
            seen.add(k)
        ro = rep / max(1, len(sents))
        bg = list(zip(w, w[1:])); d2 = len(set(bg)) / max(1, len(bg))
        by.setdefault(label, []).append((comp, ro, d2))
print("%-44s %3s  %6s %6s %6s  %s" % ("config", "n", "zlib", "open%", "dist2", "degenerate"))
for k, v in by.items():
    bad = sum(1 for c, r, d in v if c < 0.30 or r > 0.40)
    print("%-44s %3d  %6.3f %6.1f %6.3f  %d/%d" % (k[:44], len(v), st.mean(x[0] for x in v), 100 * st.mean(x[1] for x in v), st.mean(x[2] for x in v), bad, len(v)))
