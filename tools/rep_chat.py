#!/usr/bin/env python3
"""Free-running generation vs repetition: PPL scores one token at a time and can't see a model falling into a loop.
Generates from fixed prompts and seeds on the TV's server; reports the share of 6-grams that are repeats and the longest repeated run.
usage: rep_chat.py <label> [max_tokens] [repeat_penalty]     appends to bench/raw/E100-repetition.txt"""
import json, sys, urllib.request
PROMPTS = ["Write a long, detailed essay about the history of computing, from the abacus to modern AI.",
           "Explain in detail how a car engine works, and then how an electric motor differs from it.",
           "Tell me a long story about a lighthouse keeper who finds a mysterious map."]
label = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 300; rp = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
tot_rep = tot = 0; tps = []
out = open('bench/raw/E100-repetition.txt', 'a')
for i, p in enumerate(PROMPTS):
    b = {"messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0.7, "seed": 42 + i, "repeat_penalty": rp, "repeat_last_n": 128}
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    t = r['choices'][0]['message']['content']; w = t.split()
    seen = set(); rep = 0; run = best = 0
    for j in range(len(w) - 5):
        g = tuple(w[j:j + 6])
        if g in seen: rep += 1; run += 1; best = max(best, run)
        else: run = 0
        seen.add(g)
    tot_rep += rep; tot += max(1, len(w) - 5); tps.append(r['timings']['predicted_per_second'])
    out.write("=== %s prompt %d: %d words, repeated 6-grams %.1f %%, longest repeated run %d words\n%s\n\n" % (label, i, len(w), 100 * rep / max(1, len(w) - 5), best, t))
    print("  prompt %d: %d words, repeated 6-grams %5.1f %%, longest run %3d, %.2f tok/s" % (i, len(w), 100 * rep / max(1, len(w) - 5), best, r['timings']['predicted_per_second']))
print("%s: repeated 6-grams %.1f %% overall, mean %.2f tok/s" % (label, 100 * tot_rep / tot, sum(tps) / len(tps)))
out.write("##### %s: repeated 6-grams %.1f %% overall, mean %.2f tok/s\n\n" % (label, 100 * tot_rep / tot, sum(tps) / len(tps))); out.close()
