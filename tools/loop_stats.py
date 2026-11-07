#!/usr/bin/env python3
"""How often does a server's model fall into a loop? N seeds of one long prompt (and a second prompt for variety).
A run counts as looping if a repeated stretch of at least 40 words occurs (consecutive 6-grams seen earlier). Prints one summary line.
usage: URL=http://... loop_stats.py <label> [seeds=16] [max_tokens=979] [temp=0.7]    full texts append to $TXT if set"""
import json, os, statistics as st, sys, time, urllib.request
URL = os.environ.get("URL", "http://127.0.0.1:8092"); label = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MT = int(sys.argv[3]) if len(sys.argv) > 3 else 979; TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
P = ["Write a long, detailed essay about the history of computing, from the abacus to modern AI. Cover every decade.",
     "Tell me a long, detailed story about a lighthouse keeper who finds a mysterious map, and everything that happens afterwards."]
if os.environ.get("PROMPTS") == "code":   # code repeats tokens by nature: what a repeat penalty could break
    P = ["Write a complete Python module for a command-line to-do list app: add, list, complete and delete tasks, saved to a JSON file, with argparse, docstrings and error handling.",
         "Write a C program that reads a text file and prints the 10 most frequent words with their counts, using a hash table you implement yourself."]
rows = []; t0 = time.time(); DR = [0, 0, 0]
for i in range(N):
    b = {"messages": [{"role": "user", "content": P[i % 2]}], "max_tokens": MT, "temperature": TEMP, "top_k": 40, "top_p": 0.95, "seed": 100 + i, "cache_prompt": False, **json.loads(os.environ.get("SAMPLER", "{}"))}
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
    except Exception as e:
        print("%s: FAILED at seed %d: %s" % (label, i, e)); sys.exit(1)
    t = r['choices'][0]['message']['content']; w = t.split(); seen = set(); rep = run = best = 0; start = None
    for j in range(len(w) - 5):
        g = tuple(w[j:j + 6])
        if g in seen:
            rep += 1; run += 1; best = max(best, run)
            if run == 35 and start is None: start = j - 34
        else: run = 0
        seen.add(g)
    rows.append((100 * rep / max(1, len(w) - 5), best, start, r['timings']['predicted_per_second'], len(w)))
    DR[0] += r['timings'].get('draft_n', 0) or 0; DR[1] += r['timings'].get('draft_n_accepted', 0) or 0; DR[2] += r['timings'].get('predicted_n', 0) or 0
    if os.environ.get("TXT"): open(os.environ["TXT"], "a").write("=== %s seed %d: rep %.1f %% best %d\n%s\n\n" % (label, 100 + i, rows[-1][0], best, t))
loops = [x for x in rows if x[1] >= 35]
print("%-34s loops %2d/%d | repeats mean %5.1f median %5.1f %% | longest run median %4d | loop starts at word %s | %.1f tok/s | %d words" % (
    label, len(loops), N, st.mean(x[0] for x in rows), st.median(x[0] for x in rows), st.median(x[1] for x in rows),
    int(st.median(x[2] for x in loops)) if loops else '-', st.mean(x[3] for x in rows), st.mean(x[4] for x in rows)) +
    (" | drafts %d, accepted %.1f %%, %.2f tokens per pass" % (DR[0], 100 * DR[1] / DR[0], DR[2] / max(1, DR[2] - DR[1])) if DR[0] else ""), flush=True)
