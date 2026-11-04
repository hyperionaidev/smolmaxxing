#!/usr/bin/env python3
"""Benchmark the llama-server running on the TV box: fixed prompts, greedy, records speed + text.
usage: python3 -I bench.py <label> [max_tokens]     appends to bench/results.jsonl
"""
import json, os, sys, time, urllib.request

URL = "http://%s:8090" % os.environ["ADB_SERIAL"].split(":")[0]   # ADB_SERIAL=<tv-ip>:5555
PROMPTS = [
    "What is the capital of Australia, and why was it chosen over Sydney? Answer in two sentences.",
    "Explain how a TV remote control works, in simple words, in about 80 words.",
    "Write a Python function that checks if a number is prime, with a short docstring.",
    "List five interesting facts about the planet Mars, one line each.",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../bench/results.jsonl")

def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())

def metrics():
    m = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        if line.startswith("llamacpp:"):
            k, v = line[9:].split(" ", 1); m[k] = float(v)
    return m

def main():
    label = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    model = json.loads(urllib.request.urlopen(URL + "/props", timeout=30).read())["model_path"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # warm-up (first request pays page-in costs)
    post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4, "temperature": 0})
    m0 = metrics()
    rows = []
    for p in PROMPTS:
        t0 = time.time()
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0})
        tm = r.get("timings", {})
        text = r["choices"][0]["message"]["content"]
        rows.append({"prompt": p, "text": text, "pp_tps": tm.get("prompt_per_second"), "tg_tps": tm.get("predicted_per_second"),
                     "n": tm.get("predicted_n"), "wall_s": round(time.time() - t0, 1),
                     "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted")})
        print(f"  tg {tm.get('predicted_per_second', 0):5.2f} t/s  pp {tm.get('prompt_per_second', 0):5.1f} t/s  "
              f"n={tm.get('predicted_n')}  {text[:70]!r}")
    m1 = metrics()
    drafted = m1.get("spec_decode_num_draft_tokens_total", 0) - m0.get("spec_decode_num_draft_tokens_total", 0)
    acc = m1.get("spec_decode_num_accepted_tokens_total", 0) - m0.get("spec_decode_num_accepted_tokens_total", 0)
    tot_n = sum(r["n"] or 0 for r in rows)
    tot_t = sum((r["n"] or 0) / r["tg_tps"] for r in rows if r["tg_tps"])
    summary = {"label": label, "model": model, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "max_tokens": n,
               "tg_tps": round(tot_n / tot_t, 2) if tot_t else None,
               "pp_tps": round(sum(r["pp_tps"] or 0 for r in rows) / len(rows), 1),
               "draft_acceptance": round(acc / drafted, 3) if drafted else None, "rows": rows}
    with open(OUT, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"{label}: generation {summary['tg_tps']} tok/s (token-weighted), prompt {summary['pp_tps']} tok/s, "
          f"draft acceptance {summary['draft_acceptance']}")

if __name__ == "__main__":
    main()
