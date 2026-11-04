#!/usr/bin/env python3
"""Pick the rows of a pruned LM head for any GGUF model, using llama.cpp's own tokenizer (llama-tokenize),
so it works for tokenizers vocab_select.py doesn't re-implement (e.g. Qwen3 / Bonsai).

Same recipe as the Qwen3.5 H65 head (E17): always keep ids < ALWAYS (default 16384) and every id >= the
model's first special token... plus, per group, its most frequent train tokens; the per-group budget n is
raised until the union reaches the target size. Train/held-out split as vocab_select.py: within each file,
every 5th blank-line paragraph (i % 5 == 4) is held out and never counted.

usage: python3 -I head_select_tok.py <llama-tokenize> <model.gguf> <target_rows> <out.ids> <n_vocab> \
           --group <files...> [--group <files...>]... [--always N] [--special-from ID]
"""
import collections, os, subprocess, sys, tempfile


def train_text(path):
    paras = [p for p in open(path, encoding="utf-8", errors="replace").read().split("\n\n") if p.strip()]
    return "\n\n".join(p for i, p in enumerate(paras) if i % 5 != 4)


def token_counts(tok, model, files):
    c = collections.Counter()
    for f in files:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as t:
            t.write(train_text(f)); name = t.name
        out = subprocess.run([tok, "-m", model, "-f", name, "--ids", "--log-disable", "--no-bos"],
                             capture_output=True, text=True).stdout
        os.unlink(name)
        ids = [int(x) for x in out.strip().strip("[]").replace(" ", "").split(",") if x]
        c.update(ids)
    return c


def main():
    tok, model, target, outp, n_vocab = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    groups, always, special_from, a = [], 16384, None, sys.argv[6:]
    i = 0
    while i < len(a):
        if a[i] == "--group":
            g = []; i += 1
            while i < len(a) and not a[i].startswith("--"): g.append(a[i]); i += 1
            groups.append(g)
        elif a[i] == "--always": always = int(a[i + 1]); i += 2
        elif a[i] == "--special-from": special_from = int(a[i + 1]); i += 2
        else: sys.exit("bad arg " + a[i])
    base = set(range(min(always, n_vocab)))
    if special_from is not None: base |= set(range(special_from, n_vocab))
    ranked = []
    for g in groups:
        c = token_counts(tok, model, g)
        ranked.append([t for t, _ in c.most_common()])
        print(f"group {os.path.basename(g[0])}..: {sum(c.values())} train tokens, {len(c)} distinct", file=sys.stderr)
    lo, hi = 0, max(len(r) for r in ranked)
    pick = lambda n: base.union(*[set(r[:n]) for r in ranked])
    while lo < hi:                       # smallest per-group budget n whose union reaches the target
        mid = (lo + hi) // 2
        if len(pick(mid)) >= target: hi = mid
        else: lo = mid + 1
    keep = sorted(pick(lo))
    open(outp, "w").write("\n".join(map(str, keep)) + "\n")
    print(f"per-group budget {lo}: {len(keep)} rows -> {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
