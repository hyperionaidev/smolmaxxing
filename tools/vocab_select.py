#!/usr/bin/env python3
"""Choose which vocab rows to keep in a pruned LM head, from token frequencies in real text.

Re-implements the Qwen3.5 tokenizer (byte-level BPE, merges + pre-tokenizer regex from the GGUF header),
tokenizes a corpus, and reports held-out coverage for different kept-set sizes.

usage:
  python3 -I vocab_select.py <model.gguf> tokenize "<text>"           # debug: print token ids
  python3 -I vocab_select.py <model.gguf> stats <corpus files...>       # coverage table
  python3 -I vocab_select.py <model.gguf> write <n_keep> <out.ids> <corpus files...>
  python3 -I vocab_select.py <model.gguf> select <out.ids> --group <n> <files...> [--group <n> <files...>]...
      multi-language budget: always-kept ids + the top <n> train tokens of each group; prints held-out
      coverage per group
  python3 -I vocab_select.py <model.gguf> heldout <out.txt> <max_chars_per_file> <files...>
      writes held-out paragraphs (never used for selection) as evaluation text
Split rule (select / heldout): within each file, every 5th paragraph (i % 5 == 4) is held out.
The .ids file (one id per line, sorted) feeds gguf_build.py --head-ids.
"""
import collections, importlib.util, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gguf_info", os.path.join(HERE, "gguf_info.py"))
gguf_info = importlib.util.module_from_spec(spec); spec.loader.exec_module(gguf_info)

# Qwen3.5 pre-tokenizer (llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35), with \p{L}\p{M} ~ [^\W\d_] and \p{N} ~ \d
L = r"[^\W\d_]"
PRE = re.compile(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|(?:[^\r\n\w]|_)?" + L + r"+"
    r"|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+")

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

class Tok:
    def __init__(self, gguf_path):
        _, meta, _ = gguf_info.parse(gguf_info.load(gguf_path, 64))
        self.tokens = meta["tokenizer.ggml.tokens"]
        self.types = meta["tokenizer.ggml.token_type"]
        self.id = {t: i for i, t in enumerate(self.tokens)}
        self.rank = {tuple(m.split(" ", 1)): i for i, m in enumerate(meta["tokenizer.ggml.merges"])}
        self.b2u = bytes_to_unicode()
        self.cache = {}

    def bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        ids = [self.id[p] for p in parts]
        self.cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for piece in PRE.findall(text):
            out += self.bpe("".join(self.b2u[b] for b in piece.encode("utf-8")))
        return out

def read_corpus(files):
    docs = []
    for fn in files:
        t = open(fn, encoding="utf-8", errors="replace").read()
        docs += [d for d in re.split(r"\n\s*\n", t) if d.strip()]
    return docs

def always_keep(tok):
    return [i for i, t in enumerate(tok.types) if t in (3, 4)] + list(range(256))

def split_file(fn):
    docs = read_corpus([fn])
    return [d for i, d in enumerate(docs) if i % 5 != 4], [d for i, d in enumerate(docs) if i % 5 == 4]

def main_select(tok, out, args):
    groups, cur = [], None
    for a in args:
        if a == "--group": cur = None; continue
        if cur is None: cur = [int(a), []]; groups.append(cur)
        else: cur[1].append(a)
    keep = set(always_keep(tok))
    stats = []
    for n, files in groups:
        train, held = collections.Counter(), collections.Counter()
        for fn in files:
            tr, he = split_file(fn)
            for d in tr: train.update(tok.encode(d))
            for d in he: held.update(tok.encode(d))
        keep |= {i for i, _ in train.most_common(n)}
        stats.append((n, files, held))
    print(f"kept rows: {len(keep)}")
    for n, files, held in stats:
        nh = sum(held.values())
        cov = sum(c for t, c in held.items() if t in keep) / max(nh, 1)
        print(f"  group top-{n:<6d} {nh:8d} held-out tokens  coverage {cov:9.4%}  ({', '.join(os.path.basename(f) for f in files)})")
    open(out, "w").write("\n".join(map(str, sorted(keep))) + "\n")

def main():
    tok = Tok(sys.argv[1]); cmd = sys.argv[2]
    if cmd == "select":
        return main_select(tok, sys.argv[3], sys.argv[4:])
    if cmd == "heldout":
        out, budget = sys.argv[3], int(sys.argv[4])
        with open(out, "w") as f:
            for fn in sys.argv[5:]:
                n = 0
                for d in split_file(fn)[1]:
                    if n > budget: break
                    f.write(d.strip() + "\n\n"); n += len(d)
        print(f"wrote {out}: {os.path.getsize(out)} bytes"); return
    if cmd == "tokenize":
        ids = tok.encode(sys.argv[3]); print(ids); print([tok.tokens[i] for i in ids]); return
    files = sys.argv[3:] if cmd == "stats" else sys.argv[5:]
    docs = read_corpus(files)
    # deterministic split: every 5th paragraph is held out
    train, held = collections.Counter(), collections.Counter()
    for i, d in enumerate(docs):
        (held if i % 5 == 4 else train).update(tok.encode(d))
    nt, nh = sum(train.values()), sum(held.values())
    print(f"corpus: {len(docs)} paragraphs, {nt} train tokens, {nh} held-out tokens, {len(train)} distinct train tokens")
    ranked = [i for i, _ in train.most_common()]
    base = set(always_keep(tok))
    if cmd == "stats":
        print(" n_keep   held-out coverage   (id-rank coverage, first n ids)   head MB (Q6_K, 2048)")
        for n in (16384, 24576, 32768, 40000, 49152, 65536, 81920, 98304):
            keep = base | set(ranked[:n])
            cov = sum(c for t, c in held.items() if t in keep) / nh
            idk = base | set(range(n))
            covi = sum(c for t, c in held.items() if t in idk) / nh
            print(f"{len(keep):7d}   {cov:9.5%}            {covi:9.5%}                        {len(keep) * 1680 / 1e6:6.1f}")
        print(f"all train tokens ({len(base | set(ranked))}): "
              f"{sum(c for t, c in held.items() if t in base | set(ranked)) / nh:.5%}")
    elif cmd == "write":
        n, out = int(sys.argv[3]), sys.argv[4]
        keep = sorted(base | set(ranked[:n]))
        cov = sum(c for t, c in held.items() if t in set(keep)) / nh
        open(out, "w").write("\n".join(map(str, keep)) + "\n")
        print(f"wrote {len(keep)} ids to {out}; held-out coverage {cov:.5%}")

if __name__ == "__main__":
    main()
