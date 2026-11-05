#!/bin/bash
# Screening of small (~0.3-2B) models for the 0.8B speed push (EXPERIMENT.md §15), run FROM THE MAC.
# Per model: stream the GGUF from Hugging Face straight to the TV (the Mac has ~5 GB free), then on the TV,
# under tv-guard.sh: llama-bench tg64 x3 (frequent tokens) + pp128, English perplexity on eval-en.txt
# (ctx 128 x 40 chunks), the model's token count of eval-en.txt (-> bits per byte), one greedy answer.
# KEEP=1 keeps the file on the TV; otherwise it is removed after the run (TV disk is ~2.4 GB free).
# usage: tools/screen-small.sh <tag> <hf-repo> <file> [<tag> <repo> <file> ...]
set -uo pipefail
S=${ADB_SERIAL:-192.168.15.244:5555}
D=/data/local/tmp/llm
BIN=${BIN:-patched16}
OUT=bench/raw/E54-screen
mkdir -p $OUT
while [ $# -ge 3 ]; do
  T=$1 R=$2 F=$3; shift 3
  echo "== $T ($R/$F) $(date +%T)"
  if ! adb -s $S shell "[ -f $D/s-$T.gguf ]"; then
    curl -sfL "https://huggingface.co/$R/resolve/main/$F" | adb -s $S exec-in "cat > $D/s-$T.gguf.part" \
      && adb -s $S shell "mv $D/s-$T.gguf.part $D/s-$T.gguf" || { echo "download failed"; continue; }
  fi
  adb -s $S shell "cd $D && ls -l s-$T.gguf | awk '{print \$5}' && export LLAMA_BENCH_COMMON_TOKENS=4000 &&
    sh tv-guard.sh e08/$T.tg.log ./$BIN/llama-bench -m s-$T.gguf -t 3 -p 0 -n 64 -r 3 &&
    sh tv-guard.sh e08/$T.pp.log ./$BIN/llama-bench -m s-$T.gguf -t 3 -p 128 -n 0 -r 1 &&
    GGML_OPPROF=1 sh tv-guard.sh e08/$T.prof.log ./$BIN/llama-bench -m s-$T.gguf -t 3 -p 0 -n 16 -r 1;
    LIMIT_MB=130 sh tv-guard.sh e08/$T.ppl.log ./$BIN/llama-perplexity -m s-$T.gguf -f eval-en.txt -c 128 --chunks 40 -t 4 -b 128;
    ./patched/llama-tokenize -m s-$T.gguf -f eval-en.txt --ids --log-disable 2>/dev/null | tr ',' '\n' | grep -c '[0-9]' > e08/$T.ntok;
    sh tv-guard.sh e08/$T.gen.log ./$BIN/llama-cli -m s-$T.gguf -t 3 -c 512 -n 80 --temp 0 -st --reasoning off \
      -p 'What is the capital of Australia, and why was it chosen over Sydney? Answer in two sentences.'" > $OUT/$T.run.txt 2>&1
  adb -s $S shell "cd $D && cat e08/$T.tg.log e08/$T.pp.log e08/$T.ppl.log e08/$T.prof.log e08/$T.gen.log; echo NTOK \$(cat e08/$T.ntok)" > $OUT/$T.log 2>&1
  python3 -I - "$OUT/$T.log" "$T" <<'EOF' | tee -a $OUT/summary.txt
import re, sys, math
t = open(sys.argv[1], errors="replace").read()
tg = re.findall(r"tg64 \|\s+([\d.]+) ± ([\d.]+)", t); pp = re.findall(r"pp128 \|\s+([\d.]+)", t)
ppl = re.findall(r"Final estimate: PPL = ([\d.]+)", t); nt = re.findall(r"NTOK (\d+)", t)
rss = re.findall(r"max RSS (\d+) MB", t)
bpb = math.log2(float(ppl[0])) * int(nt[0]) / 22031 if ppl and nt and int(nt[0]) else float("nan")
print(f"{sys.argv[2]:<14} tg {tg[0][0] if tg else '-':>6} pp {pp[0] if pp else '-':>6}  PPL {ppl[0] if ppl else '-':>8}  tokens {nt[0] if nt else '-':>5}  bits/byte {bpb:.3f}  maxRSS {max(map(int, rss)) if rss else '-'} MB")
EOF
  [ "${KEEP:-0}" = 1 ] || adb -s $S shell "rm -f $D/s-$T.gguf"
done
