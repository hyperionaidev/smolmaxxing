#!/bin/sh
# Quality of the Kindle's model files, measured on the Mac with the repo's usual recipe (tools/mac-ppl.sh):
# llama-perplexity on data/eval/eval-en.txt, 40 chunks of 128 tokens, and bits/byte = log2(PPL) * tokens / bytes.
# The Kindle would need hours per model for this (it reads prompts at 1-4 tok/s), and the file scores the same anywhere.
# usage: kindle/mac-ppl.sh <model.gguf> ...   (LLAMA_BIN = a stock llama.cpp v0.7.0 build for the Mac)
# Appends to bench/raw/E168-kindle-ppl.txt
set -eu
cd "$(dirname "$0")/.."
B=${LLAMA_BIN:-third_party/llama.cpp/build-mac/bin}
F=data/eval/eval-en.txt; SZ=$(wc -c < $F | tr -d ' ')
mkdir -p bench/raw
for M in "$@"; do
  O=$("$B/llama-perplexity" -m "$M" -f $F -c 128 --chunks 40 -t 4 -b 128 -ngl 0 2>&1 | grep -E 'Final estimate' | sed 's/.*I //')
  P=$(echo "$O" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  N=$("$B/llama-tokenize" -m "$M" -f $F --ids --log-disable 2>/dev/null | tr ',' '\n' | grep -c '[0-9]')
  R=$(python3 -I -c "import math;print(round(math.log2($P)*$N/$SZ,4))")
  echo "$(date +%F) $(basename "$M") en PPL $P tokens $N bits/byte $R" | tee -a bench/raw/E168-kindle-ppl.txt
done
