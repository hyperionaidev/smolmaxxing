#!/bin/bash
# Quality on the Mac, same recipe as every earlier row: PPL on data/eval/eval-en.txt (and -ml), bits/byte = log2(PPL) * tokens / bytes.
# usage: tools/mac-ppl.sh <model.gguf> [en|ml|both] (environment passes to llama-perplexity)   Appends to bench/raw/mac-ppl.txt
set -u
cd "$(dirname "$0")/.."
B=third_party/llama.cpp/build-mac/bin
M=$1; W=${2:-en}
for L in en ml; do
  [ "$W" = both ] || [ "$W" = $L ] || continue
  F=data/eval/eval-$L.txt; SZ=$(stat -f %z $F)
  O=$($B/llama-perplexity -m "$M" -f $F -c 128 --chunks 40 -t 4 -b 128 -ngl 0 2>&1 | grep -E 'Final estimate|pruned-head misses' | sed 's/.*I //' | tr '\n' ' ')
  P=$(echo "$O" | sed -E 's/.*PPL = ([0-9.]+).*/\1/')
  N=$($B/llama-tokenize -m "$M" -f $F --ids --log-disable 2>/dev/null | tr ',' '\n' | grep -c '[0-9]')
  R=$(python3 -I -c "import math;print(round(math.log2($P)*$N/$SZ,4))")
  echo "$(date +%T) $(basename $M) ${TAG:-} $L PPL $P tokens $N bits/byte $R | $O" | cut -c1-260 | tee -a bench/raw/mac-ppl.txt
done
