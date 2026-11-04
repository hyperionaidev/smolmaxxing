#!/bin/bash
# 4B quant frontier on the Mac (E35): quantize Qwen3.5-4B Q8_0 with the unsloth 4B imatrix into several mixes,
# then English perplexity on eval-en.txt (ctx 128 x 40 chunks, full 248K vocab, same as the TV runs).
# usage: tools/mac-4b-frontier.sh tag:"llama-quantize flags":TYPE ...      results appended to bench/raw/E35-4b-frontier.txt
cd "$(dirname "$0")/.."
B=third_party/llama.cpp/build-mac/bin; M=models/4b; OUT=bench/raw/E35-4b-frontier.txt
for spec in "$@"; do
  FREE=$(df -m / | tail -1 | awk '{print $4}'); [ "$FREE" -lt 1800 ] && { echo "$spec | skipped: only $FREE MB free on the Mac" >> $OUT; continue; }
  T=${spec%%:*}; R=${spec#*:}; ARGS=${R%:*}; TY=${R##*:}
  [ -f $M/q4b-$T.gguf ] || $B/llama-quantize --allow-requantize --imatrix $M/imatrix_unsloth_4b.gguf $ARGS $M/Qwen3.5-4B-Q8_0.gguf $M/q4b-$T.gguf.tmp $TY 8 > $M/q4b-$T.qlog 2>&1 && mv $M/q4b-$T.gguf.tmp $M/q4b-$T.gguf
  P=$($B/llama-perplexity -m $M/q4b-$T.gguf -f data/eval/eval-en.txt -c 128 --chunks 40 -t 6 -b 128 2>&1 | grep "Final estimate")
  SZ=$(grep "quant size" $M/q4b-$T.qlog | sed 's/.*= *//')
  # layer bytes = everything except token_embd (which stays memory-mapped, rows read on demand)
  [ "${KEEP:-0}" = 1 ] || rm -f $M/q4b-$T.gguf
  echo "$T | $ARGS $TY | file $SZ | $P" >> $OUT
done
