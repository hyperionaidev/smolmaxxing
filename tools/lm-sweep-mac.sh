#!/bin/bash
# §20: Ling-mini-2.0 quality on the Mac with the TV's expert pool simulated (pool / bias / experts), one run at a time, low priority
cd "$(dirname "$0")/.."
M=models/ling/lm2-q2-h60.gguf; W=data/moe-warm-ling.txt
run(){ TAG="$1" env $2 nice -n 15 tools/mac-ppl.sh $M en 2>&1 | tail -1; }
run "nosim" ""
for cfg in "40 0 8" "40 0.2 8" "40 0.3 8" "40 0.2 6" "40 0.3 6" "40 0.4 6"; do
  set -- $cfg
  run "pool$1-b$2-top$3" "GGML_MOE_POOL=$1 GGML_MOE_BIAS=$2 LLAMA_MOE_USED=$3 GGML_MOE_WARM=$W"
done
