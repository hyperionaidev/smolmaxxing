#!/bin/bash
# Pre-gating accuracy on real generation (one token at a time), deployed routing simulated, loop fix on.
cd "$(dirname "$0")/.."
B=third_party/llama.cpp/build-mac/bin; M=models/moe; export URL=http://127.0.0.1:8094
R='{"repeat_penalty":1.15,"repeat_last_n":256}'
for K in 6 12; do for AH in 2 3; do
  GGML_MOE_PREGATE_AHEAD=$AH GGML_MOE_PREGATE=$K GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6 $B/llama-server -m $M/mb-q2-h60.gguf -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8094 --reasoning off -cram 0 -fa on -ctk q8_0 -ctv q8_0 > bench/raw/E109-server-$K.log 2>&1 &
  P=$!; for i in $(seq 1 200); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
  SAMPLER="$R" python3 -I tools/loop_stats.py "pre-gate k=$K" 4 500 > /dev/null
  kill $P; wait $P 2>/dev/null; sleep 1
  echo "ahead $AH k=$K: $(grep -h 'pre-gate' bench/raw/E109-server-$K.log | sed 's/.*moe-sim: //')"
done; done; echo PREGATE DONE
