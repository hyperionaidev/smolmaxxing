#!/bin/bash
# Can the sampler stop Q2_K's loops? DRY (penalizes extending a repeat), repeat penalty, presence penalty. 16 seeds each, on the Mac
# (routing simulated by the same hook), then the winners are re-measured on the TV. Waits for mb-matrix.sh.
cd "$(dirname "$0")/.."
while pgrep -f tools/mb-matrix.sh > /dev/null; do sleep 15; done
B=third_party/llama.cpp/build-mac/bin; M=models/moe; export URL=http://127.0.0.1:8092 TXT=bench/raw/E106-dry-texts.txt
[ -f $M/mb-q2-h60.gguf ] || { $B/llama-quantize --allow-requantize --pure --imatrix $M/imatrix-mb.gguf --token-embedding-type q4_K --output-tensor-type q4_K --tensor-type "_exps\.=q2_K" $M/mb-q8.gguf $M/mb-q2-tmp.gguf q4_K 8 > /dev/null 2>&1; python3 -I tools/gguf_build.py $M/mb-q2-tmp.gguf $M/mb-q2-h60.gguf --head-ids data/head-ids-ling-60k.txt > /dev/null 2>&1; rm -f $M/mb-q2-tmp.gguf; }
$B/llama-server -m $M/mb-q2-h60.gguf -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8092 --reasoning off -cram 0 -ctxcp 0 -fa on -ctk q8_0 -ctv q8_0 > bench/raw/E106-server.log 2>&1 &
P=$!; for i in $(seq 1 200); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
# the server's routing settings come from its environment; restart per routing config
run(){ L=$1; ENV=$2; shift 2; kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 2
  env $ENV $B/llama-server -m $M/mb-q2-h60.gguf -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8092 --reasoning off -cram 0 -ctxcp 0 -fa on -ctk q8_0 -ctv q8_0 > bench/raw/E106-server.log 2>&1 &
  P=$!; for i in $(seq 1 200); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
  for smp in "$@"; do SAMPLER="$smp" python3 -I tools/loop_stats.py "$L $smp" 16; done; }
D1='{"dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":-1}'
D2='{"dry_multiplier":1.2,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":-1}'
RP='{"repeat_penalty":1.15,"repeat_last_n":256}'
PP='{"presence_penalty":0.6}'
run "Q2 top-6 b0.25" "GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6" "$D1" "$D2" "$RP" "$PP"
run "Q2 top-8 b0"    "GGML_MOE_POOL=43 GGML_MOE_BIAS=0" "$D1"
kill $P 2>/dev/null; echo DRY DONE
