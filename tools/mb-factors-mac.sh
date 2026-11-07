#!/bin/bash
# Mac side of the loop diagnosis: expert bit width and head, with routing simulated by the same hook. 3 seeds, ~1000 tokens, as on the TV.
cd "$(dirname "$0")/.."
B=third_party/llama.cpp/build-mac/bin; S=models/moe/mb-q8.gguf; M=models/moe
export TXT=bench/raw/E103-factor-texts.txt URL=http://127.0.0.1:8092
# let the reference finish its second seed, then free the Mac for these
while pgrep -f 'mb-loops.sh ref' > /dev/null && [ "$(grep -c 'reference seed' bench/raw/E102-loops-mac.txt)" -lt 2 ]; do sleep 20; done
pkill -f 'mb-loops.sh ref'; pkill -f 'mb-q8.gguf -t 4 -ngl 0 -nr'; sleep 3
mk(){ [ -f $M/$1.gguf ] && return; $B/llama-quantize --allow-requantize --pure --imatrix $M/imatrix-mb.gguf --token-embedding-type q4_K --output-tensor-type q4_K --tensor-type "_exps\.=$2" $S $M/$1-tmp.gguf q4_K 8 > /dev/null 2>&1 || { echo "quant $1 failed"; return 1; }
  if [ "$3" = h60 ]; then python3 -I tools/gguf_build.py $M/$1-tmp.gguf $M/$1.gguf --head-ids data/head-ids-ling-60k.txt > /dev/null 2>&1; rm -f $M/$1-tmp.gguf; else mv $M/$1-tmp.gguf $M/$1.gguf; fi; }
case1(){ L=$1; F=$2; shift 2
  env "$@" $B/llama-server -m $F -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8092 --reasoning off -cram 0 -ctxcp 0 -fa on > bench/raw/E103-mac-server.log 2>&1 &
  P=$!; for i in $(seq 1 120); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
  for Sd in 42 43 44; do echo "$L seed $Sd: $(LABEL="$L" SEED=$Sd python3 -I tools/long_chat.py 2>&1 | grep -E 'server timings|repeated|URLError' | sed -E "s/server timings //; s/'prompt_per_second': [0-9.]+, //; s/'predicted_n': //; s/'predicted_per_second'/tps/" | tr '\n' ' ' | cut -c1-105)"; done
  kill $P; sleep 3; }
rm -f $M/mb-q2-full.gguf
mk mb-q3-h60 q3_K h60
case1 "Q3 h60 top-8 bias 0"   $M/mb-q3-h60.gguf GGML_MOE_POOL=33 GGML_MOE_BIAS=0
case1 "Q3 h60 top-8 bias 0.2" $M/mb-q3-h60.gguf GGML_MOE_POOL=33 GGML_MOE_BIAS=0.2
case1 "Q3 h60 top-6 bias 0.2" $M/mb-q3-h60.gguf GGML_MOE_POOL=33 GGML_MOE_BIAS=0.2 LLAMA_MOE_USED=6
rm -f $M/mb-q3-h60.gguf
mk mb-q4-h60 q4_K h60
case1 "Q4 h60 top-8 bias 0"   $M/mb-q4-h60.gguf GGML_MOE_POOL=26 GGML_MOE_BIAS=0
case1 "Q4 h60 top-8 bias 0.2" $M/mb-q4-h60.gguf GGML_MOE_POOL=26 GGML_MOE_BIAS=0.2
rm -f $M/mb-q4-h60.gguf
echo MAC-FACTORS DONE
