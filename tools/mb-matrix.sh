#!/bin/bash
# Moon-Beam loop diagnosis, large: 16 seeds per config on the Mac (40-50 tok/s), one factor at a time. Waits for mb-factors-mac.sh.
cd "$(dirname "$0")/.."
while pgrep -f mb-factors-mac.sh > /dev/null; do sleep 15; done
B=third_party/llama.cpp/build-mac/bin; S=models/moe/mb-q8.gguf; M=models/moe; export URL=http://127.0.0.1:8092 TXT=bench/raw/E104-matrix-texts.txt
mk(){ [ -f $M/$1.gguf ] && return 0; $B/llama-quantize --allow-requantize --pure --imatrix $M/imatrix-mb.gguf --token-embedding-type q4_K --output-tensor-type q4_K --tensor-type "_exps\.=$2" $S $M/$1-tmp.gguf q4_K 8 > /dev/null 2>&1 || { echo "quant $1 failed"; return 1; }
  python3 -I tools/gguf_build.py $M/$1-tmp.gguf $M/$1.gguf --head-ids data/head-ids-ling-60k.txt > /dev/null 2>&1; rm -f $M/$1-tmp.gguf; }
# one <label> <gguf> <env> <server args> [seeds]
one(){ env $3 $B/llama-server -m $2 -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8092 --reasoning off -cram 0 -ctxcp 0 -fa on $4 > bench/raw/E104-server.log 2>&1 &
  P=$!; for i in $(seq 1 200); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
  python3 -I tools/loop_stats.py "$1" ${5:-16}; kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 2; }
Q8K="-ctk q8_0 -ctv q8_0"
mk mb-q2-h60 q2_K
one "Q2 top-8 bias 0 f16"       $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0"        ""
one "Q2 top-8 bias 0 q8KV"      $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0"        "$Q8K"
one "Q2 top-8 bias 0.2 q8KV"    $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0.2"      "$Q8K"
one "Q2 top-8 bias 0.2 f16"     $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0.2"      ""
one "Q2 top-8 bias 0.05 q8KV"   $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0.05"     "$Q8K"
one "Q2 top-8 bias 0.1 q8KV"    $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0.1"      "$Q8K"
one "Q2 top-6 bias 0.25 q8KV"   $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6" "$Q8K"
one "Q2 top-6 bias 0 q8KV"      $M/mb-q2-h60.gguf "GGML_MOE_POOL=43 GGML_MOE_BIAS=0 LLAMA_MOE_USED=6"    "$Q8K"
rm -f $M/mb-q2-h60.gguf
mk mb-q3-h60 q3_K
one "Q3 top-8 bias 0 f16"       $M/mb-q3-h60.gguf "GGML_MOE_POOL=33 GGML_MOE_BIAS=0"        ""
one "Q3 top-8 bias 0.2 f16"     $M/mb-q3-h60.gguf "GGML_MOE_POOL=33 GGML_MOE_BIAS=0.2"      ""
one "Q3 top-8 bias 0.2 q8KV"    $M/mb-q3-h60.gguf "GGML_MOE_POOL=33 GGML_MOE_BIAS=0.2"      "$Q8K"
one "Q3 top-6 bias 0.2 q8KV"    $M/mb-q3-h60.gguf "GGML_MOE_POOL=33 GGML_MOE_BIAS=0.2 LLAMA_MOE_USED=6"  "$Q8K"
rm -f $M/mb-q3-h60.gguf
mk mb-q4-h60 q4_K
one "Q4 top-8 bias 0 f16"       $M/mb-q4-h60.gguf "GGML_MOE_POOL=26 GGML_MOE_BIAS=0"        ""
one "Q4 top-8 bias 0.2 q8KV"    $M/mb-q4-h60.gguf "GGML_MOE_POOL=26 GGML_MOE_BIAS=0.2"      "$Q8K"
rm -f $M/mb-q4-h60.gguf
one "Q8 reference"              $S                 "X=1"                                     "" 6
echo MATRIX DONE
