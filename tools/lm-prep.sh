#!/bin/bash
# §20 prep: Ling-mini-2.0 Q2_K experts (Mac has no room for both the full and the pruned-head file, so the pruned one is streamed to the TV)
set -euo pipefail
cd "$(dirname "$0")/.."
S=${ADB_SERIAL:-192.168.15.244:5555}; D=/data/local/tmp/llm
B=third_party/llama.cpp/build-mac/bin; M=models/ling
if [ ! -s $M/lm2-q2.gguf ]; then
  $B/llama-quantize --allow-requantize --pure --imatrix $M/lm2-imatrix.gguf --token-embedding-type q4_K --output-tensor-type q4_K \
    --tensor-type "_exps\.=q2_K" $M/lm2-q4km.gguf $M/lm2-q2.gguf q4_K 8 > bench/raw/E105-lm-quant-q2.log 2>&1
fi
echo "quant ok $(ls -la $M/lm2-q2.gguf)"
python3 -I tools/gguf_build.py $M/lm2-q2.gguf - --head-ids data/head-ids-ling-60k.txt | adb -s $S exec-in "cat > $D/lm2-q2-h60.gguf"
adb -s $S exec-out "head -c 67108864 $D/lm2-q2-h60.gguf" > $M/h60-head.bin
python3 -I tools/gguf_exp_ranges.py $M/h60-head.bin $M/lm2-q2.ranges
adb -s $S shell "ls -la $D/lm2-q2-h60.gguf"
echo PREP DONE
