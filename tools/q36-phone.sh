#!/bin/bash
# Build Qwen3.6-35B-A3B (35B MoE, 256 experts of 2048x512, top-8 + 1 shared, 40 layers, hybrid Gated DeltaNet) for the expert pager
# on the phone (OnePlus Nord CE 2) without storing the model on the Mac. Same recipe as Ling-mini-2.0 (tools/lm-ipad.sh, TV variant):
#  1. the non-expert tensors of mradermacher's imatrix Q4_K_M (range requests) -> requantized to pure Q4_K with the same imatrix
#  2. LM head pruned to the 60K-row Qwen3.5 list (data/head-ids-ml60k.txt; Qwen3.6 uses the Qwen3.5 tokenizer), or kept whole
#     with HEAD=full: the 60K list drops whole-word tokens a 35B model uses (" palindrome" came out as "palinate", E162)
#  3. experts taken as they are from mradermacher's imatrix Q2_K_S (gate/up Q2_K, down Q2_K in 35 layers and Q4_K in 5),
#     fetched with parallel range requests, written experts-last and streamed over adb to the phone
# usage: tools/q36-phone.sh [base|stream|all]     (ADB_SERIAL selects the phone)
set -euo pipefail
cd "$(dirname "$0")/.."
S=${ADB_SERIAL:-$(adb get-serialno)}; D=/data/local/tmp/llm
B=third_party/llama.cpp/build-mac/bin; M=models/q36; mkdir -p $M
MR=https://huggingface.co/mradermacher/Qwen3.6-35B-A3B-i1-GGUF/resolve/main
HEAD=${HEAD:-h60}; OUT=${OUT:-q36-q2-$HEAD.gguf}
step=${1:-all}

if [[ $step == base || $step == all ]]; then
  [ -f $M/q36-imatrix.gguf ] || curl -sfL -o $M/q36-imatrix.gguf $MR/Qwen3.6-35B-A3B.imatrix.gguf
  python3 -I tools/gguf_fetch_subset.py $MR/Qwen3.6-35B-A3B.i1-Q4_K_M.gguf $M/q36-ne-q4km.gguf --exclude _exps.
  $B/llama-quantize --allow-requantize --pure --imatrix $M/q36-imatrix.gguf --token-embedding-type q4_K --output-tensor-type q4_K \
    $M/q36-ne-q4km.gguf $M/q36-ne-q4k.gguf q4_K 8
  if [[ $HEAD == full ]]; then mv $M/q36-ne-q4k.gguf $M/q36-ne-full.gguf
  else python3 -I tools/gguf_build.py $M/q36-ne-q4k.gguf $M/q36-ne-h60.gguf --head-ids data/head-ids-ml60k.txt; fi
  rm -f $M/q36-ne-q4km.gguf $M/q36-ne-q4k.gguf
fi
if [[ $step == stream || $step == all ]]; then
  python3 -I tools/gguf_moe_assemble.py $M/q36-ne-$HEAD.gguf $MR/Qwen3.6-35B-A3B.i1-Q2_K_S.gguf - --jobs 10 \
    | adb -s "$S" shell "cat > $D/$OUT.part"
  adb -s "$S" shell "cd $D && md5sum $OUT.part && mv $OUT.part $OUT" </dev/null
fi
