#!/bin/bash
# Build Qwen3.6-35B-A3B (35B MoE, 256 experts of 2048x512, top-8 + 1 shared, 40 layers, hybrid Gated DeltaNet) for the expert pager
# on the phone (OnePlus Nord CE 2) without storing the model on the Mac. Same recipe as Ling-mini-2.0 (tools/lm-ipad.sh, TV variant):
#  1. the non-expert tensors of mradermacher's imatrix Q4_K_M (range requests) -> requantized to pure Q4_K with the same imatrix
#  2. LM head pruned to the 60K-row Qwen3.5 list (data/head-ids-ml60k.txt; Qwen3.6 uses the Qwen3.5 tokenizer), or kept whole
#     with HEAD=full: the 60K list drops whole-word tokens a 35B model uses (" palindrome" came out as "palinate", E162)
#  3. experts taken as they are from mradermacher's imatrix Q2_K_S (gate/up Q2_K, down Q2_K in 35 layers and Q4_K in 5),
#     fetched with parallel range requests, written experts-last and streamed over adb to the phone
# usage: tools/q36-phone.sh [base|stream|all]     (ADB_SERIAL selects the phone)
# Other mradermacher imatrix MoE repos work the same way: MR (repo URL), STEM (file stem), TAG (output prefix), e.g. Qwen3-Next-80B-A3B:
#   MR=https://huggingface.co/mradermacher/Qwen3-Next-80B-A3B-Instruct-i1-GGUF/resolve/main STEM=Qwen3-Next-80B-A3B-Instruct TAG=q3n HEAD=full
set -euo pipefail
cd "$(dirname "$0")/.."
S=${ADB_SERIAL:-$(adb get-serialno)}; D=/data/local/tmp/llm
MR=${MR:-https://huggingface.co/mradermacher/Qwen3.6-35B-A3B-i1-GGUF/resolve/main}; STEM=${STEM:-Qwen3.6-35B-A3B}; TAG=${TAG:-q36}
B=third_party/llama.cpp/build-mac/bin; M=models/$TAG; mkdir -p $M
HEAD=${HEAD:-h60}; OUT=${OUT:-$TAG-q2-$HEAD.gguf}
step=${1:-all}

if [[ $step == base || $step == all ]]; then
  [ -f $M/$TAG-imatrix.gguf ] || curl -sfL -o $M/$TAG-imatrix.gguf $MR/$STEM.imatrix.gguf
  python3 -I tools/gguf_fetch_subset.py $MR/$STEM.i1-Q4_K_M.gguf $M/$TAG-ne-q4km.gguf --exclude _exps.
  $B/llama-quantize --allow-requantize --pure --imatrix $M/$TAG-imatrix.gguf --token-embedding-type q4_K --output-tensor-type q4_K \
    $M/$TAG-ne-q4km.gguf $M/$TAG-ne-q4k.gguf q4_K 8
  if [[ $HEAD == full ]]; then mv $M/$TAG-ne-q4k.gguf $M/$TAG-ne-full.gguf
  else python3 -I tools/gguf_build.py $M/$TAG-ne-q4k.gguf $M/$TAG-ne-h60.gguf --head-ids data/head-ids-ml60k.txt; fi
  rm -f $M/$TAG-ne-q4km.gguf $M/$TAG-ne-q4k.gguf
fi
if [[ $step == stream || $step == all ]]; then
  python3 -I tools/gguf_moe_assemble.py $M/$TAG-ne-$HEAD.gguf $MR/$STEM.i1-Q2_K_S.gguf - --jobs 10 \
    | adb -s "$S" shell "cat > $D/$OUT.part"
  adb -s "$S" shell "cd $D && md5sum $OUT.part && mv $OUT.part $OUT" </dev/null
fi
