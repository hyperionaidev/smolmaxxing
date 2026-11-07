#!/bin/bash
# Rebuild every model file the experiments use, from the public Hugging Face sources, straight onto the TV
# (nothing large is stored on the Mac). Model files and repack caches are not in git; this script is how to get
# them back. Each target is independent:
#
#   ADB_SERIAL=<tv-ip>:5555 tools/rebuild-models.sh 2b      # Qwen3.5-2B all-Q4_K + 65K head (the default model, §11.8, §17)
#                                                  2b45    # the same with the 45,654-row head (§17, E68: +3.9 % speed, more misses)
#                                                  08      # Qwen3.5-0.8B all-Q4_K + 65K head (§15, §17; quantized on the Mac)
#                                                  g1      # Granite 4.0 H 1B all-Q4_K (Q8_0 embedding) + 48K head (§15, E62)
#                                                  4b      # Ternary-Bonsai-4B lossless + 61K head (§13)
#                                                  b8      # Bonsai-8B 1-bit + 61K head (§14)
#                                                  tb8     # Ternary-Bonsai-8B + 61K head (§14)
#                                                  q9      # Qwen3.5-9B UD-IQ2_XXS + 65K head + MTP block (§14)
#                                                  mb      # Moon-Beam-8B-A1B: Q2_K experts + 60K head, repack cache, GGUF expert bytes punched (§19)
#                                                  caches  # repack caches for b8 / tb8 (built on the TV by a 1-token run)
#
# Needs: the patched binaries on the TV (the README's quick-start build in /data/local/tmp/llm/patched25; BIN=... for another, bin/gguf_recut to
# /data/local/tmp/llm), python3 (+ numpy via `uv run --with numpy` for fetch_mtp.py), curl, adb.
# Head id lists are in data/ (made by tools/vocab_select.py for Qwen3.5 and tools/head_select_tok.py for Qwen3 / Granite).
set -euo pipefail
cd "$(dirname "$0")/.."
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
D=/data/local/tmp/llm
HF=https://huggingface.co
BIN=${BIN:-patched25}
A(){ adb -s "$S" shell "$@"; }
to_tv(){ adb -s "$S" exec-in "cat > $D/$1"; }       # stdin -> file on the TV
push(){ adb -s "$S" push "$1" "$D/" >/dev/null; }
A "mkdir -p $D/cache"
push data/head-ids-ml60k.txt; push data/head-ids-ml60k-p8.txt; push data/head-ids-ml40k.txt; push data/head-ids-idrank32k.txt; push data/head-ids-qwen3-ml64k.txt
push data/eval/eval-en.txt; push data/eval/eval-ml.txt; push tools/tv-guard.sh; push tools/tv-quant.sh

for T in "$@"; do case "$T" in
mb)
  # §19: experts Q2_K, everything else Q4_K, requantized from the published Q8_0 (8.4 GB on the Mac for the duration) with an imatrix made
  # from data/corpus; 60K head; then on the TV the repack cache, after which the GGUF's expert bytes are only taking up flash.
  B=third_party/llama.cpp/build-mac/bin; M=models/moe; mkdir -p $M
  curl -sL -o $M/mb-q8.gguf "$HF/tablinum/Moon-Beam-8B-A1B-GGUF-Preview/resolve/main/Moon-Beam-8B-A1B-Q8_0.gguf"
  cat data/corpus/wiki_topics.txt data/corpus/book_sherlock.txt | head -c 90000 > $M/imx.txt
  $B/llama-imatrix -m $M/mb-q8.gguf -f $M/imx.txt -o $M/imatrix-mb.gguf -c 256 -b 256 --chunks 40 -t 4 -ngl 0
  $B/llama-quantize --allow-requantize --pure --imatrix $M/imatrix-mb.gguf --token-embedding-type q4_K --output-tensor-type q4_K --tensor-type "_exps\.=q2_K" $M/mb-q8.gguf $M/mb-q2.gguf q4_K 8
  python3 -I tools/gguf_build.py $M/mb-q2.gguf $M/mb-q2-h60.gguf --head-ids data/head-ids-ling-60k.txt && rm -f $M/mb-q2.gguf
  python3 -I tools/gguf_exp_ranges.py $M/mb-q2-h60.gguf $M/mb-q2.ranges
  adb -s "$S" push $M/mb-q2-h60.gguf $M/mb-q2.ranges bin/gguf-punch bin/tv-watch data/moe-warm-moonbeam.txt "$D/"
  A "cd $D && chmod 755 gguf-punch tv-watch && GGML_REPACK_CACHE=$D/cache/mbq2 ./tv-watch c-mb.log ./$BIN/llama-bench -m mb-q2-h60.gguf -t 3 -p 0 -n 1 -r 1 -ctk q8_0 -ctv q8_0 -fa on && ./gguf-punch mb-q2-h60.gguf mb-q2.ranges"
  ;;
2b)
  # 1. MTP block of Qwen3.5-2B from the original BF16 safetensors (range requests, ~120 MB), kept BF16
  uv run --with numpy python3 -I tools/fetch_mtp.py Qwen/Qwen3.5-2B 24 models/mtp-2b-bf16 --bf16
  # 2. unsloth's BF16 GGUF -> + 32K id-rank head + MTP block as blk.24 -> TV (3.9 GB, never stored on the Mac)
  curl -sL "$HF/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-BF16.gguf" \
    | python3 -I tools/gguf_build.py - - --head-ids data/head-ids-idrank32k.txt --add models/mtp-2b-bf16 \
        --set qwen35.block_count=u32:25 --set qwen35.nextn_predict_layers=u32:1 | to_tv work-bf16-h32k-mtp.gguf
  curl -sL "$HF/unsloth/Qwen3.5-2B-GGUF/resolve/main/imatrix_unsloth.gguf_file" | to_tv imatrix_unsloth.gguf
  # 3. on the TV: all-Q4_K with the imatrix (tv-quant.sh caps llama-quantize's buffer: the default OOMs the box),
  #    then swap the 32K head for the 65,668-row multilingual one (gguf_recut re-cuts rows from token_embd)
  A "cd $D && sh tv-guard.sh q-2b.log sh tv-quant.sh q4k work-bf16-h32k-mtp.gguf w-q4k.gguf && ./gguf_recut w-q4k.gguf w-q4k-h65.gguf head-ids-ml60k.txt && rm w-q4k.gguf"
  # 4. §17, E71: the same head plus 4 ids (65,672 rows, a multiple of 8) so the CPU backend repacks it: +5.8 %
  A "cd $D && ./gguf_recut w-q4k-h65.gguf w-q4k-h65p8.gguf head-ids-ml60k-p8.txt"
  ;;
2b45)
  # §17, E68: re-cut the head of the 2B file (needs w-q4k-h65.gguf from the 2b target): 45,654 rows (H45 of E17)
  A "cd $D && ./gguf_recut w-q4k-h65.gguf w-q4k-h45.gguf head-ids-ml40k.txt"
  ;;
08)
  # Qwen3.5-0.8B (§15, E55): unsloth BF16 + the 65,668-row head (same tokenizer as the 2B), pure Q4_K with
  # unsloth's 0.8B imatrix. Quantized on the Mac (needs a native build: third_party/llama.cpp/build-mac/bin/llama-quantize,
  # ~25 s), since the BF16 file (1.65 GB with the head) doesn't fit next to the other models on the TV's flash.
  W=models/small; mkdir -p $W
  curl -sL "$HF/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/imatrix_unsloth.gguf_file" -o $W/imatrix-q08.gguf
  curl -sL "$HF/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-BF16.gguf" \
    | python3 -I tools/gguf_build.py - $W/q08-bf16-h65.gguf --head-ids data/head-ids-ml60k.txt
  third_party/llama.cpp/build-mac/bin/llama-quantize --imatrix $W/imatrix-q08.gguf --pure \
    --token-embedding-type q4_K --output-tensor-type q4_K $W/q08-bf16-h65.gguf $W/q08-q4k-h65.gguf Q4_K_M 8
  rm $W/q08-bf16-h65.gguf
  adb -s "$S" push $W/q08-q4k-h65.gguf "$D/" >/dev/null
  # §17, E72: padded head (65,672 rows) so it is repacked; needs a native build of tools/gguf-recut
  cc -O2 -o $W/gguf_recut tools/gguf-recut/gguf_recut.c && $W/gguf_recut $W/q08-q4k-h65.gguf $W/q08-q4k-h65p8.gguf data/head-ids-ml60k-p8.txt
  adb -s "$S" push $W/q08-q4k-h65p8.gguf "$D/" >/dev/null
  ;;
g1)
  # Granite 4.0 H 1B (§15, E62; hybrid Mamba-2, tied 100K embedding): IBM's Q8_0 -> all-Q4_K with unsloth's imatrix,
  # but token_embd kept at Q8_0 (Q4_K there costs +11.8 % PPL, Q8_0 +3.5 %), then a 49,152-row head cut from it
  # (data/head-ids-granite-48k.txt, tools/head_select_tok.py). Starts from Q8_0 because the 2.9 GB BF16 doesn't fit
  # on the Mac. Needs the Granite pruned-head port (src/models/granite-hybrid.cpp in the patch) and a native llama-quantize.
  W=models/granite; mkdir -p $W
  curl -sL "$HF/unsloth/granite-4.0-h-1b-GGUF/resolve/main/imatrix_unsloth.gguf_file" -o $W/imatrix-granite-h1b.gguf
  curl -sL "$HF/ibm-granite/granite-4.0-h-1b-GGUF/resolve/main/granite-4.0-h-1b-Q8_0.gguf" -o $W/granite-h1b-q8_0.gguf
  third_party/llama.cpp/build-mac/bin/llama-quantize --allow-requantize --pure --token-embedding-type q8_0 \
    --imatrix $W/imatrix-granite-h1b.gguf $W/granite-h1b-q8_0.gguf $W/granite-h1b-q4k-e8-full.gguf Q4_K 8
  rm $W/granite-h1b-q8_0.gguf
  python3 -I tools/gguf_build.py $W/granite-h1b-q4k-e8-full.gguf - --head-ids data/head-ids-granite-48k.txt | to_tv g1-q4k-e8-h48.gguf
  ;;
4b)
  # Ternary-Bonsai-4B (tied embedding): add the 61,592-row frequency head cut from token_embd. Loaded with
  # patched16 it is repacked to 1.875 bits/weight in RAM (E39). The TQ1_0 variant of §13 = llama-quantize
  # --allow-requantize <this> TQ1_0, then the same head.
  curl -sL "$HF/prism-ml/Ternary-Bonsai-4B-gguf/resolve/main/Ternary-Bonsai-4B-Q2_0_g64.gguf" \
    | python3 -I tools/gguf_build.py - - --head-ids data/head-ids-qwen3-ml64k.txt | to_tv tb4-q20-hml.gguf
  ;;
b8)
  # Bonsai-8B has an untied output.weight: gguf_build cuts the kept rows from it
  curl -sL "$HF/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf" \
    | python3 -I tools/gguf_build.py - - --head-ids data/head-ids-qwen3-ml64k.txt | to_tv b8-q1-hml.gguf
  ;;
tb8)
  curl -sL "$HF/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0_g64.gguf" \
    | python3 -I tools/gguf_build.py - - --head-ids data/head-ids-qwen3-ml64k.txt | to_tv tb8-q20-hml.gguf
  ;;
q9)
  # Qwen3.5-9B: MTP block from the BF16 safetensors as Q4_0 (137 MB), unsloth UD-IQ2_XXS, 65K head cut from the
  # untied Q5_K head, block_count 32 + 1 MTP layer
  uv run --with numpy python3 -I tools/fetch_mtp.py Qwen/Qwen3.5-9B 32 models/9b/mtp9-q4 --q4
  curl -sL "$HF/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-IQ2_XXS.gguf" \
    | python3 -I tools/gguf_build.py - - --head-ids data/head-ids-ml60k.txt --add models/9b/mtp9-q4 \
        --set qwen35.block_count=u32:33 --set qwen35.nextn_predict_layers=u32:1 | to_tv q9-iq2-h65-mtp.gguf
  ;;
caches)
  # The repack cache is written by the first run that sets GGML_REPACK_CACHE (one token is enough); later runs map
  # it and skip the repack. The file name gets the buffer size appended (b8c.961840480, tb8.1687042560).
  A "cd $D && GGML_REPACK_CACHE=$D/cache/b8c sh tv-guard.sh c-b8.log ./$BIN/llama-bench -m b8-q1-hml.gguf -t 3 -p 0 -n 1 -r 1"
  A "cd $D && GGML_REPACK_CACHE=$D/cache/tb8 sh tv-guard.sh c-tb8.log ./$BIN/llama-bench -m tb8-q20-hml.gguf -t 3 -p 0 -n 1 -r 1"
  A "ls -la $D/cache"
  ;;
*) echo "unknown target $T (2b 2b45 08 g1 4b b8 tb8 q9 caches)"; exit 1 ;;
esac; done
