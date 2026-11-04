#!/system/bin/sh
# Quantize recipes for Qwen3.5-2B, run ON THE TV (llama-quantize from patches/, imatrix from unsloth).
# usage (in /data/local/tmp/llm):  sh tv-quant.sh <recipe> <in-bf16.gguf> <out.gguf>
#
# Input is a BF16 GGUF built by tools/gguf_build.py (pruned head rows as output.weight + output_ids,
# optionally the MTP block as blk.24). Every recipe also sets the type of the head (output.weight),
# the input embedding table (token_embd, memory-mapped, only one row read per token) and the MTP block.
set -e
R=$1; IN=$2; OUT=$3
Q="./patched/llama-quantize --max-buffer-size 64"   # default 8 GB slab buffer OOMs the box (kernel panic)
IM="--imatrix imatrix_unsloth.gguf"
case "$R" in
  # one type everywhere ("pure"): measures each kernel family on its own
  q4k)      $Q $IM --pure --token-embedding-type q4_K --output-tensor-type q4_K "$IN" "$OUT" Q4_K_M 4 ;;
  q40)      $Q $IM --pure --token-embedding-type q4_0 --output-tensor-type q4_0 "$IN" "$OUT" Q4_0 4 ;;
  iq4nl)    $Q $IM --pure --token-embedding-type iq4_nl --output-tensor-type iq4_nl "$IN" "$OUT" IQ4_NL 4 ;;
  q4k-h6)   $Q $IM --pure --token-embedding-type q4_K --output-tensor-type q6_K "$IN" "$OUT" Q4_K_M 4 ;;
  q4k-h8)   $Q $IM --pure --token-embedding-type q4_K --output-tensor-type q8_0 "$IN" "$OUT" Q4_K_M 4 ;;
  # stock llama.cpp mixes, for reference
  q4km)     $Q $IM "$IN" "$OUT" Q4_K_M 4 ;;
  q4ks)     $Q $IM "$IN" "$OUT" Q4_K_S 4 ;;
  q3km)     $Q $IM "$IN" "$OUT" Q3_K_M 4 ;;
  # custom: sh tv-quant.sh custom IN OUT FTYPE [llama-quantize options...]
  custom)   FT=$4; shift 4; $Q $IM "$@" "$IN" "$OUT" "$FT" 4 ;;
  *) echo "unknown recipe $R"; exit 1 ;;
esac
ls -la "$OUT"
