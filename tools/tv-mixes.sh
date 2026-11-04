#!/system/bin/sh
# "Q4_K + a few upgraded sensitive tensors" mixes (middle ground between all-Q4_K and unsloth Q4_K_M, E20).
# Same 32K head as B/E/A for a direct perplexity comparison (A = Q4_K_M 8.687, B = all-Q4_K 8.890 final).
#   M1: all Q4_K + attn_v Q6_K (6 M params) + ffn_down Q5_K in layers 0-3 and 20-23 (the usual "more bits" layers)
#   M2: M1 + ssm_out Q5_K (75 M params)
cd /data/local/tmp/llm
IN=work-bf16-h32k-mtp.gguf
BASE="--pure --token-embedding-type q4_K --output-tensor-type q4_K --tensor-type attn_v=q6_K"
FD='blk\.(0|1|2|3|20|21|22|23)\.ffn_down=q5_K'
SUM=eval/mixes-summary.txt
mix() {
  N=$1; shift
  [ -f w-$N.gguf ] || sh tv-guard.sh eval/X-$N.quant.log sh tv-quant.sh custom $IN w-$N.gguf Q4_K_M "$@"
  sh tv-eval.sh w-$N.gguf X-$N speed > eval/X-$N.speed.out 2>&1
  LIMIT_MB=130 sh tv-guard.sh eval/X-$N.ppl-en.log ./patched2/llama-perplexity -m w-$N.gguf -f eval-en.txt -c 128 --chunks 40 -t 4 -b 128
  { echo "== $N  $(ls -l w-$N.gguf | awk '{printf "%.0f MB", $5/1048576}')"; grep -h -E "tg64|pp128" eval/X-$N.tg.log eval/X-$N.pp.log
    grep -h "Final estimate" eval/X-$N.ppl-en.log; } >> $SUM
}
mix m1 $BASE --tensor-type "$FD"
mix m2 $BASE --tensor-type "$FD" --tensor-type ssm_out=q5_K
echo "ALL DONE $(date)" >> $SUM
