#!/system/bin/sh
# Low-bit sweep (1 / 1.5 / 2 bits per weight) for Qwen3.5-2B, run ON THE TV after the other queues.
# For each variant: quantize from BF16 (unsloth imatrix), speed (tv-eval.sh speed), English perplexity
# (ctx 128 x 40, same 32K head as B/E so it's directly comparable: BF16 8.785 @ chunk 15, Q4_K 8.890 final),
# and two greedy answers to eyeball coherence.
# The embedding table, the head and the MTP block (blk.24) stay Q4_K in every variant: the imatrix has no
# data for them, and llama-quantize refuses IQ1/IQ2 types without it. Only the 24 transformer layers change.
# usage: sh tv-lowbit.sh            (logs in eval/L-<name>.*, summary appended to eval/lowbit-summary.txt)
cd /data/local/tmp/llm
IN=work-bf16-h32k-mtp.gguf
KEEP="--token-embedding-type q4_K --output-tensor-type q4_K --tensor-type blk\.24\.=q4_K"
SUM=eval/lowbit-summary.txt
mkdir -p eval
run() {  # name ftype [extra quantize args]
  N=$1; FT=$2; shift 2
  OUT=w-$N.gguf
  [ -f $OUT ] || sh tv-guard.sh eval/L-$N.quant.log sh tv-quant.sh custom $IN $OUT $FT $KEEP "$@"
  [ -f $OUT ] || { echo "$N: quantize failed" >> $SUM; return; }
  sh tv-eval.sh $OUT L-$N speed > eval/L-$N.speed.out 2>&1
  LIMIT_MB=130 sh tv-guard.sh eval/L-$N.ppl-en.log ./patched2/llama-perplexity -m $OUT -f eval-en.txt -c 128 --chunks 40 -t 4 -b 128
  for P in "What is the capital of Australia, and why was it chosen over Sydney? Answer in two sentences." \
           "Write a Python function that checks if a number is prime, with a short docstring."; do
    sh tv-guard.sh eval/L-$N.gen.log ./patched2/llama-cli -m $OUT -t 3 -c 512 -n 90 --temp 0 -st --reasoning off -p "$P"
    cat eval/L-$N.gen.log >> eval/L-$N.gen-all.log
  done
  {
    echo "== $N ($FT) $(ls -l $OUT | awk '{printf "%.0f MB", $5/1048576}')"
    grep -h -E "tg64|pp128" eval/L-$N.tg.log eval/L-$N.pp.log | sed 's/.*| *\(tg64\|pp128\)/\1/'
    grep -h "quant size" eval/L-$N.quant.log
    tr ',' '\n' < eval/L-$N.ppl-en.log | grep -E '^\[15\]'
    grep -h -E "Final estimate|tv-guard: killed" eval/L-$N.ppl-en.log
  } >> $SUM
  sync
}
# with arguments ("sh tv-lowbit.sh iq1s:IQ1_S"), run only those variants (evaluations re-run, quantize skipped if the file exists)
if [ $# -gt 0 ]; then
  for spec in "$@"; do run ${spec%%:*} ${spec##*:}; done
  echo "ALL DONE (subset: $*) $(date)" >> $SUM
  exit 0
fi
# order: most promising first
run iq2m   IQ2_M      # 2.7 bpw  (llama.cpp mixture: some tensors kept higher)
run iq2xxs IQ2_XXS    # 2.06 bpw
run q2k    Q2_K       # ~2.6 bpw k-quant (no repack kernel on this CPU)
run q20    Q2_0       # 2.25 bpw uniform 2-bit, group 64
run iq1m   IQ1_M      # 1.75 bpw
run iq1s   IQ1_S      # 1.56 bpw
run q10    Q1_0       # 1.125 bpw binary (Bonsai format, meant for 1-bit-trained models; has a repacked dotprod kernel)
echo "ALL DONE $(date)" >> $SUM
