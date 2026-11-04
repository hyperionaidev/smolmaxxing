#!/system/bin/sh
# Standard evaluation of one model variant, run ON THE TV (in /data/local/tmp/llm), under tv-guard.sh.
#   1. speed: llama-bench tg64 x5 (tokens drawn from 4000 frequent ids, LLAMA_BENCH_COMMON_TOKENS) and pp128 x2,
#      3 threads (generation) / 4 threads (prompt)
#   2. op profile: GGML_OPPROF=1 llama-bench tg32 (where the time goes, per op and weight type)
#   3. quality: llama-perplexity on eval-en.txt (held-out English+code, ctx 128, 40 chunks = 2560 scored tokens;
#      ctx 512 needs a 512 x 248320 float logits buffer = 508 MB and ran the box out of memory)
#      and eval-ml.txt (held-out ar/zh/es/fr/de/ru/ja); misses = target token outside the pruned head
# usage: sh tv-eval.sh <model.gguf> <tag> [speed|ppl|all]     logs: eval/<tag>.*.log
M=$1; T=$2; W=${3:-all}
mkdir -p eval
G="sh tv-guard.sh"
if [ "$W" = speed ] || [ "$W" = all ]; then
  export LLAMA_BENCH_COMMON_TOKENS=4000
  $G eval/$T.tg.log ./patched/llama-bench -m "$M" -t 3 -p 0 -n 64 -r 5
  $G eval/$T.pp.log ./patched/llama-bench -m "$M" -t 4 -p 128 -n 0 -r 2
  GGML_OPPROF=1 $G eval/$T.prof.log ./patched/llama-bench -m "$M" -t 3 -p 0 -n 32 -r 1
  grep -h -E "tg64|pp128" eval/$T.tg.log eval/$T.pp.log
  grep -E "opprof\] (MUL_MAT|GET_ROWS)" eval/$T.prof.log | head -8
fi
if [ "$W" = ppl ] || [ "$W" = all ]; then
  $G eval/$T.ppl-en.log ./patched/llama-perplexity -m "$M" -f eval-en.txt -c 128 --chunks 40 -t 4 -b 128
  $G eval/$T.ppl-ml.log ./patched/llama-perplexity -m "$M" -f eval-ml.txt -c 128 --chunks 40 -t 4 -b 128
  grep -h -E "Final estimate|pruned-head misses" eval/$T.ppl-en.log eval/$T.ppl-ml.log
fi
grep -h "tv-guard" eval/$T.*.log | sort | uniq -c
