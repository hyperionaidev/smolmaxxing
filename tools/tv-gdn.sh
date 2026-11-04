#!/system/bin/sh
# E28: fused one-pass DeltaNet kernel + state written straight into the recurrent cache (patched3/).
# Stock behaviour of the same binary: GGML_GDN_STOCK=1 (5-pass kernel) LLAMA_GDN_COPY=1 (separate state copy).
# usage: sh tv-gdn.sh [model]     results: eval/gdn-summary.txt
cd /data/local/tmp/llm
M=${1:-w-q4k-h65.gguf}
S=eval/gdn-summary.txt
B=./patched3
echo "== E28 $M $(date)" >> $S
# 1. same numbers? (prompt path: 128-token batches through the multi-token loop)
for mode in stock new stock-kernel-only; do
  case $mode in stock) E="GGML_GDN_STOCK=1 LLAMA_GDN_COPY=1";; new) E="";; stock-kernel-only) E="GGML_GDN_STOCK=1";; esac
  env $E LIMIT_MB=130 sh tv-guard.sh eval/gdn-ppl-$mode.log $B/llama-perplexity -m $M -f eval-en.txt -c 128 --chunks 12 -t 4 -b 128
  echo "ppl $mode: $(grep -h 'Final estimate' eval/gdn-ppl-$mode.log)" >> $S
done
# 2. same text? (generation path: one token at a time)
for mode in stock new; do
  case $mode in stock) E="GGML_GDN_STOCK=1 LLAMA_GDN_COPY=1";; new) E="";; esac
  env $E sh tv-guard.sh eval/gdn-gen-$mode.log $B/llama-cli -m $M -t 3 -c 512 -n 80 --temp 0 -st --reasoning off \
    -p "Explain in three sentences why the sky is blue."
  sed -n '/^> /,/^\[ Prompt/p' eval/gdn-gen-$mode.log | grep -v '^\[ Prompt' > eval/gdn-gen-$mode.txt
done
cmp -s eval/gdn-gen-stock.txt eval/gdn-gen-new.txt && echo "gen: identical text" >> $S || echo "gen: TEXT DIFFERS (see eval/gdn-gen-*.txt)" >> $S
# 3. speed, alternating to cancel drift
for r in 1 2; do
  for mode in stock new; do
    case $mode in stock) E="GGML_GDN_STOCK=1 LLAMA_GDN_COPY=1";; new) E="";; esac
    env $E LLAMA_BENCH_COMMON_TOKENS=4000 $B/llama-bench -m $M -t 3 -p 0 -n 64 -r 5 > eval/gdn-tg-$mode-$r.log 2>&1
    echo "tg $mode run$r: $(grep tg64 eval/gdn-tg-$mode-$r.log | awk -F'|' '{print $(NF-1)}')" >> $S
  done
done
for mode in stock new; do
  case $mode in stock) E="GGML_GDN_STOCK=1 LLAMA_GDN_COPY=1";; new) E="";; esac
  env $E LLAMA_BENCH_COMMON_TOKENS=4000 $B/llama-bench -m $M -t 4 -p 128 -n 0 -r 3 > eval/gdn-pp-$mode.log 2>&1
  echo "pp $mode: $(grep pp128 eval/gdn-pp-$mode.log | awk -F'|' '{print $(NF-1)}')" >> $S
  env $E GGML_OPPROF=1 LLAMA_BENCH_COMMON_TOKENS=4000 $B/llama-bench -m $M -t 3 -p 0 -n 32 -r 1 > eval/gdn-prof-$mode.log 2>&1
  { echo "--- opprof $mode"; grep "opprof" eval/gdn-prof-$mode.log | head -12; } >> $S
done
echo "ALL DONE $(date)" >> $S
