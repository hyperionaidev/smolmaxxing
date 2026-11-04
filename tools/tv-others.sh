#!/system/bin/sh
# Trained 1-bit / ternary models (not Qwen3.5), run ON THE TV after the Qwen queues.
# Per model: speed (tv-eval.sh speed), English perplexity on eval-en.txt (ctx 128 x 40 chunks) + the model's
# token count of eval-en.txt (llama-tokenize) so perplexity can be turned into bits per byte (comparable across
# tokenizers), max RSS / min free RAM (tv-guard), and two greedy answers.
# usage: sh tv-others.sh <tag>:<file.gguf> ...        summary appended to eval/others-summary.txt
cd /data/local/tmp/llm
SUM=eval/others-summary.txt
for spec in "$@"; do
  T=${spec%%:*}; M=${spec#*:}
  [ -f "$M" ] || { echo "$T: missing $M" >> $SUM; continue; }
  sh tv-eval.sh "$M" O-$T speed > eval/O-$T.speed.out 2>&1
  LIMIT_MB=130 sh tv-guard.sh eval/O-$T.ppl-en.log ./patched2/llama-perplexity -m "$M" -f eval-en.txt -c 128 --chunks 40 -t 4 -b 128
  NTOK=$(./patched/llama-tokenize -m "$M" -f eval-en.txt --ids --log-disable 2>/dev/null | tr ',' '\n' | grep -c '[0-9]')
  : > eval/O-$T.gen-all.log
  for P in "What is the capital of Australia, and why was it chosen over Sydney? Answer in two sentences." \
           "Write a Python function that checks if a number is prime, with a short docstring."; do
    sh tv-guard.sh eval/O-$T.gen.log ./patched2/llama-cli -m "$M" -t 3 -c 512 -n 120 --temp 0 -st --reasoning off -p "$P"
    cat eval/O-$T.gen.log >> eval/O-$T.gen-all.log
  done
  {
    echo "== $T  $M  $(ls -l "$M" | awk '{printf "%.0f MB", $5/1048576}')  eval-en tokens: $NTOK (bytes $(wc -c < eval-en.txt))"
    grep -h -E "tg64|pp128" eval/O-$T.tg.log eval/O-$T.pp.log
    grep -h "tv-guard: exit" eval/O-$T.tg.log
    grep -h -E "Final estimate|tv-guard: killed" eval/O-$T.ppl-en.log
  } >> $SUM
  sync
done
echo "ALL DONE $(date)" >> $SUM
