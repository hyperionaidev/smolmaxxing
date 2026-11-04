#!/system/bin/sh
# Round-2 follow-ups, run ON THE TV after the low-bit queue:
#  1. check the on-TV head re-cutter (bin/gguf_recut) against the Python tool: H45 from w-q4k must hash
#     a50eadc5... (gguf_build.py --replace-head --head-ids head-ids-ml40k.txt, verified earlier)
#  2. a full-vocab baseline (no pruned head, tied 248K output) so other models compare in bits per byte
#  3. English + multilingual perplexity of the H65 head (the D runs were stopped by the guard at 180 MB)
#  4. M2 (best quality mix) with the H65 head: speed + perplexity -> candidate for the final model
cd /data/local/tmp/llm
S=eval/heads2-summary.txt
ppl() { LIMIT_MB=130 sh tv-guard.sh eval/$2.log ./patched2/llama-perplexity -m $1 -f $3 -c 128 --chunks 40 -t 4 -b 128
        { echo "== $2"; grep -h -E "misses|Final estimate|tv-guard:" eval/$2.log; } >> $S; }
./gguf_recut w-q4k.gguf t-h45.gguf head-ids-ml40k.txt > eval/recut-test.log 2>&1
echo "recut H45: $(sha256sum t-h45.gguf | cut -c1-16) (python: a50eadc5381521445682)" >> $S; rm -f t-h45.gguf
./gguf_recut w-q4k.gguf w-q4k-full.gguf none >> eval/recut-test.log 2>&1
ppl w-q4k-full.gguf F-full-en eval-en.txt
ppl w-q4k-full.gguf F-full-ml eval-ml.txt
rm -f w-q4k-full.gguf
ppl w-q4k-h65.gguf F-h65-en eval-en.txt
ppl w-q4k-h65.gguf F-h65-ml eval-ml.txt
./gguf_recut w-m2.gguf w-m2-h65.gguf head-ids-ml60k.txt >> eval/recut-test.log 2>&1
sh tv-eval.sh w-m2-h65.gguf M2-h65 speed > eval/M2-h65.speed.out 2>&1
{ echo "== M2-h65 speed"; grep -h -E "tg64|pp128" eval/M2-h65.tg.log eval/M2-h65.pp.log; } >> $S
ppl w-m2-h65.gguf F-m2h65-en eval-en.txt
ppl w-m2-h65.gguf F-m2h65-ml eval-ml.txt
echo "ALL DONE $(date)" >> $S
