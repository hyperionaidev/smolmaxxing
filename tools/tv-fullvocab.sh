#!/system/bin/sh
# Full-vocab (unpruned, tied 248K head) English perplexity of all-Q4_K, for a fair bits-per-byte comparison with
# other models (E25). With repacking it ran the box out of memory (69 MB left, E29), so this streams the weights
# from the page cache (-nr) like the Q4_K_M reference in E20. Slow but fits.
cd /data/local/tmp/llm
S=eval/heads2-summary.txt
./gguf_recut w-q4k.gguf w-q4k-full.gguf none > eval/recut-full.log 2>&1
LIMIT_MB=130 sh tv-guard.sh eval/F-full-en-nr.log ./patched2/llama-perplexity -m w-q4k-full.gguf -f eval-en.txt -c 128 --chunks 40 -t 4 -b 128 -nr
{ echo "== F-full-en (-nr)"; grep -h -E "misses|Final estimate|tv-guard:" eval/F-full-en-nr.log; } >> $S
rm -f w-q4k-full.gguf
