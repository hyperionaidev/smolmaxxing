#!/bin/bash
# The fix, measured on the TV: Q2_K default (top-6, bias 0.25) with repeat penalty 1.15 over 256 tokens. Prose (16 seeds) and code (6 seeds),
# then the same without the penalty for code. Runs right after the TV matrix's Q3 rows; the matrix's Q4 / Q8 steps follow afterwards.
cd "$(dirname "$0")/.."
OUT=bench/raw/E105-tv-matrix.txt; D=/data/local/tmp/llm; export URL=http://${TV_IP:-192.168.15.244}:8090
until grep -q 'TV Q3 top-8 bias 0 ' $OUT 2>/dev/null; do sleep 20; done
pkill -f tools/mb-tvmatrix.sh; pkill -f 'llama-quantize.*mb-q4'; pkill -f 'gguf_build.py.*mb-q4'; sleep 2; rm -f models/moe/mb-q4-h60*.gguf
stop(){ adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 3; }
up(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health | grep -q ok && return 0; sleep 3; done; echo "server not up" >> $OUT; return 1; }
stop; adb shell "rm -f $D/mb-q3-h60.gguf"
adb shell "cd $D; sh tv-llm-mode.sh on > /dev/null 2>&1; sleep 5; am kill-all; sleep 1; (setsid env GGML_MOE_PAGER=1 GGML_MOE_WARM=$D/moe-warm-moonbeam.txt GGML_REPACK_CACHE=$D/cache/mbq2 GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6 LIMIT_MB=150 ./tv-watch server.log ./patched40/llama-server -m mb-q2-h60.gguf -t 3 -c 1024 -np 1 --host 0.0.0.0 --port 8090 --metrics --reasoning off -cram 0 -ctxcp 0 -fa on -ctk q8_0 -ctv q8_0 -ub 4 -b 64 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
up || exit 1
R='{"repeat_penalty":1.15,"repeat_last_n":256}'
TXT=bench/raw/E107-tvfix-texts.txt SAMPLER="$R" python3 -I tools/loop_stats.py "TV Q2 default + rep1.15/256 prose" 16 >> $OUT 2>&1
TXT=bench/raw/E107-tvfix-texts.txt SAMPLER="$R" PROMPTS=code python3 -I tools/loop_stats.py "TV Q2 default + rep1.15/256 code" 6 >> $OUT 2>&1
TXT=bench/raw/E107-tvfix-texts.txt PROMPTS=code python3 -I tools/loop_stats.py "TV Q2 default, no penalty, code" 6 >> $OUT 2>&1
stop; echo "TV FIX DONE" >> $OUT
# the matrix's remaining steps (Q4_K, Q8_0 reference)
sed -n '/^# 4: Q4_K/,$p' tools/mb-tvmatrix.sh > /dev/null
bash -c "$(sed -n '1,/^PG=/p' tools/mb-tvmatrix.sh | sed 's/^while pgrep.*$//'; sed -n '/^# 4: Q4_K/,$p' tools/mb-tvmatrix.sh)"
