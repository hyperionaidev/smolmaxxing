#!/bin/bash
# Pre-gated expert prefetch on the TV (E110): verify, then speed vs no prefetch. Deployed config + loop fix. Starts after the fix test.
cd "$(dirname "$0")/.."
OUT=bench/raw/E110-tv-prefetch.txt; D=/data/local/tmp/llm; export URL=http://${TV_IP:-192.168.15.244}:8090
true
# the Q4_K / Q8_0 reference runs are deferred: the TV is needed for speed work (the cause of the loops is established)
true
adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; p=$(pidof llama-bench); [ -n "$p" ] && kill $p; rm -f /data/local/tmp/llm/mb-q4-h60.gguf /data/local/tmp/llm/mb-q4.ranges /data/local/tmp/llm/cache/mbq4.*; true'
adb shell "mkdir -p $D/patched41"; adb push bin/patched41/llama-server bin/patched41/llama-bench $D/patched41/ > /dev/null; adb shell "chmod 755 $D/patched41/*"
stop(){ adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 4; }
up(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health | grep -q ok && return 0; sleep 3; done; echo "server not up" >> $OUT; return 1; }
srv(){ stop; adb shell "cd $D; sh tv-llm-mode.sh on > /dev/null 2>&1; sleep 5; am kill-all; sleep 1; (setsid env GGML_MOE_PAGER=1 GGML_MOE_STATS=1 GGML_MOE_WARM=$D/moe-warm-moonbeam.txt GGML_REPACK_CACHE=$D/cache/mbq2 GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6 $1 LIMIT_MB=150 ./tv-watch server.log ./patched41/llama-server -m mb-q2-h60.gguf -t 3 -c 1024 -np 1 --host 0.0.0.0 --port 8090 --metrics --reasoning off -cram 0 -ctxcp 0 -fa on -ctk q8_0 -ctv q8_0 -ub 4 -b 64 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"; up; }
pstats(){ stop; adb shell "grep -E 'moe-pager|tv-guard' $D/server.log | tail -2 | cut -c1-220" >> $OUT; }
R='{"repeat_penalty":1.15,"repeat_last_n":256}'
srv "GGML_MOE_VERIFY=1 GGML_MOE_PREFETCH_K=6 GGML_MOE_PREFETCH_AHEAD=2" && SAMPLER="$R" python3 -I tools/loop_stats.py "VERIFY prefetch k=6 ahead 2 (decision-aware)" 2 400 >> $OUT 2>&1; pstats
for cfg in "GGML_MOE_PREFETCH_K=6 GGML_MOE_PREFETCH_AHEAD=2" "GGML_MOE_PREFETCH_K=6 GGML_MOE_PREFETCH_AHEAD=1" "GGML_MOE_PREFETCH_K=0"; do
  srv "$cfg" && TXT=bench/raw/E110-texts.txt SAMPLER="$R" python3 -I tools/loop_stats.py "TV default+fix, $cfg" 8 >> $OUT 2>&1; pstats
done
stop; echo "PREFETCH DONE" >> $OUT
