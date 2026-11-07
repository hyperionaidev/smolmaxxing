#!/bin/bash
# Loop diagnosis on the TV itself (real pager, RAM limits, kernels): N seeds of ~1000-token answers per config (tools/loop_stats.py).
# Waits for mb-factors.sh to finish. Builds each model on the Mac when needed, pushes it, removes it from the TV afterwards.
cd "$(dirname "$0")/.."
while pgrep -f 'tools/mb-factors.sh' > /dev/null; do sleep 20; done
export URL=http://${TV_IP:-192.168.15.244}:8090 TXT=bench/raw/E105-tv-texts.txt
B=third_party/llama.cpp/build-mac/bin; S=models/moe/mb-q8.gguf; M=models/moe; D=/data/local/tmp/llm
OUT=bench/raw/E105-tv-matrix.txt
stop(){ adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 3; }
up(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health | grep -q ok && return 0; sleep 3; done; echo "server not up" >> $OUT; return 1; }
# srv <model> <env> <args>   (headless mode re-asserted every time: the launcher came back once, E103)
srv(){ stop; adb shell "cd $D; sh tv-llm-mode.sh on > /dev/null 2>&1; sleep 5; am kill-all; sleep 1; (setsid env $2 LIMIT_MB=150 ./tv-watch server.log ./patched40/llama-server -m $1 -t 3 -c 1024 -np 1 --host 0.0.0.0 --port 8090 --metrics --reasoning off -cram 0 -ctxcp 0 -fa on -ctk q8_0 -ctv q8_0 $3 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"; up; }
stats(){ python3 -I tools/loop_stats.py "$1" $2 >> $OUT 2>&1; adb shell "cd $D; echo \"   server: \$(tail -1 server.log | cut -c1-90) top=\$(dumpsys activity activities 2>/dev/null | grep -m1 topResumed | grep -oE 'calendar|launcherx|projengmenu|settings')\"" >> $OUT; }
mk(){ [ -f $M/$1.gguf ] && return 0; $B/llama-quantize --allow-requantize --pure --imatrix $M/imatrix-mb.gguf --token-embedding-type q4_K --output-tensor-type q4_K --tensor-type "_exps\.=$2" $S $M/$1-tmp.gguf q4_K 8 > /dev/null 2>&1 || return 1
  python3 -I tools/gguf_build.py $M/$1-tmp.gguf $M/$1.gguf --head-ids data/head-ids-ling-60k.txt > /dev/null 2>&1; rm -f $M/$1-tmp.gguf; }
PG="GGML_MOE_PAGER=1 GGML_MOE_WARM=$D/moe-warm-moonbeam.txt"
# 1-2: Q2_K, as deployed (repack cache, punched GGUF)
srv mb-q2-h60.gguf "$PG GGML_REPACK_CACHE=$D/cache/mbq2 GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6" "-ub 4 -b 64" && stats "TV Q2 top-6 bias 0.25 (default)" 16
srv mb-q2-h60.gguf "$PG GGML_REPACK_CACHE=$D/cache/mbq2 GGML_MOE_POOL=43 GGML_MOE_BIAS=0"                     "-ub 4 -b 64" && stats "TV Q2 top-8 bias 0" 8
# 3: Q3_K experts (no repack type: experts stay mapped, pager on the GGUF), non-experts repacked in RAM
mk mb-q3-h60 q3_K && adb push $M/mb-q3-h60.gguf $D/ > /dev/null && \
{ srv mb-q3-h60.gguf "$PG GGML_MOE_POOL=33 GGML_MOE_BIAS=0.2" "-ub 4 -b 64" && stats "TV Q3 top-8 bias 0.2" 16
  srv mb-q3-h60.gguf "$PG GGML_MOE_POOL=33 GGML_MOE_BIAS=0"   "-ub 4 -b 64" && stats "TV Q3 top-8 bias 0" 8; }
stop; adb shell "rm -f $D/mb-q3-h60.gguf"; rm -f $M/mb-q3-h60.gguf
# 4: Q4_K experts: repack cache like Q2_K, then the GGUF's expert bytes punched
mk mb-q4-h60 q4_K && python3 -I tools/gguf_exp_ranges.py $M/mb-q4-h60.gguf $M/mb-q4.ranges > /dev/null && adb push $M/mb-q4-h60.gguf $M/mb-q4.ranges $D/ > /dev/null && \
{ adb shell "cd $D; am kill-all; GGML_REPACK_CACHE=$D/cache/mbq4 LIMIT_MB=150 ./tv-watch c-mbq4.log ./patched40/llama-bench -m mb-q4-h60.gguf -t 3 -p 0 -n 1 -r 1 -ctk q8_0 -ctv q8_0 -fa on > /dev/null; ./gguf-punch mb-q4-h60.gguf mb-q4.ranges"
  srv mb-q4-h60.gguf "$PG GGML_REPACK_CACHE=$D/cache/mbq4 GGML_MOE_POOL=26 GGML_MOE_BIAS=0.2" "-ub 4 -b 64" && stats "TV Q4 top-8 bias 0.2" 8; }
stop; adb shell "rm -f $D/mb-q4-h60.gguf $D/mb-q4.ranges $D/cache/mbq4.*"; rm -f $M/mb-q4-h60.gguf $M/mb-q4.ranges
# 5: the published Q8_0 as is: no repack (-nr), all 8 experts, no bias, full head; the pager keeps it within RAM
adb push $S $D/mb-q8.gguf > /dev/null && \
  srv mb-q8.gguf "GGML_MOE_PAGER=1 GGML_MOE_POOL=12 GGML_MOE_BIAS=0" "-nr -ub 4 -b 64" && stats "TV Q8_0 reference (no bias, top-8)" 6
stop; adb shell "rm -f $D/mb-q8.gguf"
echo "TV MATRIX DONE" >> $OUT
