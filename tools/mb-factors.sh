#!/bin/bash
# Why does Moon-Beam loop in long answers? One change at a time from the baseline (Q2_K, top-8, bias 0.2, pool 43, temperature 0.7, q8 KV).
# Each case: 3 seeds, ~1000 tokens, repeated 6-grams and where the first sustained loop starts. Waits for mb-loops.sh to finish first.
cd "$(dirname "$0")/.."
while pgrep -f 'mb-loops.sh (tv|bonsai)' > /dev/null; do sleep 15; done
export TXT=bench/raw/E103-factor-texts.txt URL=http://${TV_IP:-192.168.15.244}:8090
stop(){ adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 3; }
wait_up(){ for i in $(seq 1 120); do curl -s -m 2 "$URL/health" | grep -q ok && return 0; sleep 3; done; echo "server not up"; return 1; }
# case <label> <server env> <server args> <client env>
case1(){ L=$1; stop
  # headless mode must hold for the whole run: the launcher came back once and the guard killed runs at 147 MB (E103)
  adb shell "cd /data/local/tmp/llm; sh tv-llm-mode.sh on > /dev/null 2>&1; sleep 5; am kill-all; sleep 1; (setsid env GGML_MOE_PAGER=1 GGML_MOE_POOL=43 GGML_MOE_WARM=/data/local/tmp/llm/moe-warm-moonbeam.txt GGML_REPACK_CACHE=/data/local/tmp/llm/cache/mbq2 $2 LIMIT_MB=150 ./tv-watch server.log ./patched40/llama-server -m mb-q2-h60.gguf -t 3 -c 1024 -np 1 -ub 4 -b 64 --host 0.0.0.0 --port 8090 --metrics --reasoning off -cram 0 -ctxcp 0 -fa on $3 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
  wait_up || { adb shell 'tail -2 /data/local/tmp/llm/server.log' | cut -c1-120; return; }
  for S in 42 43 44; do T=$(adb shell 'dumpsys activity activities 2>/dev/null | grep -m1 topResumed' | grep -oE 'calendar|launcherx|projengmenu|settings' | head -1); echo "$L seed $S [top=$T]: $(env $4 LABEL="$L" SEED=$S python3 -I tools/long_chat.py 2>&1 | grep -E 'server timings|repeated|URLError' | sed -E "s/server timings //; s/'prompt_per_second': [0-9.]+, //; s/'predicted_n': //; s/'predicted_per_second'/tps/" | tr '\n' ' ' | cut -c1-105)"; done
}
Q8="-ctk q8_0 -ctv q8_0"
case1 "bias0 temp1 top20 (GGUF sampling)"     "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0"    "$Q8"  "TEMP=1.0 TOPK=20 TOPP=0.95"
case1 "bias0.2 temp1 top20 (GGUF sampling)"   "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0.2"  "$Q8"  "TEMP=1.0 TOPK=20 TOPP=0.95"
case1 "bias0.1 temp0.7"                       "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0.1"  "$Q8"  "TEMP=0.7"
case1 "bias0.05 temp0.7"                      "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0.05" "$Q8"  "TEMP=0.7"
case1 "bias0.2 KV f16"                        "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0.2"  ""     "TEMP=0.7"
case1 "bias0.2 stock attention"               "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0.2 GGML_FA_Q8=0" "$Q8" "TEMP=0.7"
case1 "bias0 pool 128-free baseline third seed" "LLAMA_MOE_USED=8 GGML_MOE_BIAS=0"  "$Q8"  "TEMP=0.7"
stop; echo FACTORS DONE
