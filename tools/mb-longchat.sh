#!/bin/bash
# Moon-Beam long generation on the TV. usage (bash): tools/mb-longchat.sh <bin> <used> <pool> <bias> [extra env] ; prints speed as the context fills
cd "$(dirname "$0")/.."
adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 2
adb forward tcp:8090 tcp:8090 >/dev/null
adb shell "cd /data/local/tmp/llm; am kill-all; sleep 1; (setsid env GGML_MOE_PAGER=1 GGML_MOE_POOL=$3 GGML_MOE_BIAS=$4 LLAMA_MOE_USED=$2 GGML_MOE_WARM=/data/local/tmp/llm/moe-warm-moonbeam.txt GGML_REPACK_CACHE=/data/local/tmp/llm/cache/mbq2 ${5:-} LIMIT_MB=150 ./tv-watch server.log ./$1/llama-server -m mb-q2-h60.gguf -t 3 -c 1024 -np 1 -ub 4 -b 64 --host 0.0.0.0 --port 8090 --metrics --reasoning off -ctk q8_0 -ctv q8_0 -fa on -cram 0 -ctxcp 0 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8090/health | grep -q ok && break; sleep 2; done
python3 -I tools/long_chat.py
adb shell 'sort -k2 -n /data/local/tmp/llm/server.log.mem | head -1; tail -1 /data/local/tmp/llm/server.log.mem'
