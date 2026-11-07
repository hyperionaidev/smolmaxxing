#!/bin/bash
# Bonsai-8B long-generation test on the TV: speed as the context fills. usage: tools/b8-longchat.sh <bin> "<env>"   (run under bash)
cd "$(dirname "$0")/.."
adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 2
adb shell "cd /data/local/tmp/llm; am kill-all; sleep 1; (setsid env GGML_REPACK_CACHE=/data/local/tmp/llm/cache/b8l GGML_REPACK_RESIDENT_MB=962 $2 LIMIT_MB=150 ./tv-watch server.log ./$1/llama-server -m b8-q1-hml.gguf -t 3 -c 1024 -np 1 -ub 128 -b 128 --host 0.0.0.0 --port 8090 --metrics --reasoning off -ctk q8_0 -ctv q8_0 -fa on -cram 0 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8090/health | grep -q ok && break; sleep 2; done
python3 -I tools/long_chat.py
adb shell 'sort -k2 -n /data/local/tmp/llm/server.log.mem | head -1'
