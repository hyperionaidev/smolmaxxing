#!/bin/bash
# Memory probe (runs on the Mac, drives the TV): start llama-server with a model, send one real chat request,
# and record the server's memory breakdown (/proc/<pid>/status: RssAnon = real RAM, RssFile = mapped file pages,
# VmSwap) plus system MemAvailable/MemFree, after load and after the request. Stops the server at the end.
# usage: tools/memprobe.sh <model.gguf on TV> <tag> [extra env assignments for the server, e.g. LLAMA_KEEP_MAPPED_COPIES=1]
set -u
M=$1; T=$2; ENVS=${3:-}; BIN=${BIN:-./patched2/llama-server}
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"; IP="${S%%:*}"; A(){ adb -s $S shell "$@"; }
OUT=$(dirname "$0")/../bench/raw/memprobe-$T.txt
A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'
sleep 2
A "cd /data/local/tmp/llm && (setsid env $ENVS $BIN -m $M -t 3 -c 1024 -np 1 -ub 128 -b 128 --host 0.0.0.0 --port 8090 --reasoning off </dev/null >memprobe.log 2>&1 &) </dev/null >/dev/null 2>&1"
for i in $(seq 1 120); do curl -s -m 2 http://$IP:8090/health | grep -q ok && break; sleep 2; done
snap(){ A 'p=$(pidof llama-server); grep -E "VmRSS|RssAnon|RssFile|VmSwap" /proc/$p/status | tr -s " " | tr "\n" " "; grep -E "MemAvailable|MemFree|^Cached" /proc/meminfo | tr -s " " | tr "\n" " "'; echo; }
{
echo "# memprobe $T  model=$M  env='$ENVS'  bin=$BIN  $(date '+%F %T')"
echo "after load:    $(snap)"
R=$(curl -s -m 600 http://$IP:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain how a TV remote control works, in simple words, in about 80 words."}],"max_tokens":120,"temperature":0}')
echo "after request: $(snap)"
echo "speed: $(echo "$R" | python3 -c 'import sys,json; t=json.load(sys.stdin)["timings"]; print("gen %.2f tok/s (%d tok), prompt %.1f tok/s" % (t["predicted_per_second"], t["predicted_n"], t["prompt_per_second"]))')"
echo "text: $(echo "$R" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:300])')"
} | tee "$OUT"
A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'
