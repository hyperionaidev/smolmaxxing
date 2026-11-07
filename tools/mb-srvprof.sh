#!/bin/bash
# Where a real chat token's time goes on the TV: graph compute (ubatch profile) vs the server's wall time per token.
cd "$(dirname "$0")/.."
D=/data/local/tmp/llm; export URL=http://${TV_IP:-192.168.15.244}:8090
adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 3
adb shell "cd $D; sh tv-llm-mode.sh on > /dev/null 2>&1; sleep 5; am kill-all; sleep 1; (setsid env LLAMA_UBATCH_PROF=1 GGML_OPPROF=1 GGML_MOE_PAGER=1 GGML_MOE_STATS=1 GGML_MOE_WARM=$D/moe-warm-moonbeam.txt GGML_REPACK_CACHE=$D/cache/mbq2 GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6 $1 LIMIT_MB=150 ./tv-watch server.log ./patched41/llama-server -m mb-q2-h60.gguf -t ${T:-3} -c 1024 -np 1 --host 0.0.0.0 --port 8090 --metrics --reasoning off -cram 0 -ctxcp 0 -fa on -ctk q8_0 -ctv q8_0 -ub 4 -b 64 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
for i in $(seq 1 200); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
python3 -I - <<'PY'
import json,urllib.request,os
b={"messages":[{"role":"user","content":"Write a detailed essay about the history of computing."}],"max_tokens":400,"temperature":0.7,"seed":7,"repeat_penalty":1.15,"repeat_last_n":256}
r=json.loads(urllib.request.urlopen(urllib.request.Request(os.environ['URL']+"/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=900).read())
t=r['timings']; print("server: %d tokens, %.1f ms per token (%.2f tok/s)" % (t['predicted_n'], t['predicted_ms']/t['predicted_n'], t['predicted_per_second']))
PY
adb shell 'kill $(pidof llama-server); sleep 4; grep -E "ubprof|opprof\] [0-9]+ graphs|moe-pager" '$D'/server.log | cut -c1-200; grep -E "^\[opprof\] [A-Z]" '$D'/server.log | head -12 | cut -c1-92'
