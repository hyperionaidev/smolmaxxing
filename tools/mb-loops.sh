#!/bin/bash
# Do long answers loop because of routing, bits, or the model? Same long prompt (tools/long_chat.py), seeds 42-44, ~1000 tokens each.
# usage (bash): tools/mb-loops.sh tv "<used> <bias>" ... | tools/mb-loops.sh bonsai | tools/mb-loops.sh ref
# Output lines append to bench/raw/E102-loops-*.txt via the caller; full texts go to bench/raw/E102-long-loops.txt.
cd "$(dirname "$0")/.."
export TXT=bench/raw/E102-long-loops.txt
TV=${TV_IP:-192.168.15.244}
stop(){ adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 3; }
wait_up(){ for i in $(seq 1 ${2:-120}); do curl -s -m 2 "$1/health" | grep -q ok && return 0; sleep 3; done; echo "server not up at $1"; return 1; }
run(){ for S in 42 43 44; do echo "$1 seed $S: $(LABEL="$1" SEED=$S python3 -I tools/long_chat.py 2>&1 | grep -E 'server timings|repeated|URLError' | sed -E "s/server timings //; s/'prompt_per_second': [0-9.]+, //" | tr '\n' ' ' | cut -c1-120)"; done; }
start_tv(){ adb shell "cd /data/local/tmp/llm; am kill-all; sleep 1; (setsid env $1 LIMIT_MB=150 ./tv-watch server.log ./patched40/llama-server -m $2 -t 3 -c 1024 -np 1 $3 --host 0.0.0.0 --port 8090 --metrics --reasoning off -ctk q8_0 -ctv q8_0 -fa on -cram 0 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"; }
case "$1" in
tv)
  export URL=http://$TV:8090; shift
  for cfg in "$@"; do U=${cfg% *}; B=${cfg#* }
    stop
    start_tv "GGML_MOE_PAGER=1 GGML_MOE_POOL=43 GGML_MOE_BIAS=$B LLAMA_MOE_USED=$U GGML_MOE_WARM=/data/local/tmp/llm/moe-warm-moonbeam.txt GGML_REPACK_CACHE=/data/local/tmp/llm/cache/mbq2" mb-q2-h60.gguf "-ub 4 -b 64 -ctxcp 0"
    wait_up $URL && run "Q2 top-$U bias $B"
  done; stop; echo "LOOPS-TV DONE";;
bonsai)
  export URL=http://$TV:8090
  stop
  # first start writes the repack cache (~1 GB), a second start keeps it resident
  start_tv "GGML_REPACK_CACHE=/data/local/tmp/llm/cache/b8l" b8-q1-hml.gguf "-ub 128 -b 128"; wait_up $URL 200; stop
  start_tv "GGML_REPACK_CACHE=/data/local/tmp/llm/cache/b8l GGML_REPACK_RESIDENT_MB=962" b8-q1-hml.gguf "-ub 128 -b 128"
  wait_up $URL && run "Bonsai-8B"
  stop; echo "LOOPS-BONSAI DONE";;
ref)
  # the published Q8_0, all 8 experts, no bias, no pager, on the Mac; -nr because repacking 8.4 GB gets it killed on this 8 GB machine
  export URL=http://127.0.0.1:8091
  third_party/llama.cpp/build-mac/bin/llama-server -m models/moe/mb-q8.gguf -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8091 --reasoning off -cram 0 -ctxcp 0 > bench/raw/E102-ref-server.log 2>&1 &
  P=$!
  wait_up $URL 200 && run "Q8_0 reference"
  kill $P; echo "LOOPS-REF DONE";;
esac
