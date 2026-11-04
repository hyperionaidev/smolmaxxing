#!/bin/bash
# Streamed-model speculative decoding sweep (Mac drives the TV): for each n_max start llama-server on the TV with
# the streaming env (ENV="LLAMA_STREAM_RESIDENT_MB=... ..."), run tv-llm-ui/bench.py (warm-up + 4 fixed prompts,
# greedy, NTOK tokens, draft acceptance from /metrics; appends to bench/results.jsonl), then stop the server.
# The server runs under tv-guard.sh (kills it if MemAvailable < 180 MB).
# usage: ENV="..." EXTRA="-nr" NTOK=24 tools/stream-mtp-bench.sh <model.gguf on TV> [n_max...]   (0 = no MTP)
set -u
M=$1; shift; BIN=${BIN:-patched14}; NTOK=${NTOK:-24}; ENV=${ENV:-}; EXTRA=${EXTRA:-}
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"; IP="${S%%:*}"; A(){ adb -s $S shell "$@"; }
cd "$(dirname "$0")/.."
for N in "${@:-0 1 2}"; do
  SPEC=""; [ "$N" != 0 ] && SPEC="--spec-type draft-mtp --spec-draft-n-max $N"
  A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 2
  A "cd /data/local/tmp/llm && (setsid env $ENV sh tv-guard.sh smtp-$N.log ./$BIN/llama-server -m $M -t 3 -c 1024 -np 1 -ub 16 -b 64 --host 0.0.0.0 --port 8090 --reasoning off --metrics $EXTRA $SPEC </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
  ok=0; for i in $(seq 1 150); do curl -s -m 2 http://$IP:8090/health | grep -q ok && { ok=1; break; }
    A "grep -q 'tv-guard: exit' /data/local/tmp/llm/smtp-$N.log" && break; sleep 2; done
  echo "== n_max=$N ($M) $ENV"
  [ $ok = 1 ] || { echo "server did not start:"; A "tail -3 /data/local/tmp/llm/smtp-$N.log"; continue; }
  python3 -I tv-llm-ui/bench.py "stream MTP n_max=$N: $M, $BIN, $ENV" $NTOK
  A "p=\$(pidof llama-server); [ -n \"\$p\" ] && grep VmRSS /proc/\$p/status; grep -E 'streaming|tv-guard' /data/local/tmp/llm/smtp-$N.log | tail -2"
done
A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'
