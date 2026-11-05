#!/bin/bash
# MTP speculative decoding sweep (runs on the Mac, drives the TV): for each config start llama-server on the TV,
# run tv-llm-ui/bench.py (warm-up + 4 fixed prompts, greedy, 128 tokens; logs to bench/results.jsonl with draft
# acceptance from /metrics), then stop the server.
# usage: tools/mtp-bench.sh <model.gguf on TV> [n_max...]     (n_max 0 = no speculative decoding)
# SENV="VAR=value ..." sets the server's environment (e.g. SENV=LLAMA_STATE_F16=1).
set -u
M=$1; shift; BIN=${BIN:-patched2}   # BIN=patched3 for the fused DeltaNet build
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"; IP="${S%%:*}"; A(){ adb -s $S shell "$@"; }
cd "$(dirname "$0")/.."
for N in "${@:-0 1 2 3}"; do
  SPEC=""; [ "$N" != 0 ] && SPEC="--spec-type draft-mtp --spec-draft-n-max $N"
  A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 2
  A "cd /data/local/tmp/llm && (${SENV:+env $SENV} setsid ./$BIN/llama-server -m $M -t 3 -c 1024 -np 1 -ub 128 -b 128 --host 0.0.0.0 --port 8090 --reasoning off --metrics $SPEC </dev/null >mtp-server-$N.log 2>&1 &) </dev/null >/dev/null 2>&1"
  for i in $(seq 1 90); do curl -s -m 2 http://$IP:8090/health | grep -q ok && break; sleep 2; done
  echo "== n_max=$N ($M)"
  python3 -I tv-llm-ui/bench.py "MTP n_max=$N: $M, $BIN, t=3${SENV:+, $SENV}" 128
  A "p=\$(pidof llama-server); grep VmRSS /proc/\$p/status; grep -iE 'mtp|draft|spec' /data/local/tmp/llm/mtp-server-$N.log | head -5"
done
A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'
