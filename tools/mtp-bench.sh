#!/bin/bash
# MTP speculative decoding sweep (runs on the Mac, drives the TV): for each config start llama-server on the TV,
# run tv-llm-ui/bench.py (warm-up + 4 fixed prompts, greedy, 128 tokens; logs to bench/results.jsonl with draft
# acceptance from /metrics), then stop the server.
# usage: tools/mtp-bench.sh <model.gguf on TV> [n_max...]     (n_max 0 = no speculative decoding)
# SENV="VAR=value ..." sets the server's environment (e.g. SENV=LLAMA_STATE_F16=1).
set -u
M=$1; shift; BIN=${BIN:-patched2}   # BIN=patched3 for the fused DeltaNet build
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"; IP="${TV_IP:-${S%%:*}}"; A(){ adb -s $S shell "$@"; }
# EXTRA="..." adds server flags; TOKENS=n tokens per prompt. Always under tv-guard (LIMIT_MB, default 180): an unguarded
# chat run left the box thrashing for 6 hours on 2026-11-06 (EXPERIMENT.md 18.4).
cd "$(dirname "$0")/.."
for N in "${@:-0 1 2 3}"; do
  SPEC=""; [ "$N" != 0 ] && SPEC="--spec-type draft-mtp --spec-draft-n-max $N"
  A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 2
  A "cd /data/local/tmp/llm && (${SENV:+env $SENV} LIMIT_MB=${LIMIT_MB:-180} setsid sh tv-guard.sh mtp-server-$N.log ./$BIN/llama-server -m $M -t ${THREADS:-3} -c ${CTX:-1024} -np 1 ${UB:--ub 128 -b 128} --host 0.0.0.0 --port 8090 --reasoning off --metrics $SPEC ${EXTRA:-} </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
  for i in $(seq 1 90); do curl -s -m 2 http://$IP:8090/health | grep -q ok && break; sleep 2; done
  curl -s -m 2 http://$IP:8090/health | grep -q ok || { echo "server not up"; A "tail -3 /data/local/tmp/llm/mtp-server-$N.log"; continue; }
  echo "== n_max=$N ($M)"
  python3 -I tv-llm-ui/bench.py "MTP n_max=$N: $M, $BIN, t=3${SENV:+, $SENV} ${EXTRA:-}" ${TOKENS:-128}
  A "p=\$(pidof llama-server); grep VmRSS /proc/\$p/status; grep -iE 'mtp|draft|spec' /data/local/tmp/llm/mtp-server-$N.log | head -5"
done
A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'
