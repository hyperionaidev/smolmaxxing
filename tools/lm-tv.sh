#!/bin/bash
# §20 on the TV: Ling-mini-2.0 (16B MoE) with the Moon-Beam recipe. Pager verify run first, then speed + loops per config.
# usage: tools/lm-tv.sh "<env cfg>" ...   (each cfg: extra server environment, e.g. "GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6")
cd "$(dirname "$0")/.."
OUT=${OUT:-bench/raw/E106-ling-tv.txt}; D=/data/local/tmp/llm; export URL=http://${TV_IP:-192.168.15.244}:8090
N=${N:-6}; TOK=${TOK:-400}
stop(){ adb shell 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'; sleep 4; }
up(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health | grep -q ok && return 0; adb shell "pidof llama-server >/dev/null" || { echo "server died" >> $OUT; return 1; }; sleep 3; done; echo "server not up" >> $OUT; return 1; }
srv(){ stop; adb shell "cd $D; sh tv-llm-mode.sh on > /dev/null 2>&1; sleep 5; am kill-all; sleep 1; (setsid env GGML_MOE_PAGER=1 GGML_MOE_STATS=1 GGML_MOE_WARM=$D/moe-warm-ling.txt GGML_REPACK_CACHE=$D/cache/lmq2 GGML_MOE_POOL=40 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6 GGML_MOE_PREFETCH_K=6 GGML_MOE_PREFETCH_AHEAD=2 $1 LIMIT_MB=150 ./tv-watch server.log ./patched42/llama-server -m lm2-q2-h60.gguf -t 3 -c 1024 -np 1 --host 0.0.0.0 --port 8090 --metrics --reasoning off -cram 0 -fa on -ctk q8_0 -ctv q8_0 -ub 4 -b 64 </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"; up; }
pstats(){ stop; adb shell "grep -E 'moe-pager|tv-guard' $D/server.log | tail -2 | cut -c1-220" >> $OUT; }
R='{"repeat_penalty":1.15,"repeat_last_n":256}'
echo "=== $(date +%T) $*" >> $OUT
for cfg in "$@"; do
  case "$cfg" in VERIFY*) n=2; c="GGML_MOE_VERIFY=1 ${cfg#VERIFY}";; *) n=$N; c="$cfg";; esac
  srv "$c" && TXT=bench/raw/E106-ling-texts.txt SAMPLER="$R" python3 -I tools/loop_stats.py "TV ling $c" $n $TOK >> $OUT 2>&1; pstats
done
stop; echo "LM-TV DONE" >> $OUT
