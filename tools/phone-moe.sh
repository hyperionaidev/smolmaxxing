#!/bin/bash
# Chat benchmark of a mixture-of-experts model on the phone (OnePlus Nord CE 2) over USB adb, TV-box style: llama-server with the
# expert pager on the phone, tools/loop_stats.py on the Mac through an adb port forward.
# usage: tools/phone-moe.sh MODEL "<cfg>" ...   each cfg: extra environment and/or server flags, as "ENV=.. ENV=.. -- flags"
#   e.g. tools/phone-moe.sh lm2-q2-h60.gguf "GGML_MOE_POOL=40 -- -t 2" "GGML_MOE_POOL=80 -- -t 4"
# env: N (seeds, default 4), TOK (max tokens, default 256), OUT (log), PORT, WARM (warm-expert list), CACHE (repack cache name)
cd "$(dirname "$0")/.."
S=${ADB_SERIAL:-$(adb get-serialno)}; D=/data/local/tmp/llm; BIN=${BIN:-p44}; PORT=${PORT:-8090}
MODEL=$1; shift
OUT=${OUT:-bench/raw/phone-moe.txt}; N=${N:-4}; TOK=${TOK:-256}
A(){ adb -s "$S" shell "$@" </dev/null; }
adb -s "$S" forward tcp:$PORT tcp:$PORT >/dev/null
export URL=http://127.0.0.1:$PORT
stop(){ A 'for p in $(pidof llama-server); do kill $p; done; true'; sleep 3; }
up(){ for i in $(seq 1 400); do curl -s -m 2 $URL/health | grep -q ok && return 0; A "pidof llama-server >/dev/null" || { echo "server died: $(A "tail -3 $D/server.log")" >> $OUT; return 1; }; sleep 3; done; echo "server not up" >> $OUT; return 1; }
R=${SAMPLER:-'{"repeat_penalty":1.15,"repeat_last_n":256}'}
echo "=== $(date '+%F %T') $MODEL" >> $OUT
for cfg in "$@"; do
  env="${cfg%%--*}"; flags=""; [[ "$cfg" == *--* ]] && flags="${cfg#*--}"
  stop
  A "cd $D; (setsid env GGML_MOE_PAGER=1 GGML_MOE_STATS=1 ${WARM:+GGML_MOE_WARM=$D/$WARM} ${CACHE:+GGML_REPACK_CACHE=$D/cache/$CACHE} $env LIMIT_MB=${LIMIT_MB:-150} ./tv-watch server.log ./$BIN/llama-server -m $MODEL -c 1024 -np 1 --host 127.0.0.1 --port $PORT --metrics --reasoning off -cram 0 -fa on -ctk q8_0 -ctv q8_0 -ub 4 -b 64 $flags </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
  if up; then
    SAMPLER="$R" python3 -I tools/loop_stats.py "phone $MODEL | $cfg" $N $TOK >> $OUT 2>&1
    A "grep -E 'rss|avail' /proc/\$(pidof llama-server)/status 2>/dev/null | grep -E 'VmRSS|VmHWM'" | tr -s ' \n' ' ' >> $OUT; echo >> $OUT
  fi
  stop; A "grep -E 'moe-pager|tv-guard' $D/server.log | tail -2 | cut -c1-240" >> $OUT
done
stop; echo "PHONE-MOE DONE" >> $OUT
