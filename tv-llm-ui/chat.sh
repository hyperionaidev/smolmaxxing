#!/system/bin/sh
# Offline chat on the TV box: runs entirely on the box, no Mac, no internet, no Wi-Fi needed once installed.
# Starts llama-server (Qwen3.5-0.8B) under the memory guard, waits until the model is loaded,
# then opens the chat page in the TV's own browser at http://127.0.0.1:8090/.
#   sh /data/local/tmp/llm/chat.sh            start (headless mode on, page in the TV browser)
#   sh /data/local/tmp/llm/chat.sh stop       stop the server, launcher back
#   BROWSER=0 sh chat.sh                       server only (open the page from a phone on the same network)
#   HEADLESS=0 sh chat.sh                      keep the launcher and Google services running (less RAM for the model)
# Installed by tv-llm-ui/install-offline.sh. Overrides: MODEL BIN THREADS CTX LIMIT_MB SENV EXTRA.
D=/data/local/tmp/llm
PORT=8090
MODEL=${MODEL:-q08-q4k-h65p8.gguf}     # Qwen3.5-0.8B, 65K repacked head (§15-17); the 2B (w-q4k-h65p8.gguf) does not fit
                                       # next to the TV browser even in headless mode: the guard killed it at ~110 MB free
BIN=${BIN:-patched25}
THREADS=${THREADS:-3}
CTX=${CTX:-1024}
SENV=${SENV-LLAMA_STATE_F16=1}         # fp16 DeltaNet state, +11 % in chat (§16)
EXTRA=${EXTRA:--ub 128 -b 128}
cd $D || exit 1

stop_server() { p=$(pidof llama-server); [ -n "$p" ] && kill $p; while pidof llama-server >/dev/null; do sleep 1; done; }
if [ "$1" = stop ]; then
  stop_server
  [ "${HEADLESS:-1}" = 1 ] && sh $D/tv-llm-mode.sh off
  echo stopped; exit 0
fi

stop_server
# headless mode frees ~150-250 MB (launcher, Google services) for the TV browser (~300 MB, E7)
if [ "${HEADLESS:-1}" = 1 ]; then sh $D/tv-llm-mode.sh on; else am kill-all; fi
# always under the memory guard: an unguarded run once left the box thrashing for 6 hours (§18.4)
setsid env $SENV BOX_JSON=$D/ui/box.json LIMIT_MB=${LIMIT_MB:-120} ./tv-watch server.log ./$BIN/llama-server \
  -m $MODEL -t $THREADS -c $CTX -np 1 --host 0.0.0.0 --port $PORT --path $D/ui --metrics --reasoning off $EXTRA \
  </dev/null >/dev/null 2>&1 &
echo -n "loading $MODEL "
i=0
health() { printf 'GET /health HTTP/1.0\r\n\r\n' | nc -w 2 127.0.0.1 $PORT 2>/dev/null | grep -q '"ok"'; }   # no curl/wget on the box
until health; do
  i=$((i+1)); [ $i -gt 180 ] && { echo " timed out, see $D/server.log"; exit 1; }
  pidof llama-server >/dev/null || [ $i -lt 5 ] || { echo " server exited, see $D/server.log"; tail -3 $D/server.log; exit 1; }
  echo -n .; sleep 1
done
echo " ready"
if [ "${BROWSER:-1}" = 1 ]; then
  am start -a android.intent.action.VIEW -d http://127.0.0.1:$PORT/ com.internet.tvbrowser >/dev/null
fi
echo "Chat: http://127.0.0.1:$PORT/ on the TV; avail $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
