#!/bin/bash
# Build + deploy + (re)start the Tiny LLM web server on the TV box, then open it in the TV browser.
# Re-run after every TV reboot (the server runs as the adb shell user, so nothing auto-starts it).
#   ./start.sh           build if needed, push, restart, open the page on the TV
#   ./start.sh stop      stop the server
set -euo pipefail
cd "$(dirname "$0")"
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
D=/data/local/tmp/tinyllm
LLAMA2C=../third_party/llama2.c
MODELS=../models/llama2c
PORT=8080
adb connect "$S" >/dev/null 2>&1 || true
A(){ adb -s "$S" shell "$@"; }

# pidof, not pkill -f: a -f pattern also matches the adb shell's own command line
A 'p=$(pidof llm-server); [ -n "$p" ] && kill $p; true' >/dev/null 2>&1 || true
[ "${1:-}" = "stop" ] && { echo "stopped"; exit 0; }

if [ ! -f build/llm-server ] || [ server.c -nt build/llm-server ]; then
  mkdir -p build
  zig cc -target aarch64-linux-musl -O3 -static -include stdint.h -I "$LLAMA2C" -o build/llm-server server.c -lm
fi

A "mkdir -p $D"
adb -s "$S" push build/llm-server index.html "$MODELS/stories260K.bin" "$MODELS/tok512.bin" "$D/" >/dev/null
A "chmod 755 $D/llm-server"
# setsid in a subshell with every fd redirected: keeps running after adb disconnects,
# and adb shell returns at once (it otherwise waits on the background process's fds)
A "cd $D && (setsid ./llm-server stories260K.bin tok512.bin index.html $PORT </dev/null >server.log 2>&1 &) </dev/null >/dev/null 2>&1"
sleep 1
A "cat $D/server.log"
# the TV browser turns 127.0.0.1 / localhost into a Google search, so open the LAN IP
A "am start -a android.intent.action.VIEW -d http://${S%%:*}:$PORT/ com.internet.tvbrowser" >/dev/null
echo "On the TV:   http://${S%%:*}:$PORT/"
echo "From the Mac / phone: http://${S%%:*}:$PORT/"
