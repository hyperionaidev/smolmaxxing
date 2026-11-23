#!/bin/bash
# One-time setup from the Mac: puts the chat page and its launcher on the TV box, so it then runs with no Mac and no internet.
#   ADB_SERIAL=<tv-ip>:5555 tv-llm-ui/install-offline.sh          (127.0.0.1:15555 through tools/adb-relay.sh)
# Afterwards, on the box (any adb shell, or a terminal app on the TV):
#   sh /data/local/tmp/llm/chat.sh        start, page opens in the TV browser at http://127.0.0.1:8090/
#   sh /data/local/tmp/llm/chat.sh stop
set -euo pipefail
cd "$(dirname "$0")"
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
D=/data/local/tmp/llm
A(){ adb -s "$S" shell "$@"; }
MODEL=q08-q4k-h65p8.gguf   # Qwen3.5-0.8B: the 2B does not fit next to the TV browser
SHA=7dd2aac57c08e002543b9c30ddce88ca05d7ec33167103be6491392ca52fc232   # docs/models.md
A "mkdir -p $D/ui"
if [ "$(A "sha256sum $D/$MODEL 2>/dev/null" | cut -d' ' -f1)" != "$SHA" ]; then
  echo "model $MODEL missing or different on the box: run tools/rebuild-models.sh 08, or adb push it from Hugging Face (docs/models.md)"; exit 1
fi
A "[ -x $D/patched25/llama-server ]" || { echo "no $D/patched25/llama-server: build it first (docs/tv-box-setup.md, step 1)"; exit 1; }
adb -s "$S" push index.html config.json "$D/ui/" >/dev/null
adb -s "$S" push chat.sh ../bin/tv-watch ../tools/tv-llm-mode.sh "$D/" >/dev/null
A "chmod 755 $D/tv-watch"
echo "installed. Start it on the box with:  adb -s $S shell sh $D/chat.sh   (or from any shell on the TV)"
