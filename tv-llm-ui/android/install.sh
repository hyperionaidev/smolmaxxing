#!/bin/bash
# Install SmolChat on the TV box: the app, plus the model in the app's own storage. After this the box needs
# no Mac, adb or internet: the server starts at every boot, and "SmolChat" on the home screen opens the chat.
#   ADB_SERIAL=<tv-ip>:5555 tv-llm-ui/android/install.sh        (127.0.0.1:15555 through tools/adb-relay.sh)
# The model comes from the box itself (/data/local/tmp/llm, put there by tools/rebuild-models.sh 08) or from
# MODEL_FILE=<path on the Mac> (download: docs/models.md, hyperionaidev/Qwen3.5-0.8B-tvbox-GGUF).
set -euo pipefail
cd "$(dirname "$0")"
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
A(){ adb -s "$S" shell "$@"; }
PKG=com.smolmaxxing.tvchat
MODEL=q08-q4k-h65p8.gguf
SHA=7dd2aac57c08e002543b9c30ddce88ca05d7ec33167103be6491392ca52fc232   # docs/models.md
DST=/sdcard/Android/data/$PKG/files
[ -f build/smolchat.apk ] || ./build.sh
# the server binaries are aarch64: they need a 64-bit kernel, even under a 32-bit Android (see build.sh)
# (uname -m says armv8l under a 32-bit Android even then, so run one of them to find out)
adb -s "$S" push ../../bin/tv-watch /data/local/tmp/smolchat-probe >/dev/null
A "chmod 755 /data/local/tmp/smolchat-probe; /data/local/tmp/smolchat-probe 2>&1; rm -f /data/local/tmp/smolchat-probe" | grep -q usage \
  || { echo "this box cannot run aarch64 programs (32-bit kernel): the server binaries need a 64-bit one"; exit 1; }

# the adb-started server (chat.sh) would hold port 8090; stop it and leave headless mode
A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; [ -f /data/local/tmp/llm/tv-llm-mode.sh ] && sh /data/local/tmp/llm/tv-llm-mode.sh off >/dev/null; true'

adb -s "$S" install -r build/smolchat.apk
A "mkdir -p $DST"
if [ "$(A "sha256sum $DST/$MODEL 2>/dev/null" | cut -d' ' -f1)" != "$SHA" ]; then
  if [ -n "${MODEL_FILE:-}" ]; then adb -s "$S" push "$MODEL_FILE" "$DST/$MODEL"
  else A "cp /data/local/tmp/llm/$MODEL $DST/$MODEL"; fi
  [ "$(A "sha256sum $DST/$MODEL" | cut -d' ' -f1)" = "$SHA" ] || { echo "model checksum mismatch in $DST"; exit 1; }
fi
# created by the shell user with mode 660: the app (another uid, not in the group) could not read it
A "chmod 644 $DST/$MODEL"
# open it once: Android only sends the boot broadcast to apps that have been started since install
A "am start -n $PKG/.MainActivity" >/dev/null
echo "installed: SmolChat starts its server at boot; open SmolChat from the TV home screen to chat"
