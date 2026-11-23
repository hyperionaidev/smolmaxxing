#!/bin/bash
# Relay 127.0.0.1:15555 -> <box>:5555 through Apple's /usr/bin/nc, for when macOS Local Network privacy
# blocks adb ("failed to connect ... No route to host") although ping and nc reach the box.
# Run it yourself in a Terminal window and leave it running (Ctrl-C stops it):
#   tools/adb-relay.sh [box-ip]          # default 192.168.15.244
#   adb kill-server; adb connect 127.0.0.1:15555
#   export ADB_SERIAL=127.0.0.1:15555
# Each loop pass carries one adb connection (adb keeps one per device). When either side closes (adb
# disconnects, the box reboots or changes network) both are torn down and a fresh listener is up within
# about a second; "adb connect 127.0.0.1:15555" then brings the device back.
BOX=${1:-192.168.15.244}
BOX_PORT=${BOX_PORT:-5555}
PORT=${RELAY_PORT:-15555}
T=$(mktemp -d "${TMPDIR:-/tmp}/adb-relay.XXXXXX")
L=; R=
cleanup() { kill $L $R 2>/dev/null; rm -rf "$T"; exit 0; }
trap cleanup INT TERM
/usr/bin/nc -z -G 3 "$BOX" "$BOX_PORT" || { echo "adb-relay: $BOX:$BOX_PORT not reachable (box off, wrong IP, or network ADB disabled)"; exit 1; }
echo "adb-relay: 127.0.0.1:$PORT -> $BOX:$BOX_PORT  (now: adb connect 127.0.0.1:$PORT)"
while true; do
  rm -f "$T/up" "$T/down"; mkfifo "$T/up" "$T/down"
  # FIFO open order matters (each open blocks until the other end opens): both sides open "up" first.
  /usr/bin/nc -l 127.0.0.1 "$PORT" > "$T/up" < "$T/down" & L=$!
  /usr/bin/nc "$BOX" "$BOX_PORT" < "$T/up" > "$T/down" & R=$!
  while kill -0 $L 2>/dev/null && kill -0 $R 2>/dev/null; do sleep 0.3; done
  kill $L $R 2>/dev/null; wait $L $R 2>/dev/null
  [ -n "${RELAY_DEBUG:-}" ] && echo "adb-relay: connection closed, relaying again"
  sleep 0.5
done
