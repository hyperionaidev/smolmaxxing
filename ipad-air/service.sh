#!/bin/bash
# Optional: run the web chat as a background service on the iPad, so it starts by itself
# whenever the jailbreak is active and needs no terminal or Mac. This changes how the
# iPad boots its services, so it is a separate step.
# Usage: ./service.sh install | remove
set -euo pipefail

[[ "${1:-}" == "install" || "${1:-}" == "remove" ]] || { echo "usage: $0 install | remove"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="${IPAD_KEY:-$HOME/ipadair/.ipad-ssh/id_ed25519}"
KNOWN="$(dirname "$KEY")/known_hosts"
DIR=/var/mobile/llama
PLIST=/Library/LaunchDaemons/com.smolmaxxing.smolchat.plist

ipad() {
  ssh -p "${IPAD_PORT:-2222}" -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
      -o ConnectTimeout=10 -o "UserKnownHostsFile=$KNOWN" root@127.0.0.1 "PATH=/usr/bin:/bin:/usr/sbin:/sbin; $1"
}

if [[ "$1" == "install" ]]; then
  ipad "cat > $PLIST && chown root:wheel $PLIST && chmod 644 $PLIST" < "$HERE/smolchat.plist"
  ipad "touch $DIR/server.log && chown mobile $DIR/server.log && launchctl load $PLIST"
  echo "installed: open http://127.0.0.1:8080 in Safari on the iPad"
else
  ipad "launchctl unload $PLIST 2>/dev/null || true; rm -f $PLIST"
  echo "removed"
fi
