#!/bin/bash
# Copy the binaries (and any model files named as arguments) to the iPad and sign them.
# Usage: ./deploy.sh [models/some-model.gguf ...]
# Expects SSH on 127.0.0.1:2222 (see README.md). IPAD_KEY overrides the key path.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
KEY="${IPAD_KEY:-$HOME/ipadair/.ipad-ssh/id_ed25519}"
KNOWN="$(dirname "$KEY")/known_hosts"
DIR=/var/mobile/llama

ipad() {
  ssh -p "${IPAD_PORT:-2222}" -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
      -o ConnectTimeout=10 -o "UserKnownHostsFile=$KNOWN" root@127.0.0.1 "$@"
}

# A running binary cannot be overwritten: stop any server first (./service.sh remove if installed).
ipad "mkdir -p $DIR"
ipad "cat > $DIR/ent.xml" < "$HERE/ent.xml"

for name in smolchat llama-simple tinyppl; do
  ipad "cat > $DIR/$name && chmod 755 $DIR/$name" < "$HERE/bin/$name"
  ipad "PATH=/usr/bin:/bin ldid -S$DIR/ent.xml $DIR/$name"
done

for model in "$@"; do
  ipad "cat > $DIR/$(basename "$model")" < "$model"
done

ipad "ls -lh $DIR"
