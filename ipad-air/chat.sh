#!/bin/bash
# Open a chat with a model on the iPad.
# Usage: ./chat.sh model-file.gguf [smolchat options]
# Expects SSH on 127.0.0.1:2222 (see README.md). IPAD_KEY overrides the key path.
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: $0 model-file.gguf [smolchat options]"; exit 1; }

KEY="${IPAD_KEY:-$HOME/ipadair/.ipad-ssh/id_ed25519}"
KNOWN="$(dirname "$KEY")/known_hosts"
model="$1"; shift

args="$(printf ' %q' "$@")"
[[ $# -gt 0 ]] || args=""

exec ssh -t -p "${IPAD_PORT:-2222}" -i "$KEY" -o IdentitiesOnly=yes \
  -o "UserKnownHostsFile=$KNOWN" root@127.0.0.1 \
  "cd /var/mobile/llama && ./smolchat -m $(printf %q "$model")$args"
