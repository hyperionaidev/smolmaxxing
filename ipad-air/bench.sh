#!/bin/bash
# Run a fixed three-turn conversation on the iPad under a memory guard.
# Usage: ./bench.sh model-file.gguf [smolchat options]
# Prints the replies, the speed line for each turn, peak RSS and the lowest available memory.
# The run is killed if available memory (free + inactive + speculative) drops below LIMIT_MB.
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: $0 model-file.gguf [smolchat options]"; exit 1; }

KEY="${IPAD_KEY:-$HOME/ipadair/.ipad-ssh/id_ed25519}"
KNOWN="$(dirname "$KEY")/known_hosts"
LIMIT_MB="${LIMIT_MB:-50}"
MAX_SECONDS="${MAX_SECONDS:-300}" # evictable file pages count as available, so thrashing needs a time limit
model="$1"; shift

args="$(printf ' %q' "$@")"
[[ $# -gt 0 ]] || args=""

ssh -p "${IPAD_PORT:-2222}" -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o ConnectTimeout=10 -o "UserKnownHostsFile=$KNOWN" root@127.0.0.1 /bin/sh -s <<EOF
cd /var/mobile/llama
printf 'What is the capital of France?\nWhy does the sea have tides? Answer in two sentences.\nWhat was my first question?\n/quit\n' > bench.q
pages() { vm_stat | sed -n "s/^\$1: *\([0-9]*\)\.\$/\1/p"; }
avail_mb() { echo \$(( (\$(pages 'Pages free') + \$(pages 'Pages inactive') + \$(pages 'Pages speculative')) * 4096 / 1048576 )); }
${IPAD_ENV:-} ./${BIN:-smolchat} -m '$model' $args < bench.q > bench.out 2> bench.err &
p=\$!
peak=0; low=\$(avail_mb); killed=0; n=0
while kill -0 \$p 2>/dev/null; do
  r=\$(ps -o rss= -p \$p 2>/dev/null | tr -d ' ')
  [ -n "\$r" ] && [ "\$r" -gt "\$peak" ] && peak=\$r
  a=\$(avail_mb)
  [ "\$a" -lt "\$low" ] && low=\$a
  if [ "\$a" -lt $LIMIT_MB ] || [ "\$n" -ge $MAX_SECONDS ]; then kill -9 \$p; killed=1; break; fi
  sleep 1; n=\$((n+1))
done
wait \$p; rc=\$?
cat bench.out
tail -5 bench.err
echo "rc=\$rc guard_killed=\$killed wall=\${n}s peakRSS=\$((peak/1024))MB lowest_avail=\${low}MB"
EOF
