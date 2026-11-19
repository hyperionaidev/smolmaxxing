#!/bin/sh
# Generation speed on the Kindle with llama-bench, three runs, under a memory guard.
# usage: kindle/bench.sh <model path on the Kindle> [bin dir on the Kindle] [extra llama-bench options]
# Runs over SSH (KINDLE_SSH, default ./ssh.sh of the Inkwell repo). Prints each run's tok/s,
# the middle one, peak RSS and the lowest available memory. A run is killed if available memory
# drops below LIMIT_MB or it takes longer than MAX_SECONDS, since a model that thrashes never finishes.
# ARGS sets the llama-bench test (default: 128 generated tokens, no prompt).
set -eu
[ $# -ge 1 ] || { echo "usage: $0 model.gguf [bindir] [llama-bench options]"; exit 1; }
SSH=${KINDLE_SSH:-./ssh.sh}
M=$1; BIN=${2:-/root/smol/patched}; shift; if [ $# -gt 0 ]; then shift; fi
EXTRA="$*"; LIMIT_MB=${LIMIT_MB:-20}; MAX_SECONDS=${MAX_SECONDS:-1200}; RUNS=${RUNS:-3}
ARGS=${ARGS:--p 0 -n 128}
$SSH sh -s <<EOF
# Kernel 3.0 has no MemAvailable, so count free memory plus buffers and page cache.
avail() { awk '/^(MemFree|Buffers|Cached):/ {s+=\$2} END {print s}' /proc/meminfo; }
rm -f /tmp/kb.all; i=0
while [ \$i -lt $RUNS ]; do
  i=\$((i+1))
  sync; echo 3 > /proc/sys/vm/drop_caches; sleep 5
  $BIN/llama-bench -m '$M' $ARGS -t 1 -r 1 -o csv $EXTRA > /tmp/kb.out 2> /tmp/kb.err &
  p=\$!; peak=0; low=\$(avail); n=0; killed=0
  while kill -0 \$p 2>/dev/null; do
    r=\$(awk '/^VmRSS:/ {print \$2}' /proc/\$p/status 2>/dev/null)
    [ -n "\$r" ] && [ "\$r" -gt "\$peak" ] && peak=\$r
    a=\$(avail); [ "\$a" -lt "\$low" ] && low=\$a
    if [ "\$a" -lt \$(( $LIMIT_MB * 1024 )) ] || [ \$n -ge $MAX_SECONDS ]; then kill -9 \$p; killed=1; break; fi
    sleep 1; n=\$((n+1))
  done
  wait \$p 2>/dev/null || true
  ts=\$(tail -1 /tmp/kb.out | awk -F, '{gsub(/"/,""); print \$(NF-1)}')
  echo "run \$i: \${ts:-none} tok/s wall=\${n}s peakRSS=\$((peak/1024))MB lowest_avail=\$((low/1024))MB guard_killed=\$killed"
  echo "\${ts:-none}" >> /tmp/kb.all
  if [ \$killed = 1 ]; then tail -3 /tmp/kb.err; break; fi
done
echo "middle: \$(grep -v none /tmp/kb.all | sort -n | awk '{a[NR]=\$1} END {if (NR) print a[int((NR+1)/2)]; else print "none"}') tok/s"
rm -f /tmp/kb.all
EOF
