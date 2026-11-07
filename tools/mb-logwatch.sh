#!/bin/bash
# Saves how each TV server ended (tv-watch's last lines + memory low point) before the next run overwrites server.log.
# Runs until the factor tests finish. Output: bench/raw/E103-server-ends.txt
cd "$(dirname "$0")/.."
last=""
while pgrep -f mb-factors.sh > /dev/null; do
  cur=$(timeout 15 adb shell 'cd /data/local/tmp/llm; tail -4 server.log | cut -c1-200; echo MEMLOW $(sort -k2 -n server.log.mem | head -1); echo MEMLAST $(tail -1 server.log.mem); ps -A -o PID,RSS,ARGS | grep "[l]lama-server" | cut -c1-60' 2>/dev/null)
  if echo "$cur" | grep -qE 'tv-guard:|GGML_ASSERT|abort|Segmentation|moe pager|fatal' && [ "$cur" != "$last" ]; then
    { date +%T; echo "$cur"; echo; } >> bench/raw/E103-server-ends.txt
    last=$cur
  fi
  sleep 5
done
