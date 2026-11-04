#!/system/bin/sh
# Run a command on the TV with a memory watchdog: if MemAvailable drops below LIMIT_MB (default 180),
# the command is killed before Android / the kernel fall over. (llama-quantize with its default 8 GB slab
# buffer once ran the box out of memory and caused a kernel panic + reboot.)
# Samples go to <log>.mem (time, MemAvailable MB, SwapFree MB, RSS MB of the command); output to <log>.
# GUARD_INTERVAL (seconds, default 1; e.g. 0.2) sets the sampling period; GUARD_ECHO=1 also prints each sample to
# stdout, so a watcher on the Mac keeps the readings even if the box crashes before the log reaches flash.
# GUARD_KILL_NAMES="a b" also kills those processes (by name) on trigger, for multi-process jobs.
# usage: sh tv-guard.sh <log> <command...>
LOG=$1; shift
LIMIT=${LIMIT_MB:-180}
"$@" > "$LOG" 2>&1 &
P=$!
: > "$LOG.mem"
while kill -0 $P 2>/dev/null; do
  set -- $(awk '/MemAvailable/{a=$2} /SwapFree/{s=$2} END{print int(a/1024), int(s/1024)}' /proc/meminfo)
  R=$(awk '/VmRSS/{print int($2/1024)}' /proc/$P/status 2>/dev/null)
  echo "$(date +%T) $1 $2 ${R:-0}" >> "$LOG.mem"
  if [ -n "$GUARD_ECHO" ]; then
    D=$(awk '/^RssAnon/{a=int($2/1024)} /^RssFile/{f=int($2/1024)} END{print "anon " a " file " f}' /proc/$P/status 2>/dev/null)
    M=$(awk '/^MemFree/{f=$2} /^Cached:/{c=$2} /^Unevictable/{u=$2} /^AnonPages/{a=$2} END{printf "free %d cached %d unevict %d anonall %d", f/1024, c/1024, u/1024, a/1024}' /proc/meminfo)
    echo "mem $(date +%T) avail $1 swapfree $2 rss ${R:-0} $D | $M"
  fi
  if [ "$1" -lt "$LIMIT" ]; then
    kill -9 $P; for n in $GUARD_KILL_NAMES; do q=$(pidof $n); [ -n "$q" ] && kill -9 $q; done
    echo "tv-guard: killed, MemAvailable $1 MB < $LIMIT MB" >> "$LOG"; break
  fi
  sleep ${GUARD_INTERVAL:-1}
done
wait $P; RC=$?
echo "tv-guard: exit $RC, min MemAvailable $(sort -k2 -n "$LOG.mem" | head -1 | cut -d' ' -f2) MB, max RSS $(sort -k4 -n "$LOG.mem" | tail -1 | cut -d' ' -f4) MB" >> "$LOG"
sync
exit $RC
