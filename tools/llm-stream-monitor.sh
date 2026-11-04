#!/system/bin/sh
# Runs on the TV (push to /data/local/tmp/llm). Wraps a command and samples, every 2 s:
#   flash MB read (vmstat pgpgin), process major page faults, process RSS, MemAvailable.
# usage: sh llm-stream-monitor.sh <logfile> <command...>
# Prints the command's output to <logfile>.out and samples to <logfile>.csv
LOG="$1"; shift
pg() { grep "^pgpgin " /proc/vmstat | cut -d" " -f2; }
start_kb=$(pg)
start_s=$(date +%s)
"$@" </dev/null >"$LOG.out" 2>&1 &
P=$!
echo "t_s,flash_read_mb,flash_mb_s,majflt,rss_mb,memavail_mb" >"$LOG.csv"
prev_kb=$start_kb
while kill -0 $P 2>/dev/null; do
  sleep 2
  now_kb=$(pg)
  t=$(( $(date +%s) - start_s ))
  maj=$(cut -d" " -f12 /proc/$P/stat 2>/dev/null)
  rss=$(grep VmRSS /proc/$P/status 2>/dev/null | tr -s " " | cut -d" " -f2)
  avail=$(grep MemAvailable /proc/meminfo | tr -s " " | cut -d" " -f2)
  echo "$t,$(( (now_kb - start_kb) / 1024 )),$(( (now_kb - prev_kb) / 2048 )),${maj:-0},$(( ${rss:-0} / 1024 )),$(( avail / 1024 ))" >>"$LOG.csv"
  prev_kb=$now_kb
done
wait $P
echo "exit=$? total_s=$(( $(date +%s) - start_s )) flash_read_mb=$(( ($(pg) - start_kb) / 1024 ))" >>"$LOG.csv"
