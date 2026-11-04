#!/system/bin/sh
# Runs on the TV next to llama-server. Every second writes box.json (served by llama-server --path)
# with live memory / flash / CPU numbers the web page can't read itself.
# usage: sh boxstats.sh <ui dir>
OUT="$1/box.json"
pg() { grep "^pgpgin " /proc/vmstat | cut -d" " -f2; }
mi() { grep "^$1:" /proc/meminfo | tr -s " " | cut -d" " -f2; }
prev=$(pg)
while true; do
  sleep 1
  now=$(pg)
  pid=$(pidof llama-server)
  rss=0; [ -n "$pid" ] && rss=$(grep VmRSS /proc/$pid/status 2>/dev/null | tr -s " " | cut -d" " -f2)
  f0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)
  f3=$(cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq 2>/dev/null)
  load=$(cut -d" " -f1 /proc/loadavg)
  printf '{"t":%s,"memtotal_mb":%s,"memavail_mb":%s,"swapfree_mb":%s,"server_rss_mb":%s,"flash_read_mb_s":%s,"cpu0_mhz":%s,"cpu3_mhz":%s,"load1":%s}\n' \
    "$(date +%s)" $(( $(mi MemTotal) / 1024 )) $(( $(mi MemAvailable) / 1024 )) $(( $(mi SwapFree) / 1024 )) \
    $(( ${rss:-0} / 1024 )) $(( (now - prev) / 1024 )) $(( ${f0:-0} / 1000 )) $(( ${f3:-0} / 1000 )) "$load" > "$OUT.tmp"
  mv "$OUT.tmp" "$OUT"
  prev=$now
done
