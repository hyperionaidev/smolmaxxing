#!/bin/bash
# Free RAM on the TV box before an LLM benchmark: force-stop heavy user apps (they relaunch on demand)
# and ask Android to drop cached background processes. Changes no setting or package state.
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
A(){ adb -s "$S" shell "$@"; }
avail(){ A "grep MemAvailable /proc/meminfo" | tr -s ' ' | cut -d' ' -f2; }
before=$(avail)
# heavy apps to force-stop: FREE_RAM_APPS="pkg1 pkg2 ..." (default: apps that ship on every Google TV)
for p in ${FREE_RAM_APPS:-com.google.android.youtube.tv com.google.android.youtube.tvmusic com.android.vending}; do
  A "am force-stop $p" 2>/dev/null
done
A "am kill-all" 2>/dev/null
sleep 3
after=$(avail)
echo "MemAvailable: $((before/1024)) MB -> $((after/1024)) MB"
