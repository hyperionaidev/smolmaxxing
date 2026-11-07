#!/system/bin/sh
# "Headless" LLM mode: put a tiny activity in front, then force-stop the launchers and the heavy Google/Xiaomi background
# apps; a reboot or pressing Home brings those back. Only "stay awake" persists until "off" (see below).
# usage: sh tv-llm-mode.sh [on|status]     (LEVEL=2 also shrinks UI buffers with wm size, undone by "off")
#        sh tv-llm-mode.sh off             wm size reset, launcher back in front
DISABLE="com.google.android.katniss com.android.vending com.google.android.tts"
avail(){ awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
case "${1:-on}" in
off)
  for p in $DISABLE; do pm enable --user 0 $p >/dev/null 2>&1; done
  svc power stayon false; wm size reset; am start -a android.intent.action.MAIN -c android.intent.category.HOME >/dev/null 2>&1; echo "off, avail $(avail) MB"; exit 0;;
status)
  echo "avail $(avail) MB"; exit 0;;
esac
b=$(avail)
# No screensaver: dreams.basic animates at full frame rate and drags surfaceflinger, the composer and the display path with
# it (~40 % of a core and DRAM bandwidth, E82). stayon rather than disabling dreams, which would let the box go to sleep.
# This one setting persists: "off" and rollback.sh undo it.
svc power stayon true
input keyevent 224
# The three apps that respawn right after a force-stop; the user's "LLM mode" of 2026-11-03 disabled the same three, and
# something on the box keeps re-enabling them. "off" and rollback.sh enable them again.
for p in $DISABLE; do pm disable-user --user 0 $p >/dev/null 2>&1; done
# a small foreground activity, so the system has no reason to restart a launcher (~50 MB vs ~110-170 MB for tv.settings)
am start -n com.android.providers.calendar/.CalendarDebugActivity >/dev/null 2>&1
sleep 2
for p in com.android.tv.settings com.spocky.projengmenu com.google.android.apps.tv.launcherx com.google.android.inputmethod.latin com.google.android.tungsten.setupwraith \
         com.google.android.tv.axel com.xiaomi.android.tvsetup.partnercustomizer com.google.android.partnersetup \
         com.android.statementservice com.android.rkpdapp com.google.android.permissioncontroller \
         com.android.devicelockcontroller android.process.acore com.android.externalstorage com.mitv.overseaservice \
         com.google.android.gms ; do
  am force-stop $p 2>/dev/null
done
[ "${LEVEL:-1}" -ge 2 ] && wm size 640x360
am kill-all
sleep ${SETTLE:-15}
# second pass: whatever restarted while the first one settled
for p in com.google.android.gms com.google.android.inputmethod.latin com.google.android.apps.tv.launcherx; do am force-stop $p 2>/dev/null; done
am kill-all
sleep 3
echo "avail $b -> $(avail) MB"
