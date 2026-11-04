#!/bin/bash
# Mac side of the low-bit sweep: quantize the slow IQ variants on the Mac (many fast cores) while the TV
# evaluates, then hand each file to the TV over HTTP (tools/serve_range.py + tvfetch). Same BF16 source file
# (same SHA-256 03cb5808...) and same recipe as tools/tv-lowbit.sh, so the result is the same as quantizing on
# the TV. A file is only moved into place if the TV hasn't started that variant itself.
# usage: tools/mac-lowbit.sh <name:FTYPE>...     e.g. iq2xxs:IQ2_XXS iq1m:IQ1_M
set -u
cd "$(dirname "$0")/.."
Q=third_party/llama.cpp/build-mac/bin/llama-quantize
IN=models/gguf/work-bf16-h32k-mtp.gguf
SV=${SERVE_DIR:?set SERVE_DIR to the directory served by serve_range.py}
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"; IP="${S%%:*}"
for spec in "$@"; do
  N=${spec%%:*}; FT=${spec##*:}
  echo "== $N ($FT) $(date +%T)"
  $Q --imatrix models/gguf/imatrix_unsloth.gguf --token-embedding-type q4_K --output-tensor-type q4_K \
     --tensor-type 'blk\.24\.=q4_K' "$IN" "$SV/.w-$N.gguf.tmp" "$FT" 8 > bench/raw/mac-quant-$N.log 2>&1 || { echo "quantize failed"; tail -3 bench/raw/mac-quant-$N.log; continue; }
  # publish only the finished file (a fetch of a file still being written once delivered a truncated IQ1_S)
  mv "$SV/.w-$N.gguf.tmp" "$SV/w-$N.gguf"
  WANT=$(stat -f %z "$SV/w-$N.gguf")
  grep -E "quant size|total time" bench/raw/mac-quant-$N.log
  # (check only llama-quantize processes: a grep over every /proc/*/cmdline would match this adb shell itself)
  if adb -s $S shell "cd /data/local/tmp/llm; [ -f w-$N.gguf ] && exit 0; for p in \$(pidof llama-quantize); do grep -q w-$N.gguf /proc/\$p/cmdline && exit 0; done; exit 1" ; then
    echo "TV already has / is making w-$N.gguf; skipping hand-off"
  else
    adb -s $S shell "cd /data/local/tmp/llm && rm -f w-$N.gguf.part && ./tvfetch ${MAC_IP:?set MAC_IP=<mac-ip>} 8765 /w-$N.gguf w-$N.gguf.part >/dev/null 2>&1 && [ \$(stat -c %s w-$N.gguf.part) = $WANT ] && mv w-$N.gguf.part w-$N.gguf && ls -la w-$N.gguf || echo 'hand-off FAILED (size mismatch)'"
  fi
  rm -f "$SV/w-$N.gguf"
done
echo "MAC LOWBIT DONE $(date +%T)"
