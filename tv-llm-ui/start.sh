#!/bin/bash
# Start the live LLM page on the TV box: llama-server (patched build) + box stats loop + page in the TV browser.
#   ./start.sh                         default model (best known config)
#   MODEL=w-m2-h65.gguf ./start.sh      higher-quality mix (~5 % slower)
#   MODEL=w-q4k-h45.gguf ./start.sh     45K head: ~4 % faster, more non-English tokens it cannot produce (§17, E68)
#   MODEL=tb4-q20-hml.gguf ./start.sh   Ternary-Bonsai-4B, merged scales + 2-bit fields: 4.3 tok/s in chat, ~0.965 bits/byte (§18, E77)
#   MODEL=file.gguf BIN=patched THREADS=3 EXTRA="..." ./start.sh
#   TV=1 ./start.sh                    also open the page in the TV browser (only fits next to small models)
#   SENV="LLAMA_STREAM_RESIDENT_MB=800 ..." ./start.sh   environment for the server (weight streaming, EXPERIMENT.md §14)
#   (default BIN=patched25 with SENV=LLAMA_STATE_F16=1; fp16 state is Qwen3.5-only and asserts with MTP speculation)
#   ./start.sh stop
set -euo pipefail
cd "$(dirname "$0")"
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
IP="${S%%:*}"
D=/data/local/tmp/llm
PORT=8090
MODEL="${MODEL:-w-q4k-h65p8.gguf}"   # final config (EXPERIMENT.md §11.8, §17): all-Q4_K + 65K multilingual head (padded to a
                                     # multiple of 8 rows so it is repacked, E71) + MTP block
if [ -z "${BIN:-}" ] && [ "$MODEL" = tb4-q20-hml.gguf ]; then
  # Ternary-Bonsai-4B (EXPERIMENT.md §18, E77): one averaged scale per 128 weights, all layers in 2-bit fields.
  BIN=patched29
  SENV="${SENV-GGML_Q2_0_SCALES=128 GGML_Q2_0_SCALES_AVG=1 GGML_Q2_0_F2_LAYERS=0-35}"
elif [ -z "${BIN:-}" ]; then
  # default build (EXPERIMENT.md §16): state read by id, guided matmul scheduling, single-row ops split across
  # threads, fp16 DeltaNet state (+11 % in chat vs patched5), fused one-token conv step, embedding rows read
  # from the resident (repacked) head (§17). SENV= (empty) turns the fp16 state off.
  BIN=patched25
  SENV="${SENV-LLAMA_STATE_F16=1}"
fi
THREADS="${THREADS:-3}"
CTX="${CTX:-1024}"
TEMP="${TEMP:-0.7}"
EXTRA="${EXTRA:--ub 128 -b 128}"
SENV="${SENV:-}"                       # an explicit BIN gets no default environment
A(){ adb -s "$S" shell "$@"; }

stop_all() {
  A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; [ -f '$D'/ui/boxstats.pid ] && kill $(cat '$D'/ui/boxstats.pid) 2>/dev/null; rm -f '$D'/ui/boxstats.pid; true'
}
stop_all
[ "${1:-}" = "stop" ] && { echo "stopped"; exit 0; }

# model facts for the page (head rows / MTP) straight from the GGUF header on the TV
hdr=$(A "head -c 16000000 $D/$MODEL" | python3 -I -c '
import sys, importlib.util
spec = importlib.util.spec_from_file_location("g", "../tools/gguf_info.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
_, meta, T = g.parse(sys.stdin.buffer.read())
out = [t for t in T if t[0] == "output.weight"]
mtp = [t for t in T if ".nextn." in t[0]]
name = meta.get("general.name", "LLM")
rows = out[0][1][1] if out else 0
mb = round(out[0][4] / 1e6) if out else 0
mtpmb = round(sum(t[4] for t in T if t[0].startswith("blk.%d." % (meta.get("qwen35.block_count", 0) - 1))) / 1e6) if mtp else 0
print(name, rows, mb, mtpmb)')
read -r NAME ROWS HMB MTPMB <<<"$hdr"
MTP=""; [ "$MTPMB" != "0" ] && MTP="1 layer, ${MTPMB} MB"
cat > config.json <<EOF
{"model_name": "$NAME", "threads": $THREADS, "temp": $TEMP, "patched": true,
 "head_rows": $ROWS, "head_mb": $HMB, "mtp": "$MTP", "flags": "$EXTRA",
 "notes": "pruned LM head; repacked weights; $( [ -n "$EXTRA" ] && echo "$EXTRA" || echo "no speculation")"}
EOF

../tools/tv-free-ram.sh
A "mkdir -p $D/ui"
adb -s "$S" push index.html config.json boxstats.sh "$D/ui/" >/dev/null
A "cd $D && (setsid sh ui/boxstats.sh $D/ui </dev/null >/dev/null 2>&1 & echo \$! > ui/boxstats.pid) </dev/null >/dev/null 2>&1"
A "cd $D && (setsid env $SENV ./$BIN/llama-server -m $MODEL -t $THREADS -c $CTX -np 1 --host 0.0.0.0 --port $PORT --path $D/ui --metrics --reasoning off $EXTRA </dev/null >server.log 2>&1 &) </dev/null >/dev/null 2>&1"
echo -n "loading $MODEL "
for i in $(seq 1 180); do curl -s -m 2 "http://$IP:$PORT/health" | grep -q '"ok"' && break; echo -n .; sleep 2; done; echo
curl -s -m 2 "http://$IP:$PORT/health"; echo
[ "${TV:-0}" = "1" ] && A "am start -a android.intent.action.VIEW -d http://$IP:$PORT/ com.internet.tvbrowser" >/dev/null
echo "Page: http://$IP:$PORT/  (open the same URL from the Mac or a phone)"
