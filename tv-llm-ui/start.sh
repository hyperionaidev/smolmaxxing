#!/bin/bash
# Start the live LLM page on the TV box: llama-server (patched build) + box stats loop + page in the TV browser.
#   ./start.sh                         default model (best known config)
#   MODEL=w-m2-h65.gguf ./start.sh      higher-quality mix (~5 % slower)
#   MODEL=w-q4k-h45.gguf ./start.sh     45K head: ~4 % faster, more non-English tokens it cannot produce (§17, E68)
#   MODEL=tb4-q20-hml.gguf ./start.sh   Ternary-Bonsai-4B, merged scales + 2-bit fields: 4.3 tok/s in chat, ~0.965 bits/byte (§18, E77)
#   MODEL=mb-q2-h60.gguf ./start.sh     Moon-Beam-8B, expert pager + pre-gated prefetch: ~7.2 tok/s on long coherent answers (§19)
#   MODEL=lm2-q2-h60.gguf ./start.sh    Ling-mini-2.0 16B MoE, same expert pager recipe (§20)
#   MODEL=b8-q1-hml.gguf ./start.sh     Bonsai-8B, LUT kernel, fully resident, headless mode: 5.2 tok/s in chat, 0.974 bits/byte (§19)
#   MODEL=file.gguf BIN=patched THREADS=3 EXTRA="..." ./start.sh
#   TV=1 ./start.sh                    also open the page in the TV browser (only fits next to small models)
#   SENV="LLAMA_STREAM_RESIDENT_MB=800 ..." ./start.sh   environment for the server (weight streaming, EXPERIMENT.md §14)
#   (default BIN=patched25 with SENV=LLAMA_STATE_F16=1; fp16 state is Qwen3.5-only and asserts with MTP speculation)
#   ./start.sh stop
set -euo pipefail
cd "$(dirname "$0")"
S="${ADB_SERIAL:?set ADB_SERIAL=<tv-ip>:5555}"
IP="${S%%:*}"                 # used for the health check (127.0.0.1 with "adb forward tcp:8090 tcp:8090")
LAN_IP="${TV_IP:-$(adb -s "$S" shell "ip -4 addr show wlan0" | awk '/inet /{sub(/\/.*/,"",$2); print $2}')}"
D=/data/local/tmp/llm
PORT=8090
MODEL="${MODEL:-w-q4k-h65p8.gguf}"   # Qwen3.5-2B, the best overall (§11, §16, §17); tools/rebuild-models.sh 2b builds it
if [ -z "${BIN:-}" ] && [ "$MODEL" = tb4-q20-hml.gguf ]; then
  # Ternary-Bonsai-4B (EXPERIMENT.md §18, E77): one averaged scale per 128 weights, all layers in 2-bit fields.
  BIN=patched29
  SENV="${SENV-GGML_Q2_0_SCALES=128 GGML_Q2_0_SCALES_AVG=1 GGML_Q2_0_F2_LAYERS=0-35}"
elif [ -z "${BIN:-}" ] && [ "$MODEL" = b8-q1-hml.gguf ]; then
  # Bonsai-8B (EXPERIMENT.md §19): LUT kernel, all weights resident from the repack cache (first start writes it, ~1 GB of flash),
  # headless mode for the RAM. "tools/tv-llm-mode.sh off" on the TV restores the launcher and the screensaver.
  BIN=patched41
  SENV="${SENV-GGML_REPACK_CACHE=$D/cache/b8l GGML_REPACK_RESIDENT_MB=962}"
  EXTRA="${EXTRA:--ctk q8_0 -ctv q8_0 -fa on -cram 0}"
  LIMIT_MB="${LIMIT_MB:-150}"
  HEADLESS=1
elif [ -z "${BIN:-}" ] && [ "$MODEL" = mb-q2-h60.gguf ]; then
  # Moon-Beam-8B-A1B (EXPERIMENT.md §19): MoE, Q2_K experts, expert pager on the repack cache, cache-aware routing, warm start.
  # EXPERTS=8 BIAS=0.2 reads better at ~6.9 tok/s. Below 6 experts the text falls apart faster than the speed rises (top-2: 10.4 tok/s, gibberish).
  # Repeat penalty 1.15 over 256 tokens: without it Q2_K experts fall into loops in most long answers (E103-E107).
  BIN=patched41
  SENV="${SENV-GGML_MOE_PAGER=1 GGML_MOE_POOL=${POOL:-43} GGML_MOE_BIAS=${BIAS:-0.25} LLAMA_MOE_USED=${EXPERTS:-6} GGML_MOE_PREFETCH_K=6 GGML_MOE_PREFETCH_AHEAD=2 GGML_MOE_WARM=$D/moe-warm-moonbeam.txt GGML_REPACK_CACHE=$D/cache/mbq2}"
  EXTRA="${EXTRA:--ub 4 -b 64 -ctk q8_0 -ctv q8_0 -fa on -cram 0 -ctxcp 0 --repeat-penalty 1.15 --repeat-last-n 256}"
  LIMIT_MB="${LIMIT_MB:-150}"
  HEADLESS=1
elif [ -z "${BIN:-}" ] && [ "$MODEL" = lm2-q2-h60.gguf ]; then
  # Ling-mini-2.0 (EXPERIMENT.md §20): 16.3 B MoE, 256 experts of 1 MB, same recipe as Moon-Beam: ~5.1 tok/s. EXPERTS=6 BIAS=0.25: 5.3, much worse text.
  BIN=patched42
  SENV="${SENV-GGML_MOE_PAGER=1 GGML_MOE_POOL=${POOL:-43} GGML_MOE_BIAS=${BIAS:-0.3} LLAMA_MOE_USED=${EXPERTS:-8} GGML_MOE_PREFETCH_K=6 GGML_MOE_PREFETCH_AHEAD=2 GGML_MOE_WARM=$D/moe-warm-ling.txt GGML_REPACK_CACHE=$D/cache/lmq2}"
  EXTRA="${EXTRA:--ub 4 -b 64 -ctk q8_0 -ctv q8_0 -fa on -cram 0 --repeat-penalty 1.15 --repeat-last-n 256}"
  LIMIT_MB="${LIMIT_MB:-150}"
  HEADLESS=1
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
  A 'p=$(pidof llama-server); [ -n "$p" ] && kill $p; true'
}
stop_all
[ "${1:-}" = "stop" ] && { echo "stopped"; exit 0; }

# model facts for the page (head rows / MTP) straight from the GGUF header on the TV
hdr=$(A "head -c 48000000 $D/$MODEL" | python3 -I -c '
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

if [ "${HEADLESS:-0}" = 1 ]; then
  adb -s "$S" push ../tools/tv-llm-mode.sh "$D/" >/dev/null
  A "sh $D/tv-llm-mode.sh on"
else
  ../tools/tv-free-ram.sh
fi
A "mkdir -p $D/ui"
adb -s "$S" push index.html config.json "$D/ui/" >/dev/null
adb -s "$S" push ../bin/tv-watch ../tools/tv-guard.sh "$D/" >/dev/null
A "chmod 755 $D/tv-watch"
# always under the memory guard: an unguarded chat run left the box thrashing for 6 hours (§18.4)
A "cd $D && (setsid env $SENV BOX_JSON=$D/ui/box.json LIMIT_MB=${LIMIT_MB:-180} ./tv-watch server.log ./$BIN/llama-server -m $MODEL -t $THREADS -c $CTX -np 1 --host 0.0.0.0 --port $PORT --path $D/ui --metrics --reasoning off $EXTRA </dev/null >/dev/null 2>&1 &) </dev/null >/dev/null 2>&1"
echo -n "loading $MODEL "
for i in $(seq 1 180); do curl -s -m 2 "http://$IP:$PORT/health" | grep -q '"ok"' && break; echo -n .; sleep 2; done; echo
curl -s -m 2 "http://$IP:$PORT/health"; echo
[ "${TV:-0}" = "1" ] && A "am start -a android.intent.action.VIEW -d http://$LAN_IP:$PORT/ com.internet.tvbrowser" >/dev/null
echo "Page: http://$LAN_IP:$PORT/  (open the same URL from the Mac or a phone)"
