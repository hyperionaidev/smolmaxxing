#!/bin/bash
# Moon-Beam MTP speculation: draft acceptance with the deployed routing (simulated) and the loop fix, on the Mac. 8 seeds each.
cd "$(dirname "$0")/.."
B=third_party/llama.cpp/build-mac/bin; M=models/moe; export URL=http://127.0.0.1:8093 TXT=bench/raw/E108-mtp-texts.txt
R='{"repeat_penalty":1.15,"repeat_last_n":256}'
for N in 0 1 2; do
  SPEC=""; [ $N -gt 0 ] && SPEC="-md $M/mb-mtp-q4.gguf --spec-type draft-mtp --spec-draft-n-max $N"
  GGML_MOE_POOL=43 GGML_MOE_BIAS=0.25 LLAMA_MOE_USED=6 $B/llama-server -m $M/mb-q2-h60.gguf -t 4 -ngl 0 -nr -c 1024 -np 1 --port 8093 --reasoning off -cram 0 -fa on -ctk q8_0 -ctv q8_0 $SPEC > bench/raw/E108-server-$N.log 2>&1 &
  P=$!; for i in $(seq 1 200); do curl -s -m 2 $URL/health | grep -q ok && break; sleep 3; done
  SAMPLER="$R" python3 -I tools/loop_stats.py "Mac Q2 top-6 b0.25 rep1.15 MTP n=$N" 8 400
  kill $P; wait $P 2>/dev/null; sleep 2
done; echo MTP-MAC DONE
