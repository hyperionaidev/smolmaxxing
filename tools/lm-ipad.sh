#!/bin/bash
# Build Ling-mini-2.0 (16.3B MoE) for the iPad Air's expert pager without storing the 5.5 GB model on the Mac
# (EXPERIMENT.md section 21.11). Peak Mac disk use is about 1.4 GB.
#  1. the non-expert tensors of bartowski's Q4_K_M (0.7 GB, range requests) -> requantized to Q4_K with his imatrix (the TV recipe)
#  2. 60K-row pruned LM head (same list as Moon-Beam)
#  3. experts from mradermacher's imatrix Q2_K_S (quantized from BF16), fetched in parallel, repacked for the q2lut kernel,
#     written experts-last and streamed over SSH to the iPad; the md5 printed at the end must match md5sum on the device
# usage: tools/lm-ipad.sh      (SSH tunnel on 127.0.0.1:2222, see ipad-air/README.md)
set -euo pipefail
cd "$(dirname "$0")/.."
KEY="${IPAD_KEY:-$HOME/ipadair/.ipad-ssh/id_ed25519}"
ipad(){ ssh -p "${IPAD_PORT:-2222}" -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o "UserKnownHostsFile=$(dirname "$KEY")/known_hosts" root@127.0.0.1 "$@"; }
B=ipad-air/third_party/build-mac/bin; M=ipad-air/models/ling; mkdir -p $M
BW=https://huggingface.co/bartowski/inclusionAI_Ling-mini-2.0-GGUF/resolve/main
MR=https://huggingface.co/mradermacher/Ling-mini-2.0-i1-GGUF/resolve/main/Ling-mini-2.0.i1-Q2_K_S.gguf

[ -f $M/lm2-imatrix.gguf ] || curl -sL -o $M/lm2-imatrix.gguf $BW/inclusionAI_Ling-mini-2.0-imatrix.gguf
python3 -I tools/gguf_fetch_subset.py $BW/inclusionAI_Ling-mini-2.0-Q4_K_M.gguf $M/lm2-ne-q4km.gguf --exclude _exps.
$B/llama-quantize --allow-requantize --pure --imatrix $M/lm2-imatrix.gguf --token-embedding-type q4_K --output-tensor-type q4_K \
  $M/lm2-ne-q4km.gguf $M/lm2-ne-q4k.gguf q4_K 8
python3 -I tools/gguf_build.py $M/lm2-ne-q4k.gguf $M/lm2-ne-h60.gguf --head-ids data/head-ids-ling-60k.txt
rm -f $M/lm2-ne-q4km.gguf $M/lm2-ne-q4k.gguf
python3 -I tools/gguf_moe_assemble.py $M/lm2-ne-h60.gguf $MR - --q2lut --jobs 10 \
  | ipad 'cat > /var/mobile/llama/lm2-q2-h60-lut.gguf.part'
ipad 'cd /var/mobile/llama && md5sum lm2-q2-h60-lut.gguf.part && mv lm2-q2-h60-lut.gguf.part lm2-q2-h60-lut.gguf'
