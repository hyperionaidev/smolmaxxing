#!/system/bin/sh
# E30 GPU kernel sweep: one guarded llama-bench (tg16) per Vulkan mat-vec setting, 0.8B fully on the GPU except
# the token embedding. usage: CFGS="A=1,B=2 C=3" BENCH=... sh tv-gpu-sweep.sh [model] [ngl]
# (configs separated by spaces, variables inside one config by commas)      results: eval/gpu-sweep.txt
cd /data/local/tmp/llm
M=${1:-Qwen3.5-0.8B-Q4_K_M.gguf}; NGL=${2:-24}
S=eval/gpu-sweep.txt
echo "== $(date) $M ngl=$NGL" >> $S
i=0
B=${BENCH:-./vulkan32/llama-bench}
CFGS=${CFGS:-"BASE=1"}
for cfg in $CFGS; do
  cfg=$(echo "$cfg" | tr ',' ' ')
  i=$((i+1))
  LIMIT_MB=170 sh tv-gpu.sh eval/gpu-sweep-$i.log env $cfg LLAMA_BENCH_COMMON_TOKENS=4000 $B -m $M \
     --no-host 1 --repack 0 -ngl $NGL -ub 64 -b 64 -t 3 -p 0 -n 16 -r 2 > /dev/null 2>&1
  echo "$cfg: $(grep tg16 eval/gpu-sweep-$i.log | awk -F'|' '{print $(NF-1)}') $(grep -h 'tv-guard' eval/gpu-sweep-$i.log | tail -1)" >> $S
  sleep 5
done
echo "ALL DONE $(date)" >> $S
