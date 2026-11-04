#!/system/bin/sh
# E30: Qwen3.5-2B (all-Q4_K + H65 head) in the 32-bit Vulkan build with N of 24 layers on the Mali GPU (rest on the
# 32-bit CPU backend, mmap'd, no repack), tuned Mali settings. Every run under the GPU guard.
# usage: sh tv-gpu-2b.sh "0 8 16 20"      results: eval/gpu-2b.txt
cd /data/local/tmp/llm
S=eval/gpu-2b.txt
echo "== $(date)" >> $S
for N in ${1:-0 8 16 20}; do
  LIMIT_MB=170 timeout 400 sh tv-gpu.sh eval/gpu-2b-$N.log env GGML_VK_FORCE_MMVQ=1 GGML_VK_RM_KQ_INT=4 GGML_VK_GDN_LANES=16 \
     LLAMA_BENCH_COMMON_TOKENS=4000 ./vulkan32/llama-bench2 -m w-q4k-h65.gguf --no-host 1 --repack 0 -ngl $N -ub 64 -b 64 -t 3 \
     -p 0 -n 16 -r 2 > eval/gpu-2b-$N.echo 2>&1
  echo "ngl $N: $(grep tg16 eval/gpu-2b-$N.log | awk -F'|' '{print $(NF-1)}') $(grep -h 'tv-guard' eval/gpu-2b-$N.log | tail -1)" >> $S
  sync; sleep 5
done
echo "ALL DONE $(date)" >> $S
