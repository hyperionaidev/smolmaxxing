#!/system/bin/sh
# E30: Qwen3.5-2B with N layers on the Mali GPU and the rest on the CPU. The 64-bit patched llama.cpp (fast repacked
# CPU kernels, pruned 65K head) talks over localhost RPC to the 32-bit Vulkan ggml-rpc-server, which owns the GPU.
# Run under tv-gpu.sh with GUARD_KILL_NAMES="ggml-rpc-server llama-bench" so both processes die on low memory.
# usage: sh tv-rpc-hybrid.sh <ngl> [model]
cd /data/local/tmp/llm
N=$1; M=${2:-w-q4k-h65.gguf}
env GGML_VK_FORCE_MMVQ=1 GGML_VK_RM_KQ_INT=4 GGML_VK_GDN_LANES=16 ./vulkan32/ggml-rpc-server -d Vulkan0 -H 127.0.0.1 -p 50052 > eval/rpc-server-$N.log 2>&1 &
S=$!
sleep 3
LLAMA_BENCH_COMMON_TOKENS=4000 ./patched4/llama-bench -m $M --rpc 127.0.0.1:50052 -ngl $N -ub 64 -b 64 -t 3 -p 0 -n 32 -r 2
RC=$?
kill $S 2>/dev/null; sleep 1; kill -9 $S 2>/dev/null
exit $RC
