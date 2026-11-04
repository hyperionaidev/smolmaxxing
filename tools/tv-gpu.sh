#!/system/bin/sh
# Every GPU (Vulkan) run on the TV goes through this: the 32-bit Vulkan build pins its buffers, and an unguarded
# full offload of the 2B panicked the kernel (E30). Guard samples 5x/s, prints samples (GUARD_ECHO) so the Mac
# keeps them even if the box dies, and kills the job below LIMIT_MB (default 250).
# usage: sh tv-gpu.sh <log> <command...>
cd /data/local/tmp/llm
GUARD_ECHO=1 GUARD_INTERVAL=0.2 LIMIT_MB=${LIMIT_MB:-250} sh tv-guard.sh "$@"
RC=$?; sync; exit $RC
