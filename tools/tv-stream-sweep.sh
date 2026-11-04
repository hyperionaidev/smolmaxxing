#!/system/bin/sh
# Streaming sweep on the TV (run from /data/local/tmp/llm): each arg is cache:model:resident_MiB:window_MB[:readers]
# llama-bench pp32 + tg (N, default 16), 3 threads, ub 32, under tv-guard; prints the speed and the stream stats.
cd /data/local/tmp/llm
BIN=${BIN:-patched14}; N=${N:-16}
for cfg in "$@"; do
  set -- $(echo "$cfg" | tr : " "); C=$1; M=$2; R=$3; W=$4; RD=${5:-3}
  GGML_STREAM_MODE=${MODE:-direct} GGML_STREAM_READERS=$RD GGML_STREAM_WINDOW_MB=$W GGML_REPACK_RESIDENT_MB=$R GGML_REPACK_CACHE=/data/local/tmp/llm/cache/$C \
    sh tv-guard.sh sw-$C-$R-$W-$RD.log ./$BIN/llama-bench -m $M.gguf -t 3 -p 32 -n $N -r 1 -ub 32 -b 32 -v >/dev/null
  echo "== $M resident $R MiB window $W MB readers $RD mode ${MODE:-direct}"
  grep -E "^\|.*(pp|tg)[0-9]|tv-guard|stream:" sw-$C-$R-$W-$RD.log | sed "s/^| [^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*|[^|]*| *//"
done
