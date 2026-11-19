#!/bin/sh
# Cross-compile llama-bench, llama-perplexity and llama-tokenize for the Kindle Paperwhite 3
# (Cortex-A9, armv7 hard-float, NEON, no fp16 hardware) as static musl binaries.
# Uses Inkwell's toolchain: clone https://github.com/pragyaangaur/Inkwell, run its tools/fetch.sh
# (Alpine's musl, libc++ and compiler-rt as a sysroot, plus llama.cpp v0.7.0), and point INKWELL at it.
# Needs Homebrew llvm (for llvm-ar) and cmake + ninja. Output goes to kindle/bin/{stock,patched}.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
INKWELL=${INKWELL:?set INKWELL to an Inkwell checkout with tools/fetch.sh done}
SRC="$INKWELL/downloads/llama/src.tar.gz"   # the llama.cpp v0.7.0 release tarball from ggml-org
W="$HERE/third_party"; mkdir -p "$W" "$HERE/bin"
STRIP=$(command -v llvm-strip || echo /opt/homebrew/opt/llvm/bin/llvm-strip)

for v in stock patched; do
  rm -rf "$W/$v"; mkdir -p "$W/$v"; tar -xzf "$SRC" -C "$W/$v"
  L="$W/$v/llama.cpp-0.7.0"
  [ $v = patched ] && patch -d "$L" -p1 < "$HERE/patches/llama.cpp-v0.7.0-armv7-fp16.patch"
  cmake -S "$L" -B "$L/build-kindle" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$INKWELL/tools/kindle.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON
  cmake --build "$L/build-kindle" --target llama-bench llama-perplexity llama-tokenize
  mkdir -p "$HERE/bin/$v"
  for t in llama-bench llama-perplexity llama-tokenize; do "$STRIP" -o "$HERE/bin/$v/$t" "$L/build-kindle/bin/$t"; done
done
ls -l "$HERE/bin"/*
