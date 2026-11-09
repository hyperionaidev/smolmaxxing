#!/bin/bash
# Cross-compile CPU-only llama.cpp and smolchat for the iPad Air (A7, iOS 12).
# Needs third_party/llama.cpp with patches/llama.cpp-ios12.patch applied, and
# third_party/iPhoneOS.sdk taken from Xcode 15.4. Output goes to bin/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SDK="$HERE/third_party/iPhoneOS.sdk"
SRC="$HERE/third_party/llama.cpp"
BUILD="$HERE/third_party/build-ios"
BIN="$HERE/bin"

[[ -d "$SDK" ]] || { echo "missing SDK at $SDK"; exit 1; }
[[ -d "$SRC" ]] || { echo "missing llama.cpp at $SRC"; exit 1; }

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
  -DCMAKE_MACOSX_BUNDLE=OFF \
  -DCMAKE_C_FLAGS=-DGGML_NO_FS \
  -DCMAKE_CXX_FLAGS=-DGGML_NO_FS \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8-a \
  -DGGML_METAL=OFF \
  -DGGML_ACCELERATE=OFF \
  -DGGML_BLAS=OFF \
  -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_COMMON=ON

nice -n 15 cmake --build "$BUILD" --target llama-simple -j 4

mkdir -p "$BIN"
/usr/bin/clang++ -std=c++17 -O2 -DNDEBUG \
  -arch arm64 -isysroot "$SDK" -miphoneos-version-min=12.0 \
  -I"$SRC/include" -I"$SRC/ggml/include" \
  "$HERE/smolchat.cpp" \
  "$BUILD/src/libllama.a" \
  "$BUILD/ggml/src/libggml.a" \
  "$BUILD/ggml/src/libggml-cpu.a" \
  "$BUILD/ggml/src/libggml-base.a" \
  -o "$BIN/smolchat"
cp "$BUILD/bin/llama-simple" "$BIN/llama-simple"
ls -l "$BIN"
