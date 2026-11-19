# 📺 Device #4: BuzzTV XR4500

Amlogic SM1 (S905X3 class): 4× Cortex-A55, **2 GB of RAM** (about 1.3 GB free), stock Android 9 running **32-bit (armv7l)** userspace on kernel 4.9.113, no root. Everything runs from `/data/local/tmp` over adb on Wi-Fi.

This is the first community-submitted device and the first 32-bit one. The numbers were measured on the box by [kbhat97](https://github.com/kbhat97) ([#6](https://github.com/hyperionaidev/smolmaxxing/issues/6)), who also ported the patch to 32-bit ARM ([#8](https://github.com/hyperionaidev/smolmaxxing/pull/8)).

[← back to the main README](../README.md) · [how it works](how-it-works.md) · [the 64-bit TV box](tv-box.md)

## At a glance

The same 4× Cortex-A55 as the [Xiaomi TV Box](tv-box.md), but a 32-bit OS throws away most of what makes that box fast:

- **Qwen3-0.6B Q8_0: 1.7 tok/s**, about 750 MB peak RSS. The everyday model.
- **Qwen3.5-2B (the TV box's repacked file): 0.4 tok/s.** It runs, but the TV box's 2-bit kernels aren't there to speed it up.
- **Q8_0 beats Q4_K_M by 2×** (1.7 vs 0.8 tok/s), the reverse of every 64-bit device here.

## Results

| Model | File | tok/s (generation) | Prompt tok/s | Peak RSS |
|---|---|---|---|---|
| Qwen3-0.6B | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) Q8_0 | **1.7** (1.6-2.0 seen) | 2.0-2.3 | ~750 MB |
| Qwen3-0.6B | Q4_K_M | 0.8 | | |
| Qwen3.5-2B | the TV box file ([models.md](models.md)) | 0.4 | | |

`llama-cli -t 4 -c 1024 -n 128`; three runs of Qwen3-0.6B Q8_0 gave 1.8, 1.7 and 1.7 tok/s, and the middle one is reported. Speeds are generation with a one-word prompt, not a real chat.

## Why Q8_0 is the fastest here

The repacked fast kernels are arm64-only: the patch's Q2_K and 1-bit ones use instructions 32-bit NEON doesn't have (`vqtbl1q_u8`, `vaddw_high_u8`, `vmlal_high_*`), so a 32-bit build compiles them out, and Q4_K and Q2_K weights fall back to slow paths. Q8_0 still runs llama.cpp's own vectorized dot product, so it ends up the fastest format despite being the biggest. On a 32-bit device, pick Q8_0 and a model small enough to fit it.

## Build

The [TV box build](tv-box-setup.md#1-build-llamacpp-with-the-patch) with the 32-bit zig wrappers in `zigcc32/` (`arm-linux-musleabihf`, Cortex-A55) and no `GGML_CPU_ARM_ARCH`:

```sh
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
cd third_party/llama.cpp && git checkout 8642541 && git apply ../../patches/llama.cpp-v0.8.0-tvbox.patch
cmake -S . -B build-armv7 -G Ninja -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=armv7 \
  -DCMAKE_C_COMPILER=$PWD/../../zigcc32/cc -DCMAKE_CXX_COMPILER=$PWD/../../zigcc32/c++ -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-static -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF
ninja -C build-armv7 llama-server llama-cli llama-bench
cd ../..
adb -s $ADB_SERIAL shell mkdir -p /data/local/tmp/llm/patched25
adb -s $ADB_SERIAL push third_party/llama.cpp/build-armv7/bin/llama-{server,cli,bench} /data/local/tmp/llm/patched25/
```

The result is a static `ELF 32-bit LSB executable, ARM, EABI5`. A 64-bit build fails on this box with "not executable: 64-bit ELF file".

## Run

```sh
adb -s $ADB_SERIAL shell 'cd /data/local/tmp/llm && ./patched25/llama-cli -m q3-06b-q8.gguf -p test -n 128 -t 4 -c 1024'
```

**Things to know:**

- **Use all 4 threads** (`-t 4`, one per physical core).
- **Leave `LLAMA_STATE_F16` unset** (in the repo's scripts, run with `SENV=` empty). The fp16 recurrent state needs arm64 NEON and asserts `fp16 GDN state needs aarch64 NEON` on 32-bit. Dense models like Qwen3-0.6B don't use it either way.
- **`llama-server` works too**, and the box survived a cold power cycle: the server was relaunched over Wi-Fi adb and healthy again in about 2 minutes.

## What 32-bit ARM needed

Four changes to the patch ([#8](https://github.com/hyperionaidev/smolmaxxing/pull/8)), all behind guards that leave 64-bit builds unchanged:

| Change | Why |
|---|---|
| `vcvtnq_s32_f32` fallback skipped under clang | clang's 32-bit `arm_neon.h` already defines it |
| 32-bit shims for `vmovl_high_s8`, `vmovl_high_s16`, `vdotq_laneq_s32` | arm64-only forms the patch's kernels use; zig defines `__ARM_FEATURE_DOTPROD` for a 32-bit Cortex-A55 target, but clang ships no 32-bit dot-product intrinsics |
| `&& defined(__aarch64__)` on the dot-product guards in `repack-q2k-tv.inc` and `repack-q1lut-tv.inc` | their fast paths use arm64-only instructions; the fallbacks are correct on 32-bit |
| `long` → `long long` (and `%ld` → `%lld`) for the memory guard's counters in `tools/tv-watch.c` | `long` is 32 bits on armv7, so `1L << 40` overflowed |
