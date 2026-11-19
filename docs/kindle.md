# 📖 Device #5: Kindle Paperwhite 3

NXP i.MX6 SoloLite with **one Cortex-A9 core at 996 MHz** (32-bit ARMv7, NEON and VFPv3, no fp16 arithmetic, no dot-product instructions), **512 MB of RAM** (about 440 MB free at idle), 4 GB eMMC that reads at 14.5 MB/s, and Amazon's Linux 3.0.35 kernel. The Kindle was jailbroken with WinterBreak and runs [Inkwell](https://github.com/pragyaangaur/Inkwell), an Alpine Linux userspace that replaces Amazon's UI. Everything here ran on the CPU as root over SSH, from static armv7 musl builds of stock llama.cpp v0.7.0.

[← back to the main README](../README.md) · [lab notebook §23](../EXPERIMENT.md#23-device-5-a-kindle-paperwhite-3-2026-11-19) · [scripts](../kindle/)

## At a glance

This is the slowest device in the repo by far, and the second 32-bit one after the [BuzzTV box](buzztv-xr4500.md). It is also the only one where a model runs every day as part of the device's own software: Inkwell ships a chatbot and a small coding assistant that run on the e-reader with no network.

- **SmolLM2-135M-Instruct: 1.71 tok/s.** This is the everyday chat model on the Kindle.
- **Qwen2.5-Coder-0.5B-Instruct: 0.57 tok/s.** This is the coding assistant. It writes Python or shell, runs it on the Kindle and reads the output back.
- **Bonsai-1.7B (1-bit): 0.20 tok/s.** This is the biggest model that fits in RAM. It is too slow for a chat.
- **Bonsai-4B (1-bit): 0.03 tok/s.** Its 572 MB file is bigger than the Kindle's RAM, and it runs with most of the model read from flash for every token.
- **A one-line patch to llama.cpp makes every model 1.4 to 1.6 times faster** on this chip. It is below.

## Results

Generation speed is `llama-bench -p 0 -n 128 -t 1`, with the median of three runs for the patched build (the runs differ by under 1 %). The page cache was dropped before every run. Prompt speed is `-p 64 -n 0`, one run. Bits per byte is the repo's usual quality measure on `data/eval/eval-en.txt` (40 chunks of 128 tokens), measured on the Mac because it takes hours per model on the Kindle and depends only on the file. Lower is better.

| Model | File | Size | Generation, patched | Generation, stock | Prompt, patched | Peak RSS | Bits per byte |
|---|---|---|---|---|---|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) | Q4_0 | 92 MB | **1.71** | 1.12 | 4.24 | 106 MB | 1.150 |
| [Gemma 3 270M IT](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF) | Q4_K_M | 253 MB | **0.84** | 0.58 | 3.65 | 272 MB | 1.376 |
| [SmolLM2-360M-Instruct](https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF) | Q4_0 | 230 MB | **0.73** | 0.46 | 1.52 | 242 MB | 1.040 |
| [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/bartowski/Qwen2.5-Coder-0.5B-Instruct-GGUF) | Q4_0 | 353 MB | **0.57** | 0.35 | 1.37 | 365 MB | 1.187 |
| [Bonsai-1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) (1-bit) | Q1_0 | 248 MB | **0.20** | 0.13 | 0.28 | 291 MB | 1.158 |
| [Bonsai-4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) (1-bit) | Q1_0 | 572 MB | 0.03 | | | 398 MB | 1.073 |

Bonsai-4B was run once with 8 generated tokens. The stock column is the median of three runs for the two SmolLM2 models and one run for the others, since a stock run of Bonsai-1.7B takes 16 minutes. The Bonsai bits per byte match E25 and E133 (1.158 and 1.073), which checks the method. Raw logs are in `bench/raw/E167-kindle-suite.log`, `E168-kindle-ppl.txt` and `E169-kindle-memory.log`.

## What limits this Kindle

| | Measured | What it means |
|---|---|---|
| CPU | 1 Cortex-A9 at 996 MHz, `ondemand` governor | One thread is all there is. 135M at 1.71 tok/s reads about 160 MB of weights per second, and the fp16 patch's gain shows that the per-block arithmetic sets the speed |
| fp16 | VFPv3 with no fp16 conversion instructions | Every Q4_0 and Q4_K block stores its scale in fp16. Stock llama.cpp converts them through a slow path on 32-bit Arm (the patch below) |
| Instruction set | ARMv7, 32-bit | The repo's fast kernels need AArch64 NEON. Since [#8](https://github.com/hyperionaidev/smolmaxxing/pull/8) the patch builds for armv7 with those kernels compiled out, as on the [BuzzTV box](buzztv-xr4500.md), but that build hasn't been tried on the Kindle. Everything here uses stock llama.cpp and ordinary GGUF files |
| RAM | 503 MB visible, about 440 MB free with Inkwell running | Everything up to Bonsai-1.7B fits with room to spare. Bonsai-4B's 572 MB file does not, so it runs from flash |
| Flash | 14.5 MB/s sequential read (`dd` after dropping the cache) | Bonsai-4B, which is bigger than RAM, ran at 0.03 tok/s (35 seconds per token) because most of the file is read from flash again for every token |
| Kernel | 3.0.35 | No `MemAvailable` in `/proc/meminfo`, so the memory guard counts free memory plus buffers and page cache |

## The patch

`ggml/src/ggml-cpu/simd-mappings.h` picks the hardware fp16 conversion when `__ARM_FP16_FORMAT_IEEE` is defined. Clang defines that macro on every 32-bit Arm target, including ones whose FPU has no fp16 instructions, so on the Cortex-A9 ggml falls back to a software conversion call for each block's scale. The patch also requires `__aarch64__` or bit 1 of `__ARM_FP` (hardware half precision), and with it ggml uses its lookup table instead. It is `kindle/patches/llama.cpp-v0.7.0-armv7-fp16.patch`:

```diff
-#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
+#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) && (defined(__aarch64__) || (__ARM_FP & 2)) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
```

| Model | Stock | Patched | Gain |
|---|---|---|---|
| SmolLM2-135M Q4_0 | 1.12 | 1.71 | 1.53× |
| SmolLM2-360M Q4_0 | 0.46 | 0.73 | 1.59× |
| Qwen2.5-Coder-0.5B Q4_0 | 0.35 | 0.57 | 1.61× |
| Gemma 3 270M Q4_K_M | 0.58 | 0.84 | 1.44× |
| Bonsai-1.7B Q1_0 | 0.13 | 0.20 | 1.50× |

It should help any 32-bit Arm board built with clang whose FPU lacks fp16 conversions, such as the Cortex-A8 and A9. Cores with VFPv4 or newer, like the Cortex-A7 or the BuzzTV box's A55 in 32-bit mode, define `__ARM_FP & 2` and keep the hardware path, so they're unchanged. The change also applies to current llama.cpp master.

## Running it

The builds use Inkwell's cross toolchain, which links static armv7 hard-float binaries against Alpine's musl and libc++ with Apple clang. Clone [Inkwell](https://github.com/pragyaangaur/Inkwell), run its `tools/fetch.sh`, then:

```sh
INKWELL=/path/to/Inkwell kindle/build.sh            # kindle/bin/{stock,patched}/llama-{bench,perplexity,tokenize}
scp -P 2222 kindle/bin/patched/* root@<kindle>:/root/smol/patched/
kindle/bench.sh /mnt/us/smol/models/model.gguf        # three runs of -p 0 -n 128 -t 1 under a memory guard
LLAMA_BIN=/path/to/mac/llama.cpp/bin kindle/mac-ppl.sh model.gguf   # bits per byte, on the Mac
```

`bench.sh` runs over SSH (`KINDLE_SSH`, default Inkwell's `./ssh.sh`). It kills a run if free memory drops below 20 MB or a run takes more than 20 minutes, and prints each run's speed, peak RSS and the lowest free memory.

**Things to know:** Inkwell's apps put the Kindle to sleep after 15 idle minutes, which drops Wi-Fi and USB, so stop the UI (`kill -STOP` on `home`) during long runs and continue it afterwards. On macOS 27.2 the Kindle's USB Ethernet gadget passed no packets, and the terminal app needed Local Network permission to reach it over Wi-Fi. The Mac still couldn't open a connection to the Kindle, so the Kindle connected out to a relay on the Mac and SSH ran through that.
