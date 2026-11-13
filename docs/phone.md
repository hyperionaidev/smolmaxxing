# 📞 Device #3: OnePlus Nord CE 2

MediaTek Dimensity 900: 2× Cortex-A78 at 2.4 GHz and 6× Cortex-A55 at 2.0 GHz (dotprod, no i8mm), **7.4 GB of RAM** (about 3 GB free with the phone's own apps running), UFS 2.2 flash, Android 13, no root. Everything runs on the CPU as the adb `shell` user, over USB.

[← back to the main README](../README.md) · [lab notebook §22](../EXPERIMENT.md#22-device-3-a-oneplus-nord-ce-2-phone-16b-to-35b-2026-11-13) · [how it works](how-it-works.md) · [downloads](models.md#phone)

## At a glance

On this phone the repo runs mixture-of-experts models from 16B to 80B parameters, with most of each model on flash:

- **Ling-mini-2.0, 16B: 11.1 tok/s** with exact routing, **15.3 tok/s** with the TV box's speed settings (+17 % perplexity). Three times the TV box.
- **Qwen3.6-35B-A3B, 35B: 4.9 tok/s** with exact routing, **5.7 tok/s at +0.7 % perplexity**. The best model per answer on the phone.
- **Qwen3-Next-80B-A3B, 80B: 2.8 tok/s** with exact routing (first run: 2 prompts of 160 tokens, 5.1 GB peak RSS).

Speeds are generation in a real chat with the server's repeat penalty, 2 prompts × 160 tokens per setting.

## Results

| Model | Total / active | File | Routing | tok/s | Perplexity (eval-en) | Peak RSS |
|---|---|---|---|---|---|---|
| Ling-mini-2.0 | 16B / 1.4B | 5.5 GB (TV file) | exact, pool 128 | **11.1** | 13.92 | 3.7 GB |
| | | | bias 0.25, 6 experts | **15.3** | 16.31 (+17 %) | 3.4 GB |
| Qwen3.6-35B-A3B | 35B / 3B | 12.3 GB | exact, pool 80 | **4.9** | 7.63 | 4.8 GB |
| | | | bias 0.005 | **5.7** | 7.68 (+0.7 %) | 4.9 GB |
| | | | bias 0.01 | **6.0** | 7.78 (+2.0 %) | 5.2 GB |
| Qwen3-Next-80B-A3B | 80B / 3B | 27.6 GB | exact, pool 48 | **2.8** | not measured yet | 5.1 GB |

"Pool" is the number of experts per layer kept in RAM; the rest are read from flash when the router picks them. "Bias" nudges the router toward experts already in RAM. Perplexity was measured on the phone over 40 (Ling) or 20 (Qwen3.6) chunks of 128 tokens; it isn't comparable between models.

## What limits this phone

| | Measured | What it means |
|---|---|---|
| Flash | O_DIRECT random reads top out at **~940 MB/s**, but only with 2-4 reads in flight of 256 KB or more. One reader: 515 MB/s at 512 KB. 4 KB reads: 28-81 MB/s | Experts are read whole, several at once, by a pool of reader threads |
| RAM bandwidth | Q4_K matmuls run at **13.9 GB/s** (LPDDR4X) | With the experts cached, reading Qwen3.6's 1.1 GB of non-expert weights every token is the biggest cost (41 % of a token) |
| CPU | 2 fast cores and 6 slow ones | **2 threads pinned to the A78s beat every other mix** (Ling: 12.0 tok/s vs 7.9 at 4 threads, 6.7 at 8): the A55s hold the A78s back at every barrier |
| Memory | ~3 GB free at idle; the shell can't be OOM-killed and can `mlock` without limit | The engine can take ~5 GB if Android gives back its caches. Holding too much starves Android instead, so every run has a memory guard |
| Kernel | 4.19 | No io_uring |

## What helped, and what didn't

| Change | Effect |
|---|---|
| 2 threads on the A78 cores (`-t 2 -C 0xC0 --cpu-strict 1`) | Ling 7.9 → 12.0 tok/s |
| Bigger expert pool (Ling 40 → 128, Qwen3.6 64 → 80 per layer) | Ling 12.0 → 15.3, Qwen3.6 4.7 → 5.0 |
| The TV box's fused DeltaNet conv step and in-place state read, ported to `qwen35moe` and `qwen3next` | Qwen3.6 **3.6 → 4.7 tok/s**, bit-identical output |
| A routing bias on Qwen's scale (0.005 instead of Ling's 0.25: Qwen's router is a softmax over 256 experts) | Misses halved for +0.7 % perplexity: 4.9 → 5.7 tok/s |
| The whole LM head instead of the 60K-row pruned one | 3 % slower, but the pruned head made Qwen3.6 write "palinate" for "palindrome": a 35B model uses rarer whole-word tokens |
| More reader threads (4 → 8 → 12), readers on the A55s, an fp16 copy of the routers for the prefetch predictor, predicting two layers ahead and one | No measurable change |
| Static expert pruning by use count (keeping 192 of 256 per layer) | **+34 % perplexity**: 3,840 calibration tokens are too few to rank experts. Dropped |

## Running it

Build the TV box's static binaries ([TV setup, step 1](tv-box-setup.md#1-build-llamacpp-with-the-patch)), connect the phone by USB with USB debugging on, then:

```sh
adb shell mkdir -p /data/local/tmp/llm/bin /data/local/tmp/llm/cache
adb push third_party/llama.cpp/build-tv/bin/llama-{server,bench,perplexity} /data/local/tmp/llm/bin/
adb push bin/tv-watch data/eval/eval-en.txt /data/local/tmp/llm/

# Qwen3.6-35B-A3B straight onto the phone (nothing big is stored on the Mac; ~12 GB over USB, ~20 min)
HEAD=full tools/q36-phone.sh
# Qwen3-Next-80B-A3B (~28 GB)
MR=https://huggingface.co/mradermacher/Qwen3-Next-80B-A3B-Instruct-i1-GGUF/resolve/main STEM=Qwen3-Next-80B-A3B-Instruct TAG=q3n HEAD=full tools/q36-phone.sh

# on the phone, once: write the repack cache (a one-token bench), then chat through an adb port forward
adb shell 'cd /data/local/tmp/llm && GGML_REPACK_CACHE=$PWD/cache/q36f ./bin/llama-bench -m q36-q2-full.gguf -t 2 -C 0xC0 --cpu-strict 1 -p 0 -n 1 -r 1'
adb forward tcp:8090 tcp:8090
adb shell 'cd /data/local/tmp/llm && GGML_MOE_PAGER=1 GGML_REPACK_CACHE=$PWD/cache/q36f GGML_MOE_POOL=80 GGML_MOE_BIAS=0.005 \
  GGML_MOE_PREFETCH_K=8 GGML_MOE_PREFETCH_AHEAD=2 GGML_MOE_READERS=8 LIMIT_MB=400 ./tv-watch server.log \
  ./bin/llama-server -m q36-q2-full.gguf -c 1024 -np 1 --host 127.0.0.1 --port 8090 --reasoning off -cram 0 -fa on \
  -ctk q8_0 -ctv q8_0 -ub 4 -b 64 -t 2 -C 0xC0 --cpu-strict 1'
# then open http://127.0.0.1:8090 on the Mac
```

`tools/phone-moe.sh` runs the chat benchmarks above. Perplexity runs must use small micro-batches (`-ub 8`, or `-ub 4` for the 80B) under the memory guard: a 128-token batch makes the pager hold every expert the batch touches, and one such run restarted Android.

**Things to know:** the phone's `am kill-all` is denied for the shell user, so there's no headless mode as on the TV. The USB link dropped twice during long heavy runs (the phone kept running); a firm cable helps. Wireless adb from macOS needs Local Network permission for the terminal app.
