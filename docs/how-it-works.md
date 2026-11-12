# How it works

Everything here is a patch on llama.cpp: [`patches/llama.cpp-v0.8.0-tvbox.patch`](../patches/llama.cpp-v0.8.0-tvbox.patch) for the TV box and [`ipad-air/patches/llama.cpp-ios12.patch`](../ipad-air/patches/llama.cpp-ios12.patch) for the iPad. This page describes the TV box version, where most of it was built. The iPad's ports and additions are in [its README](../ipad-air/README.md#how-it-got-fast).

[← back to the main README](../README.md)

- [Kernels](#kernels)
- [Saving RAM](#saving-ram)
- [Expert pager for mixture-of-experts models](#expert-pager-for-mixture-of-experts-models)
- [Weight streaming for models bigger than RAM](#weight-streaming-for-models-bigger-than-ram)
- [Profiling and tooling](#profiling-and-tooling)
- [What the patch changes, file by file](#what-the-patch-changes)

## Kernels

Written for the Cortex-A55 (ARMv8.2 with dotprod, no i8mm). Each one has an exactness test against a scalar reference.

| Kernel | Before → after | Where |
|---|---|---|
| Q2_K, repacked 8 rows × 4 bytes for `vdotq_laneq` (upstream repacks Q2_K only on x86 / RISC-V) | 7.8 → 13.9 G weights/s; generation +41 %, prompt 2.4× | E34 |
| TQ1_0 / TQ2_0 ternary, repacked | 9.5 → 14.4 / 11.9 → 19.1 G weights/s | E38 |
| Ternary Q2_0 → **lossless 1.875 bits/weight** in RAM (base-3 packing that keeps the per-64 scales) | Ternary-Bonsai-4B from "thrashes" to 3.2 tok/s | E39, E42 |
| **Q1_0 bit planes**: `vtst` turns a bit plane straight into a mask for `vdot`, no table lookups; **8-bit row-affine scales** (−5.6 %, 0.07 % error) | Bonsai-4B 3.02 → 4.22 tok/s; Bonsai-8B fits in 0.96 GB | E44, E52 |
| Fused Gated-DeltaNet step with direct state write (Qwen3.5) | 2B 6.0 → 6.3-6.7 tok/s | E28 |
| DeltaNet state **read in place by id** instead of a 1 MB/layer gather every token; **fp16 state** (`LLAMA_STATE_F16=1`, on by default in the UI) | 0.8B 11.9 → 12.9 tok/s; 2B +2.5 % for +0.15 % perplexity | E56, E61, E65 |
| **The LM head on the fast kernel**: the 65,668-row head was never repacked (ggml repacks only row counts divisible by 8), so 4 token rows were added; embedding rows are then rebuilt from the repacked head instead of read from flash; the one-token DeltaNet conv step fused from 4 nodes into one NEON op | 2B +7 % (bench and chat); 0.8B +9.5 % | E69-E72 |
| **Ternary weights in 2-bit fields, with scales merged per 128**: on the in-order A55, one base-3 digit costs ~5 NEON ops per 16 weights and a 2-bit field costs 2, so spending 13 % more RAM buys speed. Merging adjacent scales costs +0.18 % perplexity | Ternary-Bonsai-4B 3.2 → 4.3 tok/s in chat, prompt 7.0 → 9.6 | E76, E77 |
| **1-bit lookup-table kernel** (Four-Russians / T-MAC style): 32 rows per group, one 16-entry table per 4 activations built straight from the float activations, ~8 NEON ops + 2 loads per 128 multiply-adds vs 16 + 4 | Bonsai-8B 3.77 → 5.1-5.4 tok/s (with E82, E88) | E81 |
| **Q8_0 attention with a threshold**: all K dots first with `sdot`, so the softmax maximum is known up front, then `expf` and V work skipped for rows below max + ln(1e-4) | a 1,000-token answer no longer halves the speed (2.50 → ~4 tok/s at the end) | E86 |
| **Guided matmul scheduling**: each thread grabs remaining / (2 × threads) rows, so a slow or preempted thread leaves a short tail instead of ~28 ms per token of idle cores; **single-row element-wise ops split by columns** (ggml left them all on thread 0) | 2B 6.26 → 6.54 tok/s; 0.8B +1.4 % | E63, E64 |

## Saving RAM

- **Pruned LM head.** Keep only the ~49-66 K most useful vocabulary rows, chosen by token frequency on English, code and 7 other languages. The other logits are set to −inf. Saves 0.2-0.5 GB and misses about 1 % of English tokens.
- **Repack cache on flash** (`GGML_REPACK_CACHE`). Repacked weights are written to a file once and mapped afterwards, so models load in seconds.
- **Headless mode** (`tools/tv-llm-mode.sh on|off`). A small foreground activity starts, then the launchers, Google services and other heavy apps are force-stopped, and `svc power stayon` keeps the animated screensaver (17 % of a core) away. Nothing persists except stay-awake, which `off` restores. Free RAM goes from about 1,000 to about 1,300 MB.
- **A fork-free memory guard** (`tools/tv-watch.c`). It replaced a shell loop that started about 25 processes a second and cost 10 % of the speed.

## Expert pager for mixture-of-experts models

In `ggml-cpu/moe-pager-tv.inc`. A mixture-of-experts model picks a few "experts" per layer for each token, so most of its weights sit unused at any moment. The pager keeps a pool of experts in RAM and reads the rest from flash when the router asks for them.

- **Experts live on flash.** Each expert tensor is overlaid with empty anonymous memory. Right after a layer's routing, the missing experts are read with parallel `O_DIRECT` straight into their own addresses, and a per-layer LRU pool (43 of 128 experts) keeps the rest. The kernels see an ordinary tensor. The GGUF's own expert bytes are hole-punched once the repack cache holds them, so the model costs one copy of its weights on flash.
- **Cache-aware routing** (`GGML_MOE_BIAS`). A small bonus for experts already in RAM cuts misses from 13 % to about 2 %, for a measured quality cost (arXiv 2412.00099 has the idea on other models). Using fewer experts per token (`LLAMA_MOE_USED=6`) is the other dial.
- **Warm start and pre-gated prefetch.** The pool starts from a list of each layer's most used experts. Then layer L+2's router is applied to layer L's hidden state, predicting what the router will actually pick (sigmoid, the model's own selection bias and the cache bonus). That loads about 60 % of misses before they're needed.
- **Why long answers looped, and the fix.** 2-bit experts break long-form coherence after about 250-300 words, and routing bias makes it worse. DRY sampling only hides it (the model paraphrases its loop). A repeat penalty of 1.15 over 256 tokens brings degenerate long answers from 12 in 13 to 1 in 13 (E99, E100).

## Weight streaming for models bigger than RAM

In `ggml-cpu/stream-tv.inc`. For dense models that don't fit:

- **What stays in RAM.** A chosen part of the weights stays resident; the rest lives in "stream zones".
- **Prefetch.** A hook on every matmul records the order the weights are used. Reader threads then fetch zone weights ahead of time with parallel 8 MB `O_DIRECT` reads straight into the weights' own addresses, at the flash's full 278 MB/s.
- **Correctness.** Every compute thread waits until its weight is completely loaded, so a wrong prediction costs time, never correctness.
- **Drop-behind.** Each zone weight is freed right after use.
- **Verify mode.** `GGML_STREAM_VERIFY=1` compares every streamed tensor with the file. It found two silent-corruption bugs, fixed and documented in E48 and E50.
- **Speculative decoding.** With the model's own MTP head, verifying 2 tokens costs about one pass over the weights when the pass is flash-bound, so it pays off (+69 %). On the in-RAM 2B it lost (E31).

## Profiling and tooling

**Profiling** (environment switches in the patch): `GGML_OPPROF=1` prints a per-op table, `GGML_OPPROF=2` a node-by-node, per-thread trace of one graph (`tools/optrace.py`), and `LLAMA_UBATCH_PROF` / `GGML_CPU_PROF` phase timings. They showed where a small model's time really goes (E57): about 71 % in Q4_K matmuls, and only about 2 ms per token in barriers and scheduling.

**Tools:**
- **Streaming GGUF rewriter.** Adds a pruned head, extra tensors and metadata in one sequential pass, so Hugging Face → Mac → TV pipelines never store the model on the Mac.
- **MTP fetcher.** Range-reads only the MTP tensors from safetensors.
- **Memory guard.** Wraps every heavy run on the TV.
- **Benchmark and perplexity harnesses**, a model screener that streams any Hugging Face GGUF to the TV and scores it (`tools/screen-small.sh`), and a live token-stream web UI.

## What the patch changes

| Path | What |
|---|---|
| `patches/llama.cpp-v0.8.0-tvbox.patch` | every change to llama.cpp for the TV: pruned head (`output_ids`) for Qwen3 / Qwen3.5 / Granite 4.0 H, the kernels above (`ggml-cpu/repack-*-tv.inc`), fused DeltaNet (state read by id, optional fp16 state, one-token conv step), the 1-bit LUT kernel, Q8_0 attention, the MoE expert pager with routing bias and prefetch (`ggml-cpu/moe-pager-tv.inc`), a pruned head for `bailingmoe3`, embedding rows from the resident head, repack cache and streaming engine (`ggml-cpu/stream-tv.inc`, loader hooks), padded small batches, op profiler, pruned-head-aware perplexity, Vulkan tuning for Mali |
| `tools/` | GGUF rewriter / head pruner / re-cutter, vocabulary selection, corpus and MTP fetchers, model rebuild script, TV memory guard and RAM helper, benchmark sweeps, kernel benches and exactness checks (`kbench/`), flash benchmark |
| `tv-llm-ui/` | the live token-stream page for llama-server (confidence colours, top-3 alternatives, speed, box stats), its launcher and the benchmark harness |
| `tiny-llm-web/` | the llama2.c story server and TV page from the first experiment |
| `bench/` | `results.jsonl` (every chat benchmark: prompts, outputs, speeds) and `raw/` (logs of every experiment) |
| `data/` | head-id lists and the two evaluation texts |
| `screenshots/` | the web pages running on the TV |

[EXPERIMENT.md](../EXPERIMENT.md) has the full write-up: §1-8 the first tiny model, §9-10 llama.cpp and the first flash streaming, §11 the 2B push, §13 the 4B push, §14 the 9B push, §15 the 0.8B speed push, §16-17 the 2B rounds 3-4, §18 the 4B round 2, §19 the 8B push (headless mode, LUT kernel, Moon-Beam with an expert pager, why long answers loop), §20 bigger than 9B (Ling-mini-2.0), and §21 the iPad Air.
