# EXPERIMENT: LLMs on the Xiaomi TV Box

> Paths below are relative to `llm-experiments/` (this folder). Device-wide logs live one level up: `../CHANGELOG.md`, `../rollback.sh`, `../REPORT.md`.

**Date:** 2026-11-03. **Device:** Xiaomi TV Box S 3rd Gen (`MiTV-AFMU0`), Amlogic S7D, 4× Cortex-A55 @ 2.5 GHz, 2 GB RAM, Android 14.
**Goal:** get a real, pretrained language model to produce a token on the box. Use the fewest parameters possible; speed doesn't matter.
**Result: ✅ it works.** A **260,032-parameter** Llama-architecture model writes complete short stories on the TV box at **~640 tokens/s**, using about 1 MB of weights.

## 1. The model: `stories260K`

The smallest published pretrained transformer LM I know of is Karpathy's **stories260K** ([karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)). It's a real Llama-2-style model trained on the TinyStories dataset, the smallest of the `llama2.c` reference models (260K / 15M / 42M / 110M).

| Hyperparameter | Value |
|---|---|
| `dim` (embedding width) | 64 |
| `hidden_dim` (FFN) | 172 |
| layers | 5 |
| attention heads / KV heads | 8 / 4 (grouped-query attention) |
| vocab | 512 (custom SentencePiece BPE, `tok512.bin`) |
| context length | 512 tokens |
| output head | tied to the embedding table |
| **total parameters** | **260,032** |
| weights file | 1,056,540 bytes, fp32 |

Where the parameters go:

| Part | Params | Share |
|---|---|---|
| Token embedding (512 × 64, also the output head) | 32,768 | 12.6 % |
| Attention (Wq, Wk, Wv, Wo) × 5 layers | 61,440 | 23.6 % |
| SwiGLU FFN (W1, W2, W3) × 5 layers | 165,120 | 63.5 % |
| RMSNorm weights | 704 | 0.3 % |

For scale: GPT-2 small (124M) is ~480× bigger and Llama-2-7B is ~27,000× bigger.

## 2. Runtime: `llama2.c`

[`karpathy/llama2.c`](https://github.com/karpathy/llama2.c) `run.c` (commit `350e04f`) is the whole Llama-2 forward pass, the BPE tokenizer and the sampler in one 973-line C file with no dependencies. **I reviewed the source before running it:** it only `open`/`mmap`s the model and tokenizer files and prints to stdout. No network, no exec, no file writes.

## 3. Discovery: a 64-bit kernel under a 32-bit Android

Android reports only `armeabi-v7a` (32-bit userland, as in `../REPORT.md`), and `uname -m` says `armv8l`. But the kernel is `5.15.153-android14` **64-bit**: a statically linked **aarch64** binary runs fine from `adb shell`. So I built both:

```sh
brew install zig        # zig 0.17.0, used as a cross-compiler
# musl doesn't pull in stdint.h implicitly like glibc does → -include stdint.h
zig cc -target arm-linux-musleabihf -mcpu=cortex_a55 -O3 -static -include stdint.h -o bin/llama2c/run-armv7 third_party/llama2.c/run.c -lm
zig cc -target aarch64-linux-musl                     -O3 -static -include stdint.h -o bin/llama2c/run-arm64 third_party/llama2.c/run.c -lm
```

Both are ~1.7 MB static ELF files and need nothing from Android's libc.

## 4. Deploy and run

```sh
D=/data/local/tmp/tinyllm
adb -s <tv-ip>:5555 shell mkdir -p $D
adb -s <tv-ip>:5555 push bin/llama2c/run-armv7 bin/llama2c/run-arm64 models/llama2c/stories260K.bin models/llama2c/tok512.bin $D/
adb -s <tv-ip>:5555 shell "cd $D && chmod 755 run-* && ./run-arm64 stories260K.bin -z tok512.bin -t 0 -n 0 -i 'Once upon a time'"
```

(`-t 0` = greedy/deterministic, `-n 0` = run to the full 512-token context or until the model emits BOS, `-z` = tokenizer.)
Local files: source in `third_party/llama2.c/`, binaries in `bin/llama2c/`, model + tokenizer in `models/llama2c/`.
SHA-256: `stories260K.bin` `b0a507e7…2696`, `tok512.bin` `037cb335…c312`, checked identical on the Mac and on the TV.

## 5. Output (verbatim, from the TV)

**Greedy, prompt "Once upon a time".** armv7 and arm64 give byte-identical text:

> Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
> Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
> Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
> Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
> Lily and her mom went to the park to play. They played together and had fun. After a while, Lily's mom came into the room and saw the park. She was so happy and said, "Thank you, Lily! You are a good friend."

**Sampled, temperature 0.8, seed 42, no prompt:**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, Lily's mom told her to teach him to cut one toys and bring it to the sunflower. Lily was scared and scared.
> Lily wanted to make a popcorn spoiled a big, red treasure. She was sad because she had started to draw and it was dead. […] The cake was happy to have a friend.

**Prompt "The cat", greedy, 60 tokens:**

> The cat and a boy were playing in the park. They liked to play with their toys and run around the park. They liked to play with their toys and seek.

**The bare minimum, "one token out"** (empty prompt, `-n 2`): `Once upon`

What 260K parameters get you: correct grammar, punctuation, quotes, and a named character kept across sentences. You also get loops ("didn't know what to do"), logic slips ("Lily's mom came into the room and saw the park") and, with sampling, nonsense phrases ("a popcorn spoiled a big, red treasure").

## 6. Numbers

| Metric | armv7 (32-bit) | arm64 (64-bit) |
|---|---|---|
| Speed, 3 runs, greedy full story | 615 / 616 / 655 tok/s | 673 / 644 / 678 tok/s |
| Best observed (short prompt) | — | ~900–1000 tok/s |
| Binary size | 1.67 MB | 1.81 MB |

The arm64 build is ~5–10 % faster (more NEON registers). Both use one core: no OpenMP, and the model is too small for threads to help.

**Memory.** Weights 1.06 MB (mmapped) + KV cache 5 × 512 × 32 × 2 × 4 B = 0.66 MB + a few KB of activations. The kernel reported `VmHWM` 868 kB mid-run (RssAnon 768 kB, RssFile 100 kB). Linux updates RSS counters lazily, so the true total is closer to ~1.8 MB. Either way, the box's ~800 MB of free RAM is untouched.

## 7. Footprint and undo

- On the TV: ~6 MB in `/data/local/tmp/tinyllm`, owned by the `shell` user. **No app, package, or setting changed.** The only thing that runs in the background is the optional web server (section 8) while it is started.
- On the Mac: `zig` (brew, ~212 MB) and this folder (`llm-experiments/`).
- Undo: `adb -s <tv-ip>:5555 shell "pkill -f llm-server; rm -rf /data/local/tmp/tinyllm"` (also in `../rollback.sh`). `brew uninstall zig` is in `../rollback.sh --mac`. Logged in `../CHANGELOG.md` 2026-11-03.

## 8. Using it from the TV itself: web page

![Tiny LLM page on the TV, mid-story](screenshots/tinyllm_04_done.png)

`tiny-llm-web/` turns the experiment into something you can use from the couch:

- **`server.c`** (~230 lines) `#include`s the same reviewed `run.c` and adds a single-threaded HTTP server. The model loads **once**, the process stays at ~1.9 MB RSS, and it serves:
  - `GET /`: the page
  - `GET /info`: model config as JSON
  - `GET /gen?prompt=…&t=…&delay=…`: a token stream. Each piece is followed by a marker byte: `0x1d` for a prompt token, `0x1f` for a generated one. A final `0x1e` is followed by stats JSON.
  - If the page starts a new story, the old request is aborted, `send` fails, and the old generation stops.
- **`index.html`**: a TV-layout page with a 5 % safe area (the Samsung crops the edges), big type, preset openings (no typing needed), Imagination = Steady / Creative / Wild (temperature 0 / 0.8 / 1.3) and Speed = Typewriter (60 ms per token, so you can watch it) / Full speed. Prompt text is blue and the newest token amber. The footer shows the live token count, then tok/s, seed and why it stopped. Arrow-key focus navigation is built in, but this browser uses an on-screen mouse cursor, so you just point and click with the remote.
- **`start.sh`**: builds `llm-server` (zig, static aarch64) if needed, pushes it, (re)starts it detached with `setsid`, and opens the page in the TV browser. `./start.sh stop` stops it.

**How to open it:** run `tiny-llm-web/start.sh` from the Mac, or on the TV open the browser (`com.internet.tvbrowser`) and go to **`http://<tv-ip>:8080/`**. Bookmark it with the ☆. Phones and the Mac on the same Wi-Fi can open the same URL.

**Gotchas found:**
- That browser treats `http://127.0.0.1:8080/` (and `localhost`) as a **Google search**, so use the LAN IP. If the router gives the box a new IP, the bookmark changes too.
- `adb shell "cmd &"` hangs, because adb waits on the background process's file descriptors. Fix: `(setsid cmd </dev/null >log 2>&1 &) </dev/null >/dev/null 2>&1`. The server keeps running after `adb disconnect` (verified).
- It does **not** survive a TV reboot, since nothing on Android can autostart a shell-user process without root. After a reboot, run `start.sh` again.
- It listens on port 8080 on the home network with no password. All it can do is write children's stories, but stop it (`./start.sh stop`) if that bothers you.

Measured through the server: "The little dog", greedy → 217 tokens, ending naturally, **731 tok/s** compute (forward passes / CPU time).

## 9. Going bigger: real chat models with llama.cpp

Same box, same trick (static aarch64 binary), but now with real instruction-tuned models.

### Build

- **Runtime:** [llama.cpp](https://github.com/ggml-org/llama.cpp) release **v0.8.0** (commit `8642541`, 28 Oct 2026). I couldn't review it line by line like `run.c`, so I built the pinned official release with your approval.
- **Why build it at all:** the official `android-arm64` release build **can't run on this box**. It needs Android's 64-bit loader, and this 32-bit Android has none (`/system/bin/linker64` and `/system/lib64` don't exist). A fully static musl build sidesteps that.
- **Compiler:** CMake + Ninja, with zig as the cross-compiler through a two-line wrapper (`zigcc/`). The wrapper drops ggml's `-march=armv8.2-a+dotprod+fp16`, which zig rejects, and passes `-mcpu=cortex_a55` instead, which enables the same dotprod + fp16 instructions.
- **CMake options:** `-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS=-static`.
- **Output:** `llama-cli`, `llama-server`, `llama-bench`, stripped with `llvm-strip` from 125–177 MB to 9–15 MB each. They're kept in `bin/llama-tv/` (with `SHA256SUMS`); the source and build tree were deleted afterwards to save Mac disk.

### Models (in `/data/local/tmp/llm`, checksums match Hugging Face)

| Model | Released | Params | File | SHA-256 |
|---|---|---|---|---|
| Gemma 3 270M-it, Q4_K_M ([unsloth](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF)) | Aug 2025 | 268 M | 253 MB | `b1baabd6…8e39` |
| Qwen3.5-0.8B, Q4_K_M ([unsloth](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)) | Mar 2026 | 752 M (text) | 533 MB | `bd258782…c517` |

### Speed (`llama-bench`, 4 threads)

| Model | Prompt processing | Generation | vs stories260K |
|---|---|---|---|
| stories260K (llama2.c, fp32) | — | ~650 tok/s | 1× params |
| Gemma 3 270M | 70 tok/s | **21.9 tok/s** | ~1,000× params |
| Qwen3.5-0.8B | 31 tok/s | **6.7 tok/s** | ~2,900× params |

Gemma generation with 1 / 2 / 4 threads: 10.8 / 18.6 / 21.9 tok/s. At 4 threads it's limited by memory bandwidth, not compute.

### Answers (verbatim, from the TV, greedy unless noted)

| Prompt | Gemma 3 270M | Qwen3.5-0.8B (thinking off) |
|---|---|---|
| "Why is the sky blue? 3 sentences" (t=0.7) | Rayleigh scattering, short, a bit circular | Fluent, mostly right; one slip ("Rayleigh scattering … at the Earth's surface") |
| "17 sheep, all but 9 run away, how many left?" | "There are 17 sheep left." ✗ | "18" ✗ |
| Haiku about a TV box that learned to think | "A tiny box, / A world of stories untold, / Learned to think, small and bright." | "A tiny box, / With eyes that glow, / And a mind that grows." |
| Capital of Australia, one word | "Sydney" ✗ | "Canberra" ✓ |

Takeaway: both models are real assistants that follow instructions and keep to the format. Neither can reason under 1B params, and trick questions fool both. Qwen knows more facts and Gemma is 3× faster.

### Memory: the real limit

- The box has ~650–770 MB free with its normal apps running.
- Qwen3.5-0.8B peaked at **~960 MB RSS**. That counts the 533 MB weights file as mmapped pages, plus the context and compute buffers.
- To make room, Android's low-memory killer **closed 7 cached background apps** (YouTube recommendations, TTS, Assistant search, Play instant-app installer, …). Android restarts these on demand, and nothing in the foreground was touched.
- A 1B+ model (Llama 3.2 1B, Gemma 3 1B ≈ 800 MB at Q4) would start pushing out things you'd notice. ~0.8B at Q4 is about the ceiling for this box.
- (AirPlay Server was found not running, but it died at 10:11, three hours before these runs, so it's unrelated.)

### Reproduce

```sh
adb -s <tv-ip>:5555 shell 'cd /data/local/tmp/llm && ./llama-cli -m Qwen3.5-0.8B-Q4_K_M.gguf -t 4 -c 2048 -n 300 --reasoning off -st -p "Your question" </dev/null'
```

`-st` = single turn then exit. Leave it out for an interactive chat in `adb shell`. `--reasoning on` lets Qwen think first, but at ~6 tok/s that takes minutes.

## 10. Bigger than RAM: streaming weights from flash (Qwen3.5-2B)

The idea: run a model that **doesn't fit in free RAM** by leaving the weights memory-mapped from flash. The kernel pages them in as each layer is used, evicts them when memory runs out, and reads them back on the next token.

**Model:** Qwen3.5-2B Q4_K_M ([unsloth](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)), 1,280,835,840 bytes, SHA-256 `aaf42c8b…9223` (matched on Mac and TV). That's 1.28 GB against ~0.85–0.9 GB free RAM.

**Run:** `-nr` (`--no-repack`) is the key flag. With repacking on, llama.cpp copies weights into anonymous RAM and the run would OOM; without it they stay file-backed and can be evicted and re-read.
```sh
sh llm-stream-monitor.sh q2b ./llama-cli -m Qwen3.5-2B-Q4_K_M.gguf -t 4 -c 512 -n 60 --temp 0 -st -nr --reasoning off \
  -p "What is the capital of Australia, and why was it chosen over Sydney? Two sentences."
```
`tools/llm-stream-monitor.sh` samples `/proc/vmstat` `pgpgin` (flash reads), the process's major page faults, RSS and MemAvailable every 2 s. `/proc/diskstats` is not readable as the shell user.

**Answer** (verbatim, capped at 60 tokens):
> The capital of Australia is Canberra, which was chosen because it was designed to be a neutral, non-capital city that avoided the political tensions of the two competing colonial capitals, Sydney and Melbourne.
> Canberra was selected as the capital in 1908 after a decade of planning, with the…

The answer is correct, and a clear step up from the 0.8B.

| Metric | Value |
|---|---|
| Speed | prompt 1.8 tok/s, **generation 0.1 tok/s (~8 s per token)** |
| Wall time | 529 s (8 min 49 s) for the prompt + 60 tokens |
| **Read from flash** | **83.8 GB** for one short answer |
| Flash throughput | ~200 MB/s sustained (peaks ~290 MB/s) |
| Major page faults | ~265,000 |
| Process RSS | flat at ~700–840 MB the whole run |
| MemAvailable | flat at ~850–870 MB, **no app kills** (unlike the 0.8B run, §9) |

**What happened:** a forward pass reads every weight once, in order. The file is bigger than what the kernel can keep cached, so in a sequential scan like that, eviction always throws out the part that will be needed soonest. In practice **the whole 1.28 GB model is re-read from flash for every token**: 60 tokens × 1.28 GB ≈ 77 GB, plus the prompt pass, ≈ 84 GB measured. Speed is set purely by the eMMC: 1.28 GB ÷ ~200 MB/s ≈ 6–8 s per token, which matches. The CPU mostly waits.

The upside: because the weights are file-backed, the page cache gives them up under pressure. The model **shrinks itself instead of killing other apps**, which makes streaming a gentler way to run something big than a model that fits.

**Stopped at 2B on purpose.** Qwen3.5-4B (2.74 GB, bigger than the box's total 1.9 GB of RAM) was downloaded and checksum-verified (`00fe7986…11a4`) but deleted unrun at your call. By the same arithmetic it would be ~14 s per token, with ~2.7 GB read per token.

**Flash wear:** this is all reads. eMMC wear comes from writes, so streaming doesn't wear the flash out.

## 11. Peak-performance push for Qwen3.5-2B

**Goal (your call):** make Qwen3.5-2B as fast as this box allows. You can edit frameworks or write our own. **Baseline:** 0.1 tok/s, streamed from flash (§10).

### 11.1 Where the bytes go

`tools/gguf_info.py` reads a GGUF header, locally or over an HTTP range request. Qwen3.5-2B Q4_K_M breaks down as:

| Part | Size | Read per token |
|---|---|---|
| `token_embd` 248,320 × 2048, Q6_K | **417 MB (33 %)** | **all of it**: embeddings are tied, so the output head multiplies the whole table |
| 24 layers (Q4_K 514 / Q5_K 208 / Q6_K 127 MB) | 853 MB | all of it |

The input side only needs one 1.7 KB row per token. Only the output head forces the full 417 MB read.

### 11.2 Experiments run so far

| # | Experiment | Result |
|---|---|---|
| E1 | **Pruned LM head**, 32,801 rows (ids < 32768 + 33 control tokens), via a llama.cpp patch + `tools/gguf_prune_head.py` | **0.1 → 4.76 tok/s** generation, prompt 14.9 tok/s (`llama-bench`, 4 threads). The model is RAM-resident: flash reads drop to ~0 after load. |
| E1b | Same, quality check | ✗ "Canberra" became "the Australian Capital City (ACQ)": `ĠCanberra` is token **66463**, outside the kept ids (likewise `ĠCairo` 50779, `ĠNile` 73583, `ĠXiaomi` 69025). Ranking tokens by id is too crude. |
| E2 | Weight repacking on vs off (`--repack 1/0`) | 4.76 vs **0.40** tok/s. Repacked weights live in anonymous RAM, which Android doesn't drop like file cache. Never use `-nr` here. |
| E3 | Threads 1/2/3/4 | 2.37 / 3.98 / **4.98** / 4.49 tok/s: memory-bandwidth bound at ~4.5 GB/s. So **speed ≈ 4.5 GB/s ÷ bytes per token**, and 3 threads is best. |
| E4 | Vocab coverage from real text (Python port of the Qwen3.5 tokenizer, matches llama.cpp token for token; 1.5 M-token corpus: Wikipedia, 4 Gutenberg books, code, docs) | Held-out coverage with ~33K rows: **97.9 %** when picked by frequency vs **91.4 %** by id. All 42,167 tokens seen in training text: 98.7 %. A hybrid set (frequent + low ids) is next. |
| E5 | Which quants get fast repacked kernels on an A55 (dotprod, no i8mm) | Q4_0, IQ4_NL, Q4_K, Q5_K, Q6_K, Q8_0: yes. IQ4_XS, Q3_K, Q2_K: no. Per-token bytes with a 33K head: Q4_K_M 908, Q4_K_S 845, IQ4_NL 840, Q4_0 842, IQ4_XS 800, Q3_K_M 734 MB. |
| E6 | **MTP head** (multi-token prediction, for speculative decoding) | unsloth's GGUFs drop it, but the original checkpoint has it (15 `mtp.*` tensors). `tools/fetch_mtp.py` range-reads only those tensors from the 4.5 GB safetensors file, applies the converter's norm +1, and quantizes to Q8_0: **64.7 MB**. Norm scales check out against the main layers. The llama.cpp patch also scatters pruned logits in the MTP graph. **Not run yet.** |
| E7 | Live web page (`tv-llm-ui/`) | Works: tokens stream with confidence colors, top-3 alternatives, live tok/s, model/config/box panels. **The TV browser (~300 MB) plus the 2B doesn't fit** (95 MB free, load 22, so I stopped it). The 0.8B runs next to the browser at 4.7 tok/s on screen. You chose to watch the 2B from a phone/Mac. |
| E8 | Server benchmark (`bench.py`, 4 fixed prompts, greedy, 128 tokens) | 3.4–3.6 tok/s on 3 prompts, then **0.38 tok/s** on the 4th: Android respawned background services (Projectivy jobs, Play Services, launcher services, Xiaomi quick settings) and pushed the model out (118 MB free, 424 MB/s flash). Short `llama-bench` runs never showed this. |
| E9 | **Disabled 7 background apps** (your request; list + undo in `../CHANGELOG.md` / `../rollback.sh`) and shrank buffers (ctx 1024, micro-batch 128) | Idle: model process 1,021 MB, 299 MB free, flash 0 MB/s. **During your generation from the page it thrashed again** (83 MB free, 1,171 MB/s flash, load 21), so I stopped the server at your request. |

### 11.3 What this means

At ~1 GB the Q4_K_M 2B is **too big for sustained use** on this box, even with apps disabled: Android's own floor plus whatever it respawns leaves ~1.0–1.15 GB at best. **The model has to drop to ~750–800 MB to stay resident.** Next steps, in order:

1. **Leaner weights:** IQ4_NL or Q4_K_S (−60 MB, still fast kernels), or Q3_K_M (−120 MB, slower kernels). Measure speed, memory over time, and answers.
2. **Frequency-chosen head** (~40K rows, ~67 MB) so "Canberra" survives; E1b gets re-checked.
3. **MTP speculative decoding** on top: ~1.5× expected if ~70 % of drafts are accepted, since generation is bandwidth-bound.
4. Every run is logged to `bench/results.jsonl` (prompt, text, speed) and summarized here.

### 11.4 Tools added in this push

| File | What it does |
|---|---|
| `tools/gguf_info.py` | GGUF header reader (local file or URL range request) |
| `tools/gguf_prune_head.py` | adds `output.weight` (kept rows, raw bytes) + `output_ids.weight` |
| `tools/vocab_select.py` | Qwen3.5 tokenizer port + coverage stats + kept-id lists |
| `tools/fetch_corpus.py` | Wikipedia + Gutenberg corpus |
| `tools/fetch_mtp.py` | range-fetch + convert the MTP tensors |
| `tools/tv-free-ram.sh` | force-stop heavy apps before a run |
| `tools/llm-stream-monitor.sh` | flash/RSS sampler |
| `tv-llm-ui/` | `start.sh`, `index.html`, `boxstats.sh`, `bench.py` |

The llama.cpp patch (`third_party/llama.cpp`, v0.8.0 + 4 files) adds the `LLM_TENSOR_OUTPUT_IDS` tensor and a `set_rows` scatter in the qwen35 main and MTP graphs. Patched binaries live in `/data/local/tmp/llm/patched/` on the TV.

### 11.5 Round 2: pushing to the limit (2026-11-03, evening)

Goal: exhaust every way to make the 2B faster on this box without breaking its answers. Every run is listed here; raw logs are in `bench/raw/`, benchmark runs in `bench/results.jsonl`.

| # | Experiment | Result |
|---|---|---|
| E10 | **What can the RAM deliver?** `tools/membw/membw.c`: NEON streaming read of a 256 MB buffer, 1–4 threads | **6.7 GB/s on 1 core, 8.3–8.4 GB/s on 2–4 cores.** llama.cpp only reached ~4.5 GB/s effective (E3), so generation is **not** memory-bound yet: there is up to ~1.85× headroom in compute. Raw: `bench/raw/E10-membw.txt` |
| E11 | **Where does a token's time go?** Added a per-op profiler to ggml (`GGML_OPPROF=1`: thread 0 times every graph node, including the barrier wait, and prints a table at exit). Q4_K_M + 32K head, 3 threads, tg32 | Q4_K matmuls 34 % at **6.2 GB/s** (close to the RAM limit) · **Q6_K matmuls 22 % at only 3.4 GB/s** (slow kernel; Q4_K_M keeps Q6_K for some layers *and* the head) · Q5_K 18 % at 4.7 GB/s · ~900 small ops per token (norms, copies, DeltaNet, conv) ~20 %, i.e. ~50–80 µs each, far more than their math: mostly thread sync / scheduling · embedding lookup (`GET_ROWS q6_K`) 5 %, page faults on the memory-mapped table. Raw: `bench/raw/E11-opprof-q4km-h32k-t3.txt` |
| E12 | **Thread options** (`llama-bench` tg48 ×3, Q4_K_M + 32K head): default, `--poll 0/100`, pinning (`-C 0xE`/`0x7 --cpu-strict 1`), `-t 4`, `--prio 2` | All within noise: 4.0–4.9 tok/s, ±0.2–0.8. Pinning is slightly *worse* (4.4–4.5): Android's own threads then compete with ours on fixed cores. No option removes the per-op overhead, so it's real work + sync, not polling. (Partly overlapped a Wi-Fi transfer; re-checked later on the final model.) Raw: `bench/raw/E12-thread-options.txt` |
| E13 | ⚠ **Incident: kernel panic + reboot.** First `llama-quantize` run on the TV | llama-quantize's default slab buffer is 8 GB, so converting the 1 GB BF16 embedding table (508 M values) wanted a ~2 GB float buffer on a 1.9 GB box: memory ran out and the **kernel panicked** (`sys.boot.reason=kernel_panic`, uptime reset). The 4 GB input survived (hash re-checked); the files written in the seconds before the crash were lost. Fix: `--max-buffer-size 64` (MiB slabs) in `tools/tv-quant.sh`, and every heavy job now runs under `tools/tv-guard.sh`, which samples MemAvailable every second and kills the job below 180 MB. With both, quantizing holds ~770 MB available. |
| E14 | **All-Q4_K mix, quantized on the TV from BF16** (`tools/tv-quant.sh q4k`: `--pure` Q4_K for every matrix, embedding table and head; unsloth imatrix) vs unsloth's Q4_K_M, both with the same id-ranked 32K head | **4.82 → 5.59 tok/s generation (+16 %), 14.5 → 17.4 tok/s prompt (+20 %)**, file 1.23 → 1.05 GiB. The Q6_K/Q5_K time is gone: all matmuls now run the fast Q4_K kernel (5.2–6.6 GB/s). Quantizing took 166 s on the TV (min 701 MB free). Raw: `bench/raw/E14-q4km-vs-q4k-pure.txt` |
| E15 | **Benchmark artefact:** the embedding lookup still costs ~10–13 ms per token in `llama-bench` | `llama-bench` feeds random token ids from the whole 248K vocab, so nearly every token reads a cold row of the memory-mapped embedding table from flash. Real text reuses frequent tokens that stay cached. Added `LLAMA_BENCH_COMMON_TOKENS=N` (draw from ids 256…256+N) to the patched llama-bench; `tools/tv-eval.sh` now uses N = 4000 and 5 repetitions (run-to-run noise on this box is ±5–10 %). |
| E16 | ⚠ **Perplexity at context 512 ran the box out of memory** (guard killed all 4 runs at 76–147 MB free; no crash) | `llama-perplexity` keeps logits for every position: 512 × 248,320 × 4 B = 508 MB on top of the model. Switched to context 128 × 40 chunks (127 MB of logits, 2,560 scored tokens). |
| E17 | **Which tokens should the head keep?** All-Q4_K model with three heads, quantized identically: B = 32,801 rows by token id (old), C = **H45** (45,654 rows: frequency-picked from English/code + ar/zh/ja/es/fr/de/ru Wikipedia, plus all ids < 16384), D = **H65** (65,668 rows, bigger budgets). *Misses* = held-out tokens the head cannot produce at all | Speed (common tokens): B **6.18**, C 6.13, D 6.03 tok/s; prompt 17.4 / 17.2 / 16.8. Doubling the head costs only ~2 %. Misses on multilingual text: **B 50.9 % → C 19.7 % → D 12.6 %**; on English **B 10.0 % → D 1.3 %**. ⚠ *Corrected in E29:* this row first said D missed 1.3 % of multilingual tokens; the guard had killed a different run for C (English) and D (multilingual) than I assumed, and the 1.3 % was D's English figure. PPL values are not comparable across heads (each skips different positions). Conclusion: the id-ranked head was badly hurting non-English text; a ~65K frequency head is nearly lossless on English and much better on other languages, for ~2 % speed. Raw: `bench/raw/E17-heads-B-C-D-raw.txt` |
| E18 | **Which 4-bit kernel is fastest on the A55?** Q4_0 and IQ4_NL "pure" variants (same 4.5 bits/weight, same 32K head) vs Q4_K | Generation / prompt tok/s: **Q4_0 6.47 / 19.7** · Q4_K 6.18 / 17.4 · IQ4_NL 5.74 / 10.6 (noisy). Q4_0 has the simplest decode, so it's ~5 % faster to generate and 13 % faster on prompts. **But quality:** English perplexity Q4_0 **9.157** vs Q4_K **8.890** (+3.0 %, same positions). Raw: `bench/raw/E18-kernels-q40-iq4nl.txt` |
| E19 | **Where does the RAM go?** `tools/memprobe.sh` (llama-server + one real chat request, `/proc/<pid>/status`), Q4_K + H65; plus a loader patch that drops the source pages of every tensor copied into the repacked buffer (`madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)`; `LLAMA_KEEP_MAPPED_COPIES=1` = stock) | Stock: RSS 1,092 MB = **832 MB anonymous** + 259 MB mapped file pages; MemFree 38 MB after load. Patched: RSS 985 MB = 851 MB anonymous + **134 MB** file; **MemFree 205 MB** after load. MemAvailable is about the same (~385 MB → ~340 MB after a request) because clean page cache already counted as "available". **The real cost is the ~850 MB of anonymous memory** (771 MB repacked layers + 76 MB head + buffers), plus 44–74 MB already pushed to zram swap. Only fewer weight bytes, or fewer background apps, can fix that. Raw: `bench/raw/memprobe-D-*.txt` |
| E20 | **Quality reference: how much does 4-bit cost?** English perplexity of the BF16 (full precision) model, same 32K head, streamed from flash (130 s per 128-token chunk, so stopped after 15 of 40 chunks); others compared at the same chunk | Running PPL at chunk 15: **BF16 8.785 · unsloth Q4_K_M 9.036 (+2.9 %) · all-Q4_K 9.262 (+5.4 %) · Q4_0 9.664 (+10.0 %)**. Final (40 chunks): Q4_K_M **8.687**, all-Q4_K 8.890, Q4_0 9.157 (Q4_K_M streamed with `-nr` so it fits). So all-Q4_K gives up ~2.3 % perplexity versus Q4_K_M for +16 % speed and −180 MB; Q4_0's extra +5 % speed costs twice that again. **Q4_K stays the base format**; a "Q4_K + a few upgraded sensitive tensors" mix is queued as a middle ground. |
| E21 | **3-bit: Q3_K_M (stock) and Q4_K with 3-bit FFN up/gate** | Generation / prompt: Q3_K_M **4.12 / 7.2**, Q4_K+Q3_K-FFN **4.66 / 8.8** (vs all-Q4_K 6.18 / 17.4). Q3_K has no repacked kernel on this CPU and runs at only **1.7–2.0 GB/s** of weights. Side effect: tensors that aren't repacked stay memory-mapped file pages, so min free RAM during the runs was 680–800 MB instead of ~350 MB. **3-bit needs a repacked kernel**; that is now on the custom-kernel list (it would cut ~25 % of the weight bytes). Raw: `bench/raw/E21-q3k.txt` |
| E22 | **2-bit sweep** (§11.6): layers at IQ2_M / IQ2_XXS / Q2_K / Q2_0 with the imatrix; embedding table, head and MTP block stay Q4_K; same 32K head. IQ2_XXS was quantized by the TV and IQ1_M/IQ1_S by the Mac (`tools/mac-lowbit.sh`, 104 s on the Mac vs several minutes on the TV) | Gen / prompt tok/s, file, final PPL (BF16 ≈ 8.8, Q4_K 8.89): **IQ2_M** 3.12 / 4.96, 823 MiB, 12.32 · **IQ2_XXS** 4.16 / 5.57, 751 MiB, 20.23 · **Q2_K** 4.46 / 7.23, 857 MiB, 12.20 · **Q2_0** 4.56 / 7.32, 712 MiB, **377,511 (collapsed)**. All are slower than Q4_K (no repacked kernels) and none is much smaller (the Q4_K embedding/head/MTP parts are ~350 MB). Answers: IQ2_M and Q2_K are fluent but say Australia's capital is "**Canterbury**" and loop; IQ2_XXS says "Sydney" and degenerates into "I'm sorry. I'm sorry…" / `self = self`. **Post-training 2-bit breaks this 2B.** Raw: `bench/raw/E22-lowbit-2bit.txt` |
| E23 | **1-bit sweep**: IQ1_M (1.75 bits), IQ1_S (1.56), Q1_0 (1.125, binary ±1; the Bonsai format) on the 24 layers | **IQ1_M**: 3.32 / 5.30 tok/s, PPL **35.4**, answers are "**Answer** **Answer** **Answer**…". **Q1_0**: the fastest generation in this whole push, **7.40 tok/s, prompt 24.0** (it has a repacked dotprod kernel, and the layers are only ~190 MB), but PPL **75,679** and both answers empty: binary weights applied after training destroy a normal model, as expected. **IQ1_S** (re-run after the fix below): 4.16 / 6.87 tok/s, PPL **70.5**, answers "Answered by the answer is the answer… 1. 1. 1.". The first copy failed to load. The Mac→TV hand-off had fetched the file while llama-quantize was still writing it (591 MB of 731 MB). Fixed in `tools/mac-lowbit.sh` (write to a temp name, publish when finished, check the size on the TV before renaming); redone and re-queued. **Conclusion: post-training 1-bit is unusable; the only route to 1-bit is a model trained for it** (§11.7). Raw: `bench/raw/E23-lowbit-1bit.txt` |
| E24 | **Middle-ground mixes**: all-Q4_K with the most sensitive tensors upgraded (`tools/tv-mixes.sh`, same 32K head). M1 = Q4_K + `attn_v` at Q6_K + `ffn_down` of the first and last 4 layers (0–3, 20–23) at Q5_K; M2 = M1 + `ssm_out` (the DeltaNet output projections) at Q5_K | Gen / prompt tok/s, file, final English PPL: all-Q4_K 6.18 / 17.4, 1,091 MB, 8.890 · **M1** 5.82 ± 0.84 / 17.1, 1,104 MB, 8.832 · **M2** **6.09** ± 0.20 / 15.2 ± 2.0, 1,113 MB, **8.805** · unsloth Q4_K_M 4.82 / 14.5, 1,260 MB, 8.687. M2 recovers ~42 % of the gap to Q4_K_M for +22 MB and no measurable generation-speed loss (the upgraded tensors are only ~4 % of the bytes). The prompt numbers are noisy (±2). **M2 is the new quality/speed pick**; queued again with the H65 multilingual head (`tools/tv-heads2.sh`). Raw: `bench/raw/E24-mixes.txt` |
| E25 | **Models trained for 1-bit / ternary** (§11.7, `tools/tv-others.sh`): speed, peak RSS, English perplexity → **bits per byte** (lower is better; each model's own token count of the 22,031-byte `eval-en.txt`), two greedy answers | Gen / prompt tok/s · max RSS · PPL · bits/byte: **Bonsai-1.7B (Q1_0)** 6.81 / 28.8 · 514 MB · 21.59 · 1.158 — **Bonsai-4B (Q1_0)** 3.17 / 11.4 · 1,090 MB · 17.21 · 1.073 — **Ternary-Bonsai-1.7B (Q2_0)** 4.33 / 7.6 · 556 MB · 17.13 · 1.071 — **BitNet b1.58 2B4T (TQ2_0)** 4.60 / 8.2 · 794 MB · 66.42 · 1.546. Reference, measured with the full 248K vocabulary (nothing skipped, E29): **Qwen3.5-2B all-Q4_K = PPL 10.478 = 0.909 bits/byte**, clearly better than all four (the earlier 0.845 estimate skipped the 10 % of tokens the 32K head can't produce). Answers: all three Bonsai models answer **Canberra** correctly and write a correct, documented `is_prime` (their reasons for choosing Canberra are vague); BitNet loops ("chosen over Sydney due to its being chosen over Sydney…") and writes broken code. **Verdict:** trained 1-bit models are usable, unlike post-training 1-bit (E23), but none beats Qwen3.5-2B at Q4_K: Bonsai-1.7B is ~10 % faster with half the RAM but clearly weaker; Bonsai-4B is about as good as the 1.7B ternary but half the speed. Raw: `bench/raw/E25-trained-lowbit.txt` |
| E26 | **Why don't fewer bits mean more speed?** Op profiles of every format measured so far, turned into weights per second | Matmul throughput in **billions of weights/s** (3 threads): Q4_K **11.0** (6.2 GB/s) · Q1_0 **12.1–13.4** (1.7–1.9 GB/s) · TQ2_0 **12.3** (3.2 GB/s) · Q2_0 8.3 · Q5_K 6.8 · Q6_K 4.1. Every repacked/simple format lands at **~11–13 G weights/s ≈ 1.5 weights per cycle per core**, whatever the bit width: the A55 matmuls are limited by **decode + dot-product instructions**, not by memory. That's why Bonsai-1.7B (1/4 the bytes of our 2B) is only 10 % faster, and why the 1-bit Qwen (E23) only reached 7.4 tok/s. Consequences: (1) fewer bits only help if the kernel's per-weight work drops too; (2) the Q4_K kernel itself is the lever: at the RAM limit (8.4 GB/s, E10) it would do ~15 G weights/s, **+35 %** headroom. The kernel micro-benchmark (`tools/kbench`) is next in the TV queue. Source: the `[opprof]` tables in `bench/raw/E11`, `E14`, `E25` |
| E27 | **Can a better Q4_K kernel get closer to the RAM limit?** `tools/kbench` on the TV: llama.cpp's 8x4 repacked gemv ("ref") vs an A55-scheduled variant ("v2": all 16 loads issued first, 8 independent dot-product accumulators, bias folded per super-block), each with software prefetch at 0–8 KB ahead; a 54 KB matrix (stays in cache) and a 27 MB one (from DRAM) | Cache-resident: ref **4.05 GB/s on 1 core, 11.4 on 3**; v2 3.9 / 10.6. From DRAM: ref **3.37 / 6.69 GB/s**, v2 3.24 / 6.83. Software prefetch never helps (−2 % to −25 %): the A55's hardware prefetcher already streams these sequential reads. So the stock kernel runs at **81 % of the measured 8.3 GB/s RAM ceiling** on 3 cores, and its pure compute rate (11.4 GB/s) is only 1.4× the RAM rate, so a perfect overlap would gain at most ~20 % and my rescheduling gained nothing. **The Q4_K kernel route is closed.** Same queue: final English PPL **Q3_K_M 9.64**, **Q4_K + Q3_K FFN 9.25** (vs Q4_K 8.89), slower and worse, which also closes 3-bit (a Q3_K repack kernel would cut bytes 25 % but adds decode work on a core that's already near its compute limit). Raw: `bench/raw/E27-kbench-and-q3-ppl.txt` (includes the op profile of the current best model: matmuls 74 %, GATED_DELTA_NET 6.2 %, RMS_NORM 5.8 %, CPY 3.9 %, GET_ROWS f32 2.9 %) |
| E28 | **Fused one-pass DeltaNet kernel** (patch, `patched3/`): the CPU `GATED_DELTA_NET` kernel made 5 passes over each head's 64 KB state per token (copy, decay, k-dot, update, q-dot); it now makes one (row j only needs its own delta, so k-dot, update and q-dot happen while the row is in L1). The new state is also written straight into the recurrent cache (`src[6]` = cache view) instead of a separate 1 MB/layer copy. Switches back to stock: `GGML_GDN_STOCK=1`, `LLAMA_GDN_COPY=1` | Op profile (Q4_K + H65, tg32): `GATED_DELTA_NET` **614 → 399 µs/call (−35 %)**, the 2,376 state `CPY`s per 33 tokens (224 ms) are gone, total profiled time **5.27 → 4.91 s (−6.9 %)**. llama-bench alternated stock/new: tg64 6.31 / **6.71** and 6.15 / 6.09 (±0.1–0.8, this box's noise), pp128 18.06 → **18.65**. Accuracy: the direct state write alone is bit-identical (PPL 8.7254 = stock); the fused kernel gives PPL 8.7356 (+0.12 %, ±9 % stderr) and different greedy text. `tools/gdn-test/gdn_test.c` (2,000 recurrent steps, same NEON code vs a double-precision reference) shows the fused update is **3× more accurate** than stock (4.3e-8 vs 1.35e-7 max error: FMA, fewer roundings), so the difference is rounding propagated through a 4-bit network, not a bug. Note: the per-op profiler now also reports compute-only time; the earlier "~100 µs per RMS_NORM" (E27) was barrier/scheduling noise in that run (15–16 µs per call here). Raw: `bench/raw/E28-gdn-ab-and-vkprobe.txt` |
| E29 | **Head follow-ups** (`tools/tv-heads2.sh`): on-TV head re-cutter check; full-vocab baseline; H65 perplexity on both texts; M2 mix with the H65 head | `bin/gguf_recut` (C, runs on the TV) gives a **byte-identical** H45 file to `gguf_build.py` (SHA-256 a50eadc5…), so heads can now be re-cut on the box in seconds. H65: English PPL 9.957 with 1.31 % misses, multilingual 9.100 with **12.58 %** misses (corrects E17). **M2 + H65** (`w-m2-h65.gguf`, 1,205 MB): 5.82 / 16.1 tok/s, English PPL **9.859** (−1.0 % vs Q4_K + H65 on the same positions), multilingual **9.068** (−0.35 %). The full-vocab run (248K tied head, repacked) **ran out of memory** (69 MB left, guard kill); re-run streamed (`-nr`, `tools/tv-fullvocab.sh`, 16 min): **PPL 10.478, 0 misses = 0.909 bits/byte**, the fair number for comparing with other models (E25). Raw: `bench/raw/E29-heads-baseline.txt` |
| E30 | **The GPU** (Mali-G310, your request: "fastest speed on GPU, up to 2B"). The box's userland is 32-bit and its only GPU driver is `/vendor/lib/egl/libGLES_mali.so` (32-bit, also the Vulkan ICD), so GPU code must be a 32-bit Android build: Android NDK r28c, `armeabi-v7a`, llama.cpp's Vulkan backend (`bin/vulkan32/`), shaders compiled with Homebrew `glslc` 2026.4 (the NDK's is too old for the int8 dot-product path). `tools/vkprobe` first: Vulkan works from the adb shell (no root; `/dev/mali0` is world-writable), **Mali-G310, Vulkan 1.3.247, fp16, int8 + integer dot product, subgroup 16, 1,910 MB unified heap, no cooperative matrix; GPU read bandwidth 7.5–8.0 GB/s** (same DRAM as the CPU, 8.3), 93 µs per submit round trip | ⚠ **Kernel panic #2**: the first run offloaded the whole 2B (`-ngl 99`) under the old 1-second guard; the box rebooted (CHANGELOG 03:10), files pushed seconds before were lost. Since then every GPU run goes through `tools/tv-gpu.sh` (5 samples/s, echoed live to the Mac, kills multi-process jobs by name). **Where GPU-path memory goes** (0.8B, 497 MB file): llama.cpp's default 512-token compute buffers 299 MB GPU + 272 MB pinned host (fix `-ub 64`); CPU-side weights put in a pinned Vulkan host buffer, 361 MB (fix `--no-host 1`); the 32-bit CPU backend repacking its weights, +430 MB (fix `--repack 0`); the tied 248K embedding copied again as the output layer, 208 MB; Mali shader compiles, a ~100–250 MB peak in warm-up. **Tuning, 0.8B tg16** (CPU, 64-bit: **8.30 tok/s**): GPU default 2.05 · forced int-dot mat-vec 2.01 · 4-subgroup workgroups 1.63–1.66 · K-quant rows/workgroup 1/4/8: 2.08/2.01/1.98 · **int-dot + 4 rows: 2.25** · plus DeltaNet with 4/8/16 lanes per column (new `GGML_VK_GDN_LANES`; the Mali default packs 64 floats of state per lane): 2.36/2.52/**2.53** (`GATED_DELTA_NET` 3.76 → 0.80 ms/call). Best GPU = **2.53 tok/s = 30 % of the CPU**. **Ceiling test**: a minimal hand-written 4-bit × int8 mat-vec shader (`tools/vkgemv`, coalesced 16-byte loads, `dotPacked4x8AccSatEXT`, subgroup reduce) peaks at **4.3 GB/s (7.6 G weights/s)** on large matrices and 1.3–2.5 GB/s on the 2B's layer shapes, vs **6.7 GB/s** for the CPU kernel (E27): this GPU can't beat the CPU at generation even with a perfect kernel. **Prompts** (Gemma 3 270M, the only model whose batched GPU path fits): GPU pp64 38.5 / tg32 8.3 vs CPU 72.9 / 22.0, also slower. **2B**: doesn't fit at any split (`tools/tv-gpu-2b.sh`, 0/8/16/20 GPU layers: all stopped by the guard, lowest 121 MB free); the GPU path needs the same ~1.1 GB of weights plus ~300 MB of driver/compile/staging overhead, on a box with ~1.0 GB available. Projected from the 0.8B per-op profile it would run at ~1.4 tok/s vs 6.0–6.7 on the CPU. A CPU+GPU hybrid over llama.cpp RPC (64-bit CPU process + 32-bit GPU server, `tools/tv-rpc-hybrid.sh`) fails in the RPC handshake on this box, and couldn't help anyway: layers run in sequence, so moving them to a 3× slower device only adds time. **Verdict: the GPU route is closed**: the Mali-G310 is ~3× slower than the CPU per weight and the RAM can't hold the 2B on its path. Raw: `bench/raw/E30-*` (sweeps, per-op GPU profiles, vkgemv, memory traces) |
| E31 | **MTP speculative decoding** (the 2B's built-in next-token head, blk.24, already in our files): `llama-server --spec-type draft-mtp --spec-draft-n-max N`, `tools/mtp-bench.sh` (4 fixed chat prompts, greedy, 128 tokens, real HTTP requests), `patched3` (fused DeltaNet), Q4_K + H65 | **No MTP: 6.58 tok/s** generation, 17.3 prompt (best real-chat number so far; E14 was 5.6–6.2 in llama-bench). **n_max=1: 4.50** (draft acceptance 0.76) · **n_max=2: 1.59** (0.59) · **n_max=3: 2.88** (0.48). The draft is often right (the prime-number answer: 97 % accepted), yet every setting is slower. Why: on this core each extra token in the verification batch costs nearly a full pass (E26: matmuls are compute-bound, so batching 2 tokens saves little), plus the draft pass itself, plus K snapshots of every DeltaNet state. Some runs also hit memory pressure (one 0.77 tok/s answer). Raw: `bench/raw/E31-mtp.txt` |
| E32 | **Cheaper 2–3 token batches** (patch, `patched5/`): stock llama.cpp runs the 1-row gemv once per row for batches of 2–3 tokens; they are now padded to one 4-row gemm tile (each weight block decoded once). `GGML_REPACK_PAD4=0` = stock | llama-bench, 2B Q4_K + H65, 3 threads, tok/s: pp1 5.98 → 6.10 · pp2 **7.25 → 7.33** (no gain: a 4-row tile costs about two 1-row passes here, and stock pp2 already costs only 1.65× pp1) · pp3 **6.99 → 10.48 (+50 %)** · pp4 13.6 → 13.8. Helps any 3-token batch (MTP n_max=2 verification). MTP re-measured with it: see below. Raw: `bench/raw/E32-pad4.txt` |

**Plan from E10/E11:** (1) an all-4-bit mix with no Q6_K/Q5_K, quantized on the TV from the original BF16 weights, which is both smaller and runs on the fast kernel; (2) cut the per-op sync overhead; (3) a frequency-chosen, multilingual head; (4) MTP speculative decoding; (5) look at the embedding page faults.

**New tooling for round 2:**
- `tools/gguf_build.py`: streaming GGUF rewriter. Adds a pruned head from an id list (`--head-ids`), extra tensors (`--add`, e.g. the MTP layer) and metadata (`--set`) in one sequential pass, so `curl HF | gguf_build.py | adb exec-in` writes a modified 3.9 GB model to the TV without storing it on the Mac (~4 GB free). Verified: rebuilding the old 32K-head file gives an identical size, tensor table and head bytes.
- `tools/fetch_mtp.py --bf16`: keeps the MTP matrices as raw BF16 so llama-quantize quantizes them along with the model.
- The llama.cpp patch gains `GGML_OPPROF` (ggml-cpu.c) and pruned-head-aware perplexity (`llama-perplexity`: positions whose correct token is outside the kept vocab are counted as *misses* and skipped, so PPL is the pruned model's own renormalized distribution).
- BF16 checkpoint streamed straight to the TV (`/data/local/tmp/llm/Qwen3.5-2B-BF16.gguf`, 3,775,709,216 bytes, SHA-256 `dc11be2c…19f1`, matches Hugging Face). Note: a two-way pipe (TV → Mac → TV over adb) only does ~1 MB/s on Wi-Fi; one-way HF → Mac → TV does ~8 MB/s.

### 11.6 Queued: the 1 / 1.5 / 2-bit sweep (your request)

**Research first:**
- **Q1_0** (1.125 bits/weight, binary weights ±1 with one scale per 128) is PrismML's Bonsai format, made for models *trained* to be 1-bit ([PrismML formats](https://docs.prismml.com/download/formats), [Bonsai-8B GGUF](https://huggingface.co/prism-ml/Bonsai-8B-gguf)). It's merged into llama.cpp, and our v0.8.0 even has a repacked dotprod kernel for it. Applied after the fact to a normal model, expect gibberish; it's measured anyway for completeness.
- **IQ1_S / IQ1_M** (1.56 / 1.75 bits) and **IQ2_XXS…IQ2_M** (2.06–2.7 bits) are llama.cpp's post-training codebook quants and need an importance matrix (we have unsloth's). There's no published perplexity for Qwen3.5-2B at these sizes. A Qwen3.5-2B study found no layer tolerates 1-bit without training for it ([study](https://huggingface.co/daksh-neo/qwen3.5-1bit-quantization-study)); 2-bit is reported workable with tuning ([write-up](https://dev.to/genrius_ai/how-we-fit-a-27b-model-on-a-12-gb-laptop-gpu-14eo)); IQ2_XXS loses noticeably even at 7B ([guide](https://tonisagrista.com/blog/2026/quantization/)).
- **TQ1_0 / TQ2_0** (ternary) only fit BitNet-trained models. Skipped.
- None of the IQ1/IQ2/Q2_K types has a repacked kernel on this CPU (dotprod, no i8mm), so they run the slower plain NEON dot products. The trade is far less RAM (the layers drop from 771 MB at 4.5 bits to ~350 MB at 2 bits, ~190 MB at 1.1 bits) against speed and quality.

**Plan** (`tools/tv-lowbit.sh`, queued to start when the Q4_K_M / Q3_K jobs finish). Variants in order: IQ2_M, IQ2_XXS, Q2_K, Q2_0, IQ1_M, IQ1_S, Q1_0. Each one:
1. Quantize the 24 layers from BF16 with the imatrix. The embedding table, head and MTP block stay Q4_K (no imatrix data for them).
2. Speed (`tv-eval.sh speed`: tg64 common tokens, pp128, op profile).
3. English perplexity, same 32K head as B/E, so it compares directly with BF16 8.785 / Q4_K 9.262 at chunk 15.
4. Two greedy answers (Canberra question, prime function) to check whether the output is still coherent.

Results go to `eval/lowbit-summary.txt` on the TV, then here.

### 11.7 Trained 1-bit and ternary models (your call: not limited to Qwen)

Post-training 1–1.75 bits can't be expected to work on Qwen3.5-2B (§11.6), but models *trained* at those widths exist:

| Model | Bits | File | Notes |
|---|---|---|---|
| [PrismML Bonsai-1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) | 1 (Q1_0, ±1 weights) | 248 MB | Qwen3 architecture, 28 layers; Q1_0 has a repacked dotprod kernel in our llama.cpp |
| [PrismML Bonsai-4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) | 1 (Q1_0) | 572 MB | Qwen3, 36 layers: 4B parameters in less RAM than our 2B's head + MTP |
| [PrismML Ternary-Bonsai-1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) | 1.58 (Q2_0, group 64) | 490 MB | the `_g64` file matches mainline's Q2_0 layout |
| [Microsoft BitNet b1.58 2B4T](https://arxiv.org/pdf/2504.12285v1) ([TQ2_0 GGUF](https://huggingface.co/Synapticode/bitnet-b1.58-2B-4T-tq2_0-gguf)) | 1.58 (TQ2_0) | 1.2 GB | 657 MB of it is an F16 tied embedding/head; requantized to Q4_K on the Mac (ternary layers copied byte-for-byte) |

Bonsai-8B (1.16 GB of 1-bit weights, all of which would be repacked into RAM) and Ternary-Bonsai-4B (1.07 GB) are over this box's RAM budget. Evaluation (`tools/tv-others.sh`): speed, memory, English perplexity converted to **bits per byte** (each model's own token count of `eval-en.txt`; Qwen3.5 uses 5,905 tokens for its 22,031 bytes), and the same two prompts.

Work is split between the machines (your call): the Mac quantizes the slow IQ2_XXS / IQ1_M / IQ1_S variants from the identical BF16 file (same SHA-256) with a native llama-quantize build (`tools/mac-lowbit.sh`) and hands them to the TV over HTTP, while the TV quantizes the fast types and runs every evaluation.

### 11.8 Final state of the 2B push (2026-11-04)

**Final configuration** (`tv-llm-ui/start.sh` defaults): `patched5/llama-server`, model `w-q4k-h65.gguf` (all-Q4_K from the BF16 weights with the unsloth imatrix, 65,668-row frequency-chosen multilingual head, MTP block kept but unused), 3 threads, context 1024, `-ub 128 -b 128`, MTP off.

| Step | Generation tok/s | Prompt tok/s | English PPL |
|---|---|---|---|
| Start of the push: unsloth Q4_K_M, id-ranked 32K head, stock kernels | 4.8 (llama-bench) | 14.5 | 8.69* |
| All-Q4_K from BF16 (E14) | 5.6–6.2 | 17.4 | 8.89* |
| + 65K multilingual head (E17, E29) | 6.0 | 16.8 | 9.96** |
| + fused DeltaNet kernel, direct state write (E28) | 6.3–6.7 | 18.7 | 9.96** |
| **Final, real chat requests via llama-server** (E31–E33) | **6.55** (llama-bench **6.69**) | **17.3** | 9.96** |
| Quality option `w-m2-h65.gguf` (E24, E29) | 6.20 | 16.4 | 9.86** |

\* 32K-head positions, \*\* 65K-head positions (not comparable across heads; each skips the tokens its head can't produce).

**Routes tried and closed** (each measured, see the E-rows):

| Route | Result | Why it doesn't help on this box |
|---|---|---|
| Fewer bits after training: 3-bit (E21, E27), 2-bit (E22), 1-1.75 bit (E23) | slower, and quality breaks below 4 bits | no repacked kernels for those types; matmuls are compute-bound (E26) |
| Models trained for 1-bit / ternary (E25) | usable, but none beats Qwen3.5-2B Q4_K | same compute-bound kernels; Bonsai-1.7B is 10 % faster but clearly weaker |
| Q4_0 / IQ4_NL (E18) | +5 % speed for +3 % perplexity / slower | |
| Better Q4_K kernel: rescheduling, software prefetch (E27) | no gain | stock kernel already at 81 % of the RAM bandwidth |
| Thread count, pinning, polling, priority (E12, E33) | 3 threads best | Android's own threads need the 4th core |
| GPU (Mali-G310, Vulkan, E30) | 30 % of CPU speed at best; the 2B doesn't fit | GPU reads 4-bit weights at ≤4.3 GB/s (CPU 6.7); its path needs ~300 MB more RAM; two kernel panics on the way |
| MTP speculative decoding (E31, E32) | slower (3.9–4.5 vs 6.55) even with 48-76 % draft acceptance and a 50 %-cheaper 3-token batch | verifying extra tokens costs almost a full pass on a compute-bound core |
| Middle-ground mix M2 (E24) | −1 % perplexity for −5 % speed | kept as an option, not the default |

**What's left, and why it's not worth it:** the remaining time per token is ~75 % Q4_K matmuls already near the RAM ceiling, ~5 % DeltaNet (now fused), and ~20 % small ops and synchronization (E11, E28). The only bigger levers would be a smaller model or a model trained for this hardware, both outside "the 2B".

## 12. Possible next rungs

- Go **smaller than 260K**: train a toy model with llama2.c's `train.py` (for example dim 16, 2 layers, ~20K params). At that size it stops being a "published LLM" and becomes a model built for the experiment.
- Put **Gemma 3 270M behind the web page** as a chat (via `llama-server`; it would hold ~300 MB while running). Qwen is too heavy to keep resident.
- Make it start on boot: a small Android app could launch the same server from its native library directory (that needs the Android SDK on the Mac).

## 13. The 4B push (2026-11-04)

Your request: "now lets try 4b models, no restriction on model just figure it out and push the limits". Same box, same rules: every run on the TV goes through `tv-guard.sh` (fast 0.2 s sampling, kill below 150 MB available), quality is English perplexity on `eval-en.txt` (ctx 128 × 40 chunks) turned into **bits per byte** (`log2(PPL) × tokens / 22,031 bytes`; Qwen3.5 uses 5,905 tokens, Qwen3/Bonsai 5,756) so different tokenizers compare fairly. Reference: **Qwen3.5-2B all-Q4_K = 0.909 bits/byte, 6.55 tok/s in chat** (§11.8).

### 13.1 The constraint

A 4B model has ~3.5-3.6 B layer weights. The 2B's final config already uses ~850 MB of anonymous RAM (E19), and the 4B runs below show the box tops out at **~0.9-1.0 GB of weights in RAM** (the rest is Android). That is **≤ ~2 bits per weight** for a 4B. Streaming from flash is no way out: flash reads at **192 MB/s** (`dd` of a 939 MB file), so a 2.7 GB Q4 4B would need ~8 s per token.

### 13.2 Experiments

| # | What | Result |
|---|---|---|
| E34 | **New repacked Q2_K kernel for ARM dotprod** (`ggml-cpu/repack-q2k-tv.inc`). Upstream repacks Q2_K only for x86 AVX-512 and RISC-V; on ARM it runs the per-row `vec_dot`, which does a horizontal add (`vaddvq`) and a scalar scale multiply every 16 weights. New layout: 8 rows interleaved 4 bytes at a time, `vdotq_laneq` against the activation, 2-bit fields masked in place (×4/×16 folded into a shift of the int32 sums), scales applied in int16. Gemv + 4-row gemm (prompt batches). `GGML_Q2K_TV=0` = stock | `tools/kbench/kbench_q2k.c` on the TV, 3 threads: **7.8 → 13.9 G weights/s** from DRAM (+78 %), 8.6 → 19.4 cache-resident (2.2×); software prefetch only hurts. Exact: `tools/kbench/q2k_check.cpp` (includes the kernel file itself) matches a double reference to 1.3e-6 for both gemv and gemm; gemm is 1.7× four gemv calls. In llama.cpp (0.8B requantized to pure Q2_K as a test model): **generation 8.47 → 11.97 tok/s, prompt 13.6 → 32.4**; perplexity 247.3 vs 246.4 on this deliberately broken test model (activation-rounding differences, amplified by a flat distribution). Raw: `bench/raw/E34-*` |
| E35 | **Qwen3.5-4B quality at the sizes that matter** (Mac, `tools/mac-4b-frontier.sh`, unsloth Q8_0 + unsloth's 4B imatrix, Q4_K embedding) | **Q8_0: PPL 8.252 = 0.816 bits/byte** (the 4B is ~10 % better than the 2B when unconstrained) · pure **Q2_K** (1,462 MiB file, ~1.12 GiB of layers): **12.351 = 0.972** · pure **IQ2_XXS** (1,222 MiB, ~0.88 GiB of layers, the largest that fits): **19.049 = 1.139**. At the bit width the RAM allows, the 4B is far worse than the 2B at Q4_K (0.909). Raw: `bench/raw/E35-4b-frontier.txt` |
| E36 | **Does the 4B Q2_K fit anyway?** (TV, `patched6` = E34 kernel) | Full 248K tied head: guard kill at RSS 1,166 MB. With the 65K head (`q4b-q2k-h65.gguf`): kill at **RSS 1,045 MB**, after `tv-free-ram.sh`. No. Raw: `bench/raw/tv-logs-4b/E36-*` |
| E37 | **Ternary-Bonsai-4B** (PrismML, Qwen3-4B architecture trained with ternary weights, `Q2_0_g64` file, 1.14 GB, sha256 9d968b04…) | Quality (Mac, full vocab): **PPL 12.902 = 0.964 bits/byte**: the best 4B that can fit, better than Bonsai-4B 1-bit (1.073, E25), still behind the 2B (0.909). Requantized to llama.cpp's ternary types TQ1_0 (1.69 bits, 810 MiB) or TQ2_0 (2.06 bits, 990 MiB): **PPL 13.934 = 0.993** for both (same values: one scale per 256 weights instead of per 64 costs +8 %). Head: id-rank 64K head misses 4.2 % / 37 % of English / multilingual tokens; a frequency-chosen head from the same corpora as H65 (`tools/head_select_tok.py`, llama.cpp's own tokenizer, 61,592 rows) misses **0.95 % / 6.7 %** (the 2B's H65: 1.3 % / 12.6 %). Pruned heads needed a port to the Qwen3 architecture (`src/models/qwen3.cpp`). TV, stock kernels: TQ1_0 + head **2.20 tok/s, prompt 2.70**, RSS 873 MB; TQ2_0 (1.04 GB mapped) thrashes the page cache (prompt 3.19, generation never finished, killed after 20 min). Raw: `bench/raw/E37-*`, `E35-4b-frontier.txt` |
| E38 | **New repacked TQ1_0 / TQ2_0 kernels** (`ggml-cpu/repack-tq-tv.inc`, same 8-row × 4-byte interleave; TQ1_0 keeps ggml's base-3 decode `(x·3^k mod 256)·3 >> 8`; `GGML_TQ_TV=0` = stock) | `tools/kbench/kbench_tq.c`, DRAM, 3 threads: **TQ1_0 9.5 → 14.4 G weights/s** (4T 17.0), **TQ2_0 11.9 → 19.1** (4T 24.0); exact vs a scalar reference. On the TV: Ternary-Bonsai-4B TQ1_0 **2.20 → 3.17 tok/s, prompt 2.70 → 7.91**; TV perplexity 13.394 vs Mac's stock 13.378 (rounding). Raw: `bench/raw/E38-*` |
| E39 | **Lossless ternary in less RAM**: load the original Q2_0_g64 file and repack it at load time into base-3 packing (5 weights/byte) that **keeps the per-64 scales**: 480 bytes per 8 rows × 256 instead of 576 = **1.875 bits/weight**. Needed: a `get_alloc_size` for the repack buffer (5/6 of the file size) and type-aware row offsets in the repack mat-mul (`repack_rows_bytes`). Aborts if a Q2_0 tensor holds the 4th code (+2); `GGML_Q2_0_TERNARY=0` = stock | The file is purely ternary (0 of 1.0 G bytes use +2; within a 256 block the 4 group scales differ by ~7 % median, what TQ1_0 rounds away). Mac A/B on the same file: **PPL 12.3729 (stock Q2_0) vs 12.3711 (new path)** with the 61,592-row head. TV: stock path thrashes (1.18 GB mapped: prompt 2.07, killed); new path **3.22 tok/s, prompt 5.60, peak RSS 996 MB**. Side finding: the 61,588-row heads of E37/E38 weren't repacked at all (rows not a multiple of 8); padded to 61,592 rows, TQ1_0 goes **3.17 → 3.51 tok/s** (prompt 7.82, RSS 945 MB). Raw: `bench/raw/E39-*` |
| E40 | **Threads and op profile** of the lossless 4B | 2 threads 2.54, **3 threads 3.20**, 4 threads 2.43 tok/s (Android needs the 4th core, as for the 2B). Profile per token: **MUL_MAT 90 %** (~15 G weights/s, what the kernel bench predicts), embedding lookup (GET_ROWS on the mapped Q2_0 table) **4.5 % = 14 ms/token** (page faults), flash attention 1.5 %. Raw: `bench/raw/E40-*` |
| E41 | **Embedding as a lazy, randomly-accessed mapping** (Qwen3/Qwen3.5 mark `token_embd` `TENSOR_READ_LAZY` when the head is pruned; active with `-lzm on`) | GET_ROWS 14.2 → 3.0 ms/token, but generation 3.11-3.13 vs 3.14-3.22 tok/s in alternating runs: **no gain**, left off (the default `auto` keeps stock behaviour). Raw: `bench/raw/E41-lazy-embd.txt` |
| E42 | **Unrolled 4-row kernel for the lossless format** (one 64-group at a time, its 16 (chunk, trit) pairs written out instead of a table) | Prompt **5.60 → 7.20 tok/s**, generation unchanged (3.21); Mac perplexity identical (12.3711). Raw: `bench/raw/E42-q20-gemm.txt` |
| E43 | **Real chat through llama-server** (`tv-llm-ui/start.sh`, `MODEL=tb4-q20-hml.gguf BIN=patched11`, `bench.py`, 4 prompts, greedy) | **3.07 tok/s generation, 7.2 tok/s prompt**, server RSS 963 MB, 193 MB left. Answers: Canberra correct (reason vague: "central… neutral"), TV-remote explanation correct, correct documented `is_prime`, Mars facts fluent but two errors (Olympus Mons "11× Everest", it's ~2.5×; overstates evidence of microbial life). Multilingual perplexity 14.35 (6.7 % misses). Raw: `bench/results.jsonl` (label `E43-tb4-q20-chat`) |

### 13.3 Where the 4B ends up

| Model on this TV | Bits/weight in RAM | Quality (bits/byte, lower is better) | Generation (tok/s) | Prompt (tok/s) | Peak RSS |
|---|---|---|---|---|---|
| **Qwen3.5-2B all-Q4_K + H65** (final 2B, §11.8) | 4.5 | **0.909** | **6.55** chat | 17.3 | ~0.99 GB |
| Qwen3.5-4B pure Q2_K | 2.6 | 0.972 | doesn't fit (E36) | | > 1.05 GB |
| Qwen3.5-4B IQ2_XXS | 2.06 | 1.139 | (not run: quality already out) | | |
| Bonsai-4B 1-bit (E25) | 1.125 | 1.073 | 3.17 | 11.4 | 1.09 GB |
| **Ternary-Bonsai-4B, lossless, new kernels** (`tb4-q20-hml.gguf`, `patched11`) | 1.875 | **0.964** (0.948 on the pruned head's positions) | **3.07** chat / 3.21 bench | 7.2 | 1.0 GB |
| Ternary-Bonsai-4B TQ1_0, new kernels (`tb4-TQ1_0-h8.gguf`) | 1.69 | 0.993 | **3.51** bench | 7.8 | 0.95 GB |
| Ternary-Bonsai-4B, stock llama.cpp | 1.69-2.25 | | 2.20 (TQ1_0) or thrashes (Q2_0, TQ2_0) | 2.7 | |

**Verdict.** The best 4B on this box is **Ternary-Bonsai-4B**, made usable by three new kernels and a lossless in-RAM format: from "thrashes" / 2.2 tok/s to **3.1-3.5 tok/s with 2.7-2.9× faster prompts**. It still loses to the 2B: about half the speed and ~6 % more bits per byte. A normal 4B (Qwen3.5-4B, the strongest one) cannot be squeezed into ~0.9 GB without collapsing (IQ2_XXS: 1.139 bits/byte). Other 4B-class models (Phi-4-mini, Gemma 3 4B, SmolLM3, Llama 3.2 3B) face the same ~2-bit wall with weaker starting points, so they weren't downloaded. Bonsai-8B (1.16 GB of 1-bit weights) is over the RAM budget.

**Not tried, with reasons:** speculative decoding with Ternary-Bonsai-1.7B as the draft (both models together need ~1.3 GB, and E31/E32 showed verification costs almost a full pass on this CPU); 4 threads (measured slower); software prefetch (measured slower in both kernel benches).

**Useful beyond the 4B:** the Q2_K and TQ1_0/TQ2_0 kernels speed up any model in those formats on Cortex-A55-class ARM chips (dotprod without i8mm): Q2_K +41 % generation / 2.4× prompt, TQ1_0 +44-60 % generation / 2.9× prompt.

**To use it:** `MODEL=tb4-q20-hml.gguf BIN=patched11 tv-llm-ui/start.sh` (quality), or `MODEL=tb4-TQ1_0-h8.gguf BIN=patched11 tv-llm-ui/start.sh` (+10 % speed, ~3 % worse bits/byte). The default stays the 2B.

## 14. The 9B push: streaming weights and a 1-bit kernel (2026-11-04)

Your request: "let's now try 9b models, and do the best in optimizing streaming and kernels and so on". Same rules (every TV run under `tv-guard.sh`, kill below 180 MB available; quality = bits per byte on `eval-en.txt`).

### 14.1 The constraint, measured

- **RAM:** with nothing running, MemAvailable is ~1.07-1.2 GB, and Android keeps ~350 MB of it free on purpose (its watermarks: the page cache never grew past ~880 MB in any run). One process can hold **~0.96 GB of anonymous memory** before the guard trips (E46). A 9B model is 2-2.8 GB at 2 bits, so **every 9B needs streaming**.
- **Flash** (`tools/kbench/flashbench.c`, E45): O_DIRECT reads 159 MB/s at 128 KB × 1 thread, **278 MB/s at 8 MB × 4 threads**; buffered reads 235-251 MB/s; the kernel's own page-fault path ~150-200 MB/s. So a token that must stream X MB costs at least X / 278 MB/s.

| Candidate (all 8-9 B) | Weights in our format | Must stream per token at ~0.9 GB resident | Quality (Mac, bits/byte) |
|---|---|---|---|
| Bonsai-8B (1-bit Q1_0, Qwen3-8B arch) | 1.01 GB | ~50-120 MB | 0.991 |
| Ternary-Bonsai-8B (Q2_0_g64, repacked 1.875 bits) | 1.69 GB | ~0.8 GB | 0.919 |
| Qwen3.5-9B (unsloth UD-IQ2_XXS, 65K head + MTP block) | 2.36 GB of mat-mul weights | ~1.4-1.7 GB | **0.885** |

### 14.2 Experiments

| # | What | Result |
|---|---|---|
| E44 | **New 1-bit kernel** (`ggml-cpu/repack-q1-tv.inc`): Q1_0 (sign bits + fp16 scale per 128) repacked into **bit planes**, 8 rows × 4 weights per 16-byte vector; `vtstq_u8` against `1<<k` turns plane k straight into a 0/-1 mask for `vdotq_laneq`, and the dot is `d·(−2T − S)` with S from the activation bsums. Two NEON ops per 16 multiply-adds, no table lookups (upstream expands every byte through a 256-entry table). Same bytes as the plain blocks. Gemv + 4-row gemm vs Q8_K; 4 accumulator chains. `GGML_Q1_TV=0` = upstream | `tools/kbench/q1_check.cpp` (includes the kernel file): exact to 1.2e-7 vs a double reference for gemv and gemm. TV, DRAM-sized matrix: **9.3 / 17.4 / 24.5 / 27.0 G weights/s at 1/2/3/4 threads** (2 chains: 20.8 at 3 threads). llama.cpp, Bonsai-4B: **generation 3.02 → 4.22 tok/s (+40 %), prompt 9.4 → 11.0**; perplexity 10.0139 vs upstream 10.0162 (6 chunks). Raw: `bench/raw/E44-*`, `tv-logs-9b/` |
| E45 | **Bonsai-8B quality, and does it fit?** (prism-ml/Bonsai-8B-gguf, 1.16 GB) | Mac: **PPL 13.877 = 0.991 bits/byte** (multilingual 19.63); with the 61,592-row head (cut from its untied `output.weight`, new `gguf_build.py` path): 13.25 / 15.58 on the kept positions (0.95 % / 6.7 % misses). TV: repacked weights need 1,012 MB of anonymous RAM; the guard stopped the load at 949 MB (MemAvailable 177 MB). **About 60-120 MB too big.** Raw: `bench/raw/E44-mac-ppl-b8.txt` |
| E46 | **Repack cache on flash** (`GGML_REPACK_CACHE=<prefix>` in `repack.cpp`): repacked weights go to a `MAP_SHARED` file instead of anonymous RAM; the first run writes it (tensor by tensor, written back and released), later runs map it read-only and skip the repack. Load in seconds; file-backed pages can be dropped under pressure instead of the process being killed | Works (rebuilt cache is byte-identical, `cmp`). But **0.11 tok/s**: with the weights as page cache, the cache holds only ~750 MB of the 1,012 MB working set (the ~350 MB Android keeps free counts against it), so every token re-reads it all. Plain page cache is the wrong tool; it only stays as the cache format. Raw: `bench/raw/tv-logs-9b/b8-cache*` |
| E47 | **Hybrid resident + streamed, v1** (`ggml-cpu/stream-tv.inc`, `GGML_REPACK_RESIDENT_MB`): the first N MB of the cache are copied into anonymous RAM, the rest is a *stream zone*. A hook at every MUL_MAT records the order zone tensors are used (the trace); a prefetch thread reads ahead in trace order into the page cache; the previous zone tensor is dropped after use (`MADV_DONTNEED` + `POSIX_FADV_DONTNEED`) so the stream never pushes anything else out | Bonsai-8B, resident MiB / window → tok/s: 800/128 0.94 · 850/16 0.99 · 850/128 1.39 · 900/128 1.85 · **920/128 2.24** (95 % of zone bytes already cached when used) · 940 and fully resident: guard kill. At 850 only 67 % was cached in time: prefetch can't keep up because flash is the limit. Raw: `bench/raw/E46-b8-stream-sweep.txt` |
| E48 | **Streaming v2: O_DIRECT into anonymous memory.** The zone's address range is remapped to empty anonymous pages; 3 reader threads `pread` 8 MB chunks with `O_DIRECT` straight into each tensor's own addresses (the 278 MB/s path, no page cache, no 4 KB faults), in trace order up to a window ahead; every compute thread waits until its weight is completely loaded, so a wrong prediction only costs time, never correctness; pages freed after use. `GGML_STREAM_MODE=cache` keeps v1's page-cache behaviour | **Bug found and fixed:** first results were garbage (Bonsai-8B PPL 80,946; uniform logits). `GGML_STREAM_VERIFY=1` (compares each tensor with the file when it is used) showed only the 4 KB pages two tensors share were wrong, and only with >1 reader: two concurrent O_DIRECT reads into one page corrupt it. Fix: shared edge pages are read once with a plain `pread` and kept; O_DIRECT covers whole pages only. After the fix: 0 mismatches; streamed Bonsai-8B PPL **8.1719 vs Mac 8.1632** (6 chunks; first 3 identical to the non-streamed TV run). Speed: Bonsai-8B 850/64 **1.52**, 880/32 1.66, 900/16 1.83 tok/s (the window now costs RAM, so less fits than in v1's 920/128 = 2.24); **Ternary-Bonsai-8B 0.21 → 0.31 tok/s** (796 MB streamed per token at ~250 MB/s; 4 readers 0.30; prompt 3.4). Raw: `bench/raw/E48-*` |
| E49 | **Ternary-Bonsai-8B** (prism-ml, Q2_0_g64, 2.31 GB) with the lossless 1.875-bit repack (E39) + 61,592 head | Mac: **PPL 11.457 = 0.919 bits/byte** (multilingual 13.92): almost the 2B's 0.909. Repack cache 1.69 GB. TV: **0.31 tok/s** generation, **3.45 tok/s** prompt (one streaming pass serves the whole batch). Raw: `bench/raw/E47-*`, `E48-tb8-sweep-fixed.txt` |
| E50 | **Qwen3.5-9B** (unsloth UD-IQ2_XXS: IQ2_XXS / IQ3_XXS / IQ2_S / Q4_K layers, Q5_K head, 3.19 GB), rebuilt for the TV in one pipe (`gguf_build.py`: 65,668-row head cut from the untied Q5_K head = 185 MB, + the model's own MTP block fetched from the BF16 safetensors and quantized to Q4_0 by `fetch_mtp.py --q4` = 137 MB, `block_count` 33, `nextn_predict_layers` 1): 2.81 GB | Mac: **PPL 9.859 = 0.885 bits/byte** (multilingual 11.32): **the best quality of anything tried on this box**. Greedy answer (same on the Mac): "The capital of Australia is **Sydney**": better perplexity, but a 2-bit 9B still gets facts wrong. TV, stock mmap: 0.08 tok/s. Streaming the mapped (non-repacked) weights needed a loader path (`LLAMA_STREAM_RESIDENT_MB`, `LLAMA_STREAM_PIN`: head + MTP pinned first) and a second bug fix: zones must stop at any tensor that is not a quantized 2-D weight (norms and conv kernels between the matrices were left as empty pages; found when the server answered "/****!**??"). After the fix: fluent, 0 `GGML_STREAM_VERIFY` mismatches. **Real chat (llama-server, `bench.py`, 4 prompts × 24 tokens, 600 MiB resident + head/MTP pinned, ~1.6 GB streamed per token): 0.16 tok/s, prompt 0.8 tok/s**, RSS 762 MB; answers correct in chat (Canberra, IR remote, `is_prime`, Mars). (A pre-fix llama-bench run at 800 MiB resident gave 0.18 tok/s / prompt 0.82: same I/O, wrong maths.) The server needed `-ctxcp 0 -cram 0`: by default it keeps up to 32 copies of the recurrent DeltaNet state (50 MB each for the 9B) and an 8 GB RAM prompt cache, which the guard killed at ~1 GB RSS. Raw: `bench/raw/E49-*`, `E50-q9-stream-mtp*.txt`, `bench/results.jsonl` |
| E51 | **MTP speculative decoding on the streamed 9B** (`--spec-type draft-mtp`, the model's own next-token block, pinned in RAM). On the 2B (E31) it lost because extra tokens cost almost a full compute pass; here a pass is dominated by flash reads, so verifying 2 tokens costs about the same as 1 | **n_max=1: 0.16 → 0.27 tok/s (+69 %), draft acceptance 91 %**, RSS 822 MB. n_max=2: guard kill at RSS 966 MB (one more DeltaNet snapshot per draft token); not retried with less resident weight at your stop. Raw: `bench/raw/E50-q9-stream-mtp.txt` |
| E52 | **Compact 1-bit scales: Bonsai-8B in 0.96 GB.** Bonsai's per-128 scales are all positive and within 0.44-1.0× of their row's maximum (range reads of 5 tensors from Hugging Face, `q1scales.py` in the session scratchpad), so each is stored as 1 byte: `d = dmin_row + code × dstep_row`, two fp16 row parameters per row in a 32-byte header per 8 rows. 17 bytes per 128 weights instead of 18 (`repack-q1-tv.inc`, now the default for Q1_0) | `q1_check`: exact (3e-7) vs a reference using the stored scales; scales vs fp16 originals **mean 0.07 %, max 0.38 %** (fp16's own rounding is ~0.05 %). Repack cache **1,012 → 962 MB**. Fully resident: prompt 5.06 tok/s, then guard kill at 176-179 MB available (1-4 MB short, also after `tv-free-ram.sh`). Near-full: **920 MB resident + 42 MB streamed: 2.21 tok/s** (direct mode) · 944 MB + 18 MB in page-cache mode: **2.23 tok/s** · 920 MB cache mode: 1.95. Raw: `bench/raw/E51-q1-compact-check.txt`, `E52-b8-compact.txt` |
| E53 | **Bonsai-8B in real chat** (llama-server, `bench.py`, 4 prompts × 64 tokens, KV cache q8_0 + flash attention, `-cram 0`, ctx 1024; the f16 KV cache alone would be 151 MB) | **839 MB resident (123 MB streamed, page-cache mode): 1.21 tok/s generation, 4.9 tok/s prompt**, RSS 900 MB. 912 MB resident (cache mode) and 922 MB (direct): guard kills at ~1.14 GB RSS, so ~840-880 MB is the chat ceiling. Answers: Canberra correct ("chosen to be the seat of government"), correct TV-remote explanation, correct documented `is_prime`, Mars facts fluent. Raw: `bench/raw/E53-b8-chat.txt`, `bench/results.jsonl` |

### 14.3 Where the 9B ends up

| Model on this TV | In RAM / streamed per token | Quality (bits/byte) | Generation (tok/s) | Prompt (tok/s) |
|---|---|---|---|---|
| **Qwen3.5-2B all-Q4_K + H65** (default, §11.8) | 0.99 GB / 0 | **0.909** | **6.55** chat | 17.3 |
| Ternary-Bonsai-4B, lossless (§13) | 1.0 GB / 0 | 0.964 | 3.07 chat | 7.2 |
| **Bonsai-8B 1-bit, compact scales** (`b8-q1-hml.gguf`, cache `b8c`) | 0.84 GB / 0.12 GB | 0.991 | **1.21 chat**, 2.2 bench | 4.9 |
| Ternary-Bonsai-8B (`tb8-q20-hml.gguf`, cache `tb8`) | 0.89 GB / 0.80 GB | 0.919 | 0.31 bench | 3.45 |
| **Qwen3.5-9B UD-IQ2_XXS + MTP** (`q9-iq2-h65-mtp.gguf`) | 0.63 GB / ~1.6 GB | **0.885** | 0.16 chat, **0.27 with MTP** | 0.8 |
| Qwen3.5-9B, stock llama.cpp (plain mmap) | page cache / everything | 0.885 | 0.08 | 0.65 |

**Verdict.** Your 1-2 tok/s target is reached only by **Bonsai-8B** (8.2 B parameters, trained 1-bit): **1.21 tok/s in real chat, 2.2 tok/s in llama-bench**, thanks to the new bit-plane kernel (+40 % on its 4B sibling), compact 8-bit scales (−50 MB) and the streaming engine for the last ~0.1 GB. It is the weakest of the three in quality (0.991 bits/byte, below the 2B's 0.909). The genuinely better 9B, **Qwen3.5-9B** (0.885, the best quality ever measured on this box), is flash-bound: ~1.6 GB of its 2.4 GB of weights must be read for every token at ≤ 278 MB/s, so even with streaming (2×) and MTP speculation (+69 %) it stays at **0.27 tok/s**. Reaching 1 tok/s with it would need ~1.2 bits per weight, where post-training quantization breaks normal models (E23, E35). **For everyday use the 2B stays the best trade-off.**

**What's reusable beyond these models:**
- **Q1_0 bit-plane kernel** (any Bonsai 1-bit model on dotprod-only ARM).
- **Repack cache:** `GGML_REPACK_CACHE` loads in seconds, and `GGML_REPACK_RESIDENT_MB` sets how much stays in RAM.
- **Streaming engine:** trace-ordered prefetch, parallel O_DIRECT reads into the weights' own addresses, drop-behind, and per-weight completion waits. It's enabled with `GGML_REPACK_RESIDENT_MB` for repacked weights or `LLAMA_STREAM_RESIDENT_MB` / `LLAMA_STREAM_PIN` for plain mapped ones, and tuned with `GGML_STREAM_MODE`, `GGML_STREAM_WINDOW_MB` and `GGML_STREAM_READERS`.
- **`GGML_STREAM_VERIFY`:** compares each streamed weight with the file, the check that found both bugs.

**Not tried, with reasons:**
- **MTP with 2+ drafted tokens at a lower resident size.** Stopped at your request.
- **A draft model for Ternary-Bonsai-8B.** A 1.7B draft costs ~200 MB of RAM, which would add ~0.8 s of streaming to every pass.
- **Lower-bit requantization of Qwen3.5-9B.** The Mac's 4 GB of free disk can't hold the BF16 source, and IQ1 quality collapses (E23).
- **4 threads.** Slower on this box (E40).

**To use them** (`tv-llm-ui/start.sh`, new `SENV=` passes the streaming settings):
- Bonsai-8B, about 1.2 tok/s:
  `MODEL=b8-q1-hml.gguf BIN=patched16 EXTRA="-ctk q8_0 -ctv q8_0 -fa on -cram 0" SENV="GGML_REPACK_CACHE=/data/local/tmp/llm/cache/b8c GGML_REPACK_RESIDENT_MB=800 GGML_STREAM_MODE=cache GGML_STREAM_WINDOW_MB=64" tv-llm-ui/start.sh`
- Qwen3.5-9B with MTP, about 0.27 tok/s:
  `MODEL=q9-iq2-h65-mtp.gguf BIN=patched16 EXTRA="-nr -ctxcp 0 -cram 0 -ub 16 -b 64 --spec-type draft-mtp --spec-draft-n-max 1" SENV="LLAMA_STREAM_RESIDENT_MB=600 LLAMA_STREAM_PIN=output.,blk.32. GGML_STREAM_WINDOW_MB=64" tv-llm-ui/start.sh`
- The default stays the 2B.

## 15. The 0.8B speed push (2026-11-04 → 05)

Your request: "go back to 0.8b models, push to the max speed possible, research the latest and try new things, be creative and don't limit yourself to Qwen". Same rules as before: every TV run goes through `tv-guard.sh`. Speed is `llama-bench` tg64 with 3 threads on frequent tokens (`LLAMA_BENCH_COMMON_TOKENS=4000`, as in E15) plus pp128. Quality is English perplexity on `eval-en.txt` (ctx 128 × 40 chunks), turned into bits per byte with each model's own token count. **Starting point:** Qwen3.5-0.8B Q4_K_M (unsloth), stock head, `patched16`: **7.47 tok/s**, prompt 28.3 (`bench/raw/E55-tv-speed.txt`).

### 15.1 Research: what's new at ~0.3-2B (Nov 2026)

Found on Hugging Face (`api/models`, model cards):
- **Liquid AI:** LFM2.5 (230M / 350M / 1.2B, with "QAD" Q4_0 files distilled for Q4_0), LFM2-700M, LFM3-1.7B (Oct 2026, "tri-hybrid"), and the d1 decision models (not generative, skipped).
- **PrismML:** **Ternary-Bonsai-2-2B** (Sep 2026, Qwen3.5-2B made ternary with a folded Hadamard rotation, 0.52-0.55 GB).
- **Others:** Gemma 3 1B, Llama 3.2 1B, Qwen3-0.6B, Granite 4.0 H 1B (hybrid Mamba-2), Falcon-H1-Tiny (90M-0.6B), Falcon-E-1B (BitNet; only an `i2_s` bitnet.cpp file, skipped), MobileLLM-R1 (reasoning-only, skipped).

Our llama.cpp v0.8.0 already supports every architecture above, including `lfm3`.

### 15.2 Experiments

| # | What | Result |
|---|---|---|
| E54 | **Screening of other small models** (`tools/screen-small.sh`: each GGUF streams from Hugging Face straight to the TV, runs speed + perplexity + one greedy answer, then is deleted; stock files, stock full heads, `patched16`) | Generation / prompt tok/s, bits/byte, max RSS: **Qwen3.5-0.8B Q4_0** 8.15 / 31.8, **1.024**, 750 MB · **LFM2.5-350M** QAD-Q4_0 **23.7** / 104, 1.805, 298 MB (answer correct; perplexity climbs on the later code-heavy chunks) · **LFM2-700M** Q4_0 11.5 / 52.8, 1.169, 523 MB · **LFM2.5-1.2B** QAD-Q4_0 8.26 / 31.6, 1.133, 783 MB (answer correct) · **Qwen3-0.6B** Q8_0 9.35 / 50.6, 1.059, 798 MB. Still running when this was written: Gemma 3 1B, Llama 3.2 1B, Granite 4.0 H 1B, Ternary-Bonsai-2-2B, LFM3-1.7B (an adb drop stopped the first attempt at Gemma). Summary: `bench/raw/E54-screen/summary.txt` |
| E55 | **Qwen3.5-0.8B rebuilt with the 2B recipe** (Mac): unsloth BF16 → `gguf_build.py` 65,668-row head (`data/head-ids-ml60k.txt`, same tokenizer as the 2B) → `llama-quantize --pure` with unsloth's 0.8B imatrix, one type everywhere | English PPL (Mac, head positions, 1.31 % misses): BF16 12.358 · Q8_0 12.368 · Q6_K 12.454 · Q5_K 12.626 · **Q4_K 13.241 (+7.1 %)** · Q4_0 14.168 (+14.6 %). TV gen / prompt tok/s: **Q4_K 12.14 / 33.1** (RSS 566 MB) · Q4_0 12.50 / 35.6 · Q5_K 9.21 / 26.1 · Q6_K 6.78 / 20.8 · Q8_0 8.23 / 29.4 (RSS 925 MB). **7.47 → 12.14 tok/s (+63 %)**: the stock file spent 43 % of each token on Q6_K matmuls, mostly the 248K-row head. Small models lose more to 4 bits than the 2B did (+7 % vs +2.3 %). Q4_K stays the format: Q4_0 is 3 % faster for twice the loss. Raw: `bench/raw/E55-*` |
| E56 | **DeltaNet reads its state from the cache by id** (patch, `patched17`): stock llama.cpp gathers each layer's 1 MB recurrent state into a temporary with GET_ROWS before the GDN op, every token. Now `build_rs` returns a view of the cache, and the GDN op gets the row ids (`src[7]`) and the cache (`src[8]`), reading row `s_copy[seq]` at compute time and updating it in place (row j of the state only needs its own row). `LLAMA_GDN_GATHER=1` = stock | **11.94 → 12.88 tok/s (+8 %)**; stock gather on the same binary 11.86. Perplexity and greedy text identical on the Mac (13.2409). Raw: `bench/raw/E56-byid.txt` |
| E57 | **Where the remaining time goes**: a node-by-node trace (`GGML_OPPROF=2`: per-thread compute time and start time of every node of one graph, `tools/optrace.py`), `LLAMA_UBATCH_PROF=1` (graph build / inputs / compute), `GGML_CPU_PROF=1` (plan vs node loop), and a barrier microbenchmark (`tools/kbench/barrier.c`) | Per token, ~73 ms: **Q4_K matmuls ~71 %** (52 ms; FFN and QKV at ~7 GB/s, the 65K head 7.7 ms at 4.9 GB/s, `z` 4 GB/s because it runs right after DeltaNet's 1 MB state write-back), DeltaNet 7 %, ~830 small nodes 12 %. Overheads are small: graph build 0.25-0.33 ms (the graph is reused), inputs 0.02 ms, plan 0.39 ms, barrier 0.25 µs idle (2.5 µs under load), start skew + gaps between nodes 1.8 ms. Two fixes: the head scatter's `FILL` of 248,320 logits ran one 1-element row at a time (1.2 ms), now one flat range per thread; for a single output row the scatter skips both transposes. **12.88 → 13.2-13.6 tok/s.** Run-to-run noise on this box is ±0.2-0.8 tok/s. Raw: `bench/raw/E56-trace17.txt`, `E57-*` |
| E58 | **Activation sparsity, simulated** (TEAL / CATS style, training-free: `LLAMA_SPARSE_SIM=a,d,g` zeroes the smallest-magnitude fraction `a` of every layer mat-mul input, `d` for `ffn_down`, in groups of `g` dims). Skipping the matching weight columns would cut bytes read | Q8_0 + H65, Mac PPL 12.368 → 25 %: 12.760 (+3.2 %) · 40 %: 14.772 (+19 %) · 50 %: 19.10 · 30 % / 60 % down: 14.06. **Grouped skipping, which a fast int8 kernel needs, collapses:** groups of 4 at 40 %: 44.4; groups of 16: 230. ✗ Not worth a kernel: at most ~15 % fewer bytes for +3 % perplexity. Raw: `bench/raw/E58-sparse-sim.txt` |
| E59 | **fp16 DeltaNet state, simulated** (`GGML_GDN_F16SIM=1`: the fused kernel rounds the stored state to fp16) | PPL ctx 128: 12.7040 → 12.7051; ctx 512: 12.7887 → 12.8018 (+0.1 %). Raw: `bench/raw/E59-gdn-f16-sim.txt` |
| E60 | **Approximate LM head** (`tools/head_approx.py`, hidden states dumped with `LLAMA_DUMP_HIDDEN`): score all tokens with a rank-r SVD of the head, re-score the top K exactly | The exact head reproduces the perplexity (12.707), so the setup is right. **Low rank can't find the token:** rank 128 / K 64 has the true top-1 in its candidates 52 % of the time (PPL 722); rank 384 / K 256, 83 % (PPL 71). ✗ The head is genuinely high-rank; the pruned head stays. Raw: `bench/raw/E60-head-lowrank-h65.txt` |
| E61 | **fp16 DeltaNet state, real** (`LLAMA_STATE_F16=1`, `patched18`): fp16 recurrent cache, a NEON fp16 path in the fused GDN kernel (read by id, convert in registers, write back in place), F16 `scale` for the state clear | Mac PPL ctx 512 12.7887 → 12.8058 (+0.13 %); greedy text stays fluent but diverges after ~2 sentences. TV: GDN 314 → 262 µs/call (−17 %), but a whole token only −0.7 % (72.7 → 72.2 ms over 65 graphs). Plain tg64 runs can't separate it from noise (f32 12.64 / 13.18, f16 13.41 / 12.92). Kept as an option, off by default. Raw: `bench/raw/E61-state-f16.txt` |

### 15.3 Where the 0.8B stands (so far)

| Qwen3.5-0.8B on this TV | Generation (tok/s) | Prompt (tok/s) | English PPL |
|---|---|---|---|
| Stock unsloth Q4_K_M, full head | 7.47 | 28.3 | not measured* |
| All-Q4_K + 65K head (E55), `patched16` | 12.14 | 33.1 | 13.24** |
| + state read by id (E56), `patched17` | 12.88 | | 13.24** |
| **+ flat FILL, transpose-free scatter (E57)** | **13.2-13.6** | ~33 | 13.24** |
| + fp16 state (E61, optional) | +0.7 % | | 13.26** |

\* the stock unsloth Q4_0 file (full vocabulary, TV) has PPL 14.14 = 1.024 bits/byte (E54); \*\* 65K-head positions (1.31 % misses), Mac; not comparable with full-vocabulary numbers.

**What limits it now:** about 71 % of each token is Q4_K matmuls, already at ~7 GB/s on the big matrices (the kernel's DRAM ceiling measured in E27 is 6.7-8.3 GB/s). The remaining small ops and sync are ~15 % together. A big step past ~14 tok/s needs fewer weight bytes per token: a smaller or ternary model, which is what E54 is screening.

**Not done yet:** a real-chat run through llama-server (sampling over the full 248K vocabulary is not in the llama-bench numbers), and the remaining E54 models.

**New environment switches** (all off by default; the patch is `patches/llama.cpp-v0.8.0-tvbox.patch`):
- `LLAMA_GDN_GATHER=1`: stock state gather instead of the by-id read (E56).
- `LLAMA_STATE_F16=1`: fp16 recurrent state, Qwen3.5 only, no MTP (E61).
- `GGML_OPPROF=2` (+ `GGML_OPPROF_GRAPH=n`): node-by-node trace of one graph; `LLAMA_UBATCH_PROF=1`, `GGML_CPU_PROF=1`: phase timings (E57).
- `LLAMA_SPARSE_SIM=a,d,g`, `GGML_GDN_F16SIM=1`, `LLAMA_DUMP_HIDDEN=<file>`: quality simulations (E58-E60).

**To use it:** `tools/rebuild-models.sh 08` builds `q08-q4k-h65.gguf` (SHA-256 `901f3838…8523a6b`), then `MODEL=q08-q4k-h65.gguf BIN=patched18 tv-llm-ui/start.sh`. The default stays the 2B.
