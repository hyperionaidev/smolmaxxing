# Device #2: iPad Air (2013)

llama.cpp, running locally on a first-generation iPad Air: A7 (2 cores, NEON, no dotprod), 1 GB of RAM, iOS 12.5.7, CPU only. The full write-up, including what failed, is in [`EXPERIMENT.md`](../EXPERIMENT.md) §21.

## Results

Every speed uses 2 threads and a 1,024-token context, through `smolchat` on a fixed three-turn conversation. The ranges are across those turns, on a run that didn't follow a large file copy.

<p align="center"><img src="../assets/scoreboard-ipad.svg" width="760" alt="Best model per size class on the iPad Air, first run to now, and where one Moon-Beam token goes"></p>

| Model | File | Generation | Prompt | Peak RSS | Three-question test |
|---|---|---|---|---|---|
| SmolLM2-135M-Instruct Q4_K_M | 101 MB | 18.6-29.3 tok/s | 42-52 tok/s | 141 MB | right on France and on recall, weak on tides |
| Gemma 3 270M-it Q4_K_M | 241 MiB | 15.4-19.0 tok/s | 45-64 tok/s | 248-294 MB | wrong on tides, empty reply on recall |
| Bonsai-4B Q1_0 (1-bit, `-c 512 -b 64`) | 546 MiB | 9.4-11.4 tok/s, 9.2-10.3 through long chats; as a server, iOS resets it within minutes | 11-14 tok/s (first about 3.5) | 615-682 MB | right on France, tides mostly right, fails recall every run; better per byte than the 1.7B |
| Moon-Beam-8B-A1B (MoE, Q2_K experts, expert pager, `-c 512 -b 64`) | 2.6 GiB, 474 MB mapped | 3.5-3.6 tok/s on longer answers (lookup-table file, prefetch, frequency cache); 4.9-5.1 with `--experts 4` | 4.2-4.3 tok/s | about 600-625 MB | all three right; a session model, not for the always-on server |
| **Ling-mini-2.0 (16.3B MoE, 1.4B active; Q2_K experts, expert pager, `-c 512 -b 64`)** | 5.2 GiB, 466 MB mapped | **2.6 tok/s** with unchanged routing; 3.3 with bias 0.1 (+2.6 % perplexity); 4.4-4.6 with bias 0.3; 5.2 adding `--experts 6` | 2.8-3.6 tok/s | 556-576 MB | not run (perplexity in §21.11); the biggest model on the iPad, but not better than Moon-Beam (+3 % perplexity); doesn't fit on a 16 GB iPad next to it |
| SmolLM2-360M-Instruct Q4_K_M | 258 MiB | 10.9-12.5 tok/s | 18-20 tok/s | 269-316 MB | all three right; the lightest good option |
| **Bonsai-1.7B Q1_0 (1-bit, LUT kernel)** | 237 MiB | **23.7-25.8 tok/s**, 18-22 through long chats | 29-33 tok/s (first prompt about 7) | 605 MB | **all three right, and writes correct code: the one to use** |
| Qwen3.5-0.8B Q4_K_M | 508 MiB | about 0.2 tok/s | n/a | 506 MB | thrashes the page cache, stopped by hand |

`stories15M-q4_0` (18 MB) reaches about 290 tok/s in `llama-simple`.

Bonsai-1.7B started at 2.75 tok/s. A multiplication-free NEON kernel took it to 3.45 with identical output, and the TV's lookup-table kernel, ported to CPUs without dotprod, took it to about 11. Giving the shared output head that kernel, reading attention row by row with pruning, and a faster sampler then took it to about 25. See `EXPERIMENT.md` §21.4 and §21.5. Bonsai-4B needs `-c 512 -b 64`: about 700 MB is the practical memory ceiling here, and a longer context crosses it (§21.6).

### Moon-Beam-8B-A1B with the expert pager

The model is bigger than a process may even map on iOS 12, so its experts are read from flash as the router picks them (§21.8-21.10). Rewrite the file once with its experts last, then repack the experts for the lookup-table kernel, copy it over, and run:

```sh
python3 ../tools/gguf_experts_last.py models/mb-q2-h60.gguf models/mb-q2-h60-xl.gguf
python3 ../tools/gguf_q2lut.py models/mb-q2-h60-xl.gguf models/mb-q2-h60-lut.gguf
GGML_MOE_PAGER=1 GGML_MOE_POOL=14 GGML_MOE_BIAS=0.03 GGML_MOE_PREFETCH=1 GGML_MOE_LFU=0.95 ./smolchat -m mb-q2-h60-lut.gguf -c 512 -b 64 --no-think --repeat-penalty 1.15 --repeat-last 256
```

`GGML_MOE_POOL` is experts kept in RAM per layer (17.8 MB each across the model). 14 is the tested maximum; at 16 the iPad kernel-panicked and rebooted, and so did wiring the weights with `LLAMA_MLOCK_HOT=1`: don't use either. `GGML_MOE_PREFETCH=1` reads the next layer's 8 likeliest experts early (`GGML_MOE_PREFETCH_K` to change the count). `GGML_MOE_LFU=0.95` evicts the least frequently used expert (recent uses count more) instead of the least recently used: about 9 % fewer reads, same quality. `--experts N` uses N experts per token instead of 8: 7 gives about 3.9 tok/s (+1-2 % perplexity), 6 about 4.3 (+4-5 %), 4 about 5.0 (+27 %). `GGML_MOE_BIAS` trades quality for speed: 0.05 gives about 3.7 tok/s at +1.2 % perplexity. The `-xl` file also works, a little slower (2.94 vs 3.06 tok/s with top-4 prefetch). `GGML_MOE_STATS=1` prints hits, misses and flash time at exit. The source file is `mb-q2-h60.gguf` from [hyperionaidev/Moon-Beam-8B-A1B-tvbox-GGUF](https://huggingface.co/hyperionaidev/Moon-Beam-8B-A1B-tvbox-GGUF). The finished iPad files are on Hugging Face: `-lut` at [hyperionaidev/Moon-Beam-8B-A1B-ipad-GGUF](https://huggingface.co/hyperionaidev/Moon-Beam-8B-A1B-ipad-GGUF) and `-xl` at [hyperionaidev/Moon-Beam-8B-A1B-ipad-xl-GGUF](https://huggingface.co/hyperionaidev/Moon-Beam-8B-A1B-ipad-xl-GGUF).

### Ling-mini-2.0 (16B) with the expert pager

<p align="center"><img src="../assets/tradeoff-ipad.svg" width="760" alt="Speed against perplexity cost for Moon-Beam-8B-A1B and Ling-mini-2.0 on the iPad"></p>

The same pager runs a 16.3B model (§21.11). Its 256 experts per layer are only 1.03 MB each, so it's as fast as Moon-Beam at twice the size. Download it from [hyperionaidev/Ling-mini-2.0-ipad-GGUF](https://huggingface.co/hyperionaidev/Ling-mini-2.0-ipad-GGUF), or rebuild it: `../tools/lm-ipad.sh` builds the file without ever storing it on the Mac: it takes the non-expert tensors of bartowski's Q4_K_M and the Q2_K experts of mradermacher's imatrix Q2_K_S over HTTP range requests, repacks the experts for the lookup-table kernel, and streams the result over SSH to the iPad (5.55 GB; check the md5 it prints). Then:

```sh
GGML_MOE_PAGER=1 GGML_MOE_POOL=14 GGML_MOE_PREFETCH=1 GGML_MOE_LFU=0.95 GGML_MOE_BIAS=0.1 ./smolchat -m lm2-q2-h60-lut.gguf -c 512 -b 64 --repeat-penalty 1.15 --repeat-last 256
```

Bias 0 is full quality at about 2.6 tok/s; 0.1 gives 3.3 (+2.6 % perplexity); 0.3 gives 4.5 at roughly +9-18 %. Don't pass `--no-think`. `tinyppl` measures perplexity on the device: `./tinyppl -m lm2-q2-h60-lut.gguf -f eval-en.txt -c 512 --chunks 2 --experts 8` (with the same environment).

## Rebuilding it

You need a jailbroken device. This one uses [Amethyst](https://github.com/staturnzz/amethyst), following the [cfw.guide instructions](https://ios.cfw.guide/installing-amethyst/). The jailbreak does not survive a reboot. Change the default `root` and `mobile` passwords before anything else, and install `ldid` from Procursus on the device.

1. Put the Xcode 15.4 SDK at `third_party/iPhoneOS.sdk`. It is the last SDK that targets iOS 12. Only the SDK is used, so Xcode doesn't have to run.
2. Clone llama.cpp to `third_party/llama.cpp`, check out `1945793e00c53f69eac0bfc89e37429a5503f12a`, and apply `patches/llama.cpp-ios12.patch`.
3. Run `./build.sh`. Binaries land in `bin/`.
4. Forward SSH over USB with `pymobiledevice3 usbmux forward 2222 22`, and authorize an SSH key for `root`. Point `IPAD_KEY` at it.
5. Run `./deploy.sh models/SmolLM2-360M-Instruct-Q4_K_M.gguf`, then `./chat.sh SmolLM2-360M-Instruct-Q4_K_M.gguf`.

### Web chat in Safari

`smolchat --serve 8080` serves a chat page written for Safari 12. Start it on the iPad, then open `http://127.0.0.1:8080` in Safari:

```sh
cd /var/mobile/llama && ./smolchat -m Bonsai-1.7B-Q1_0.gguf --no-think --serve 8080
```

It listens only on the iPad itself unless you pass `--host`, and it has no login, so don't expose it to a network you don't trust. To have it start on its own without a terminal, run `./service.sh install` from the Mac. `./service.sh remove` undoes that.

`models/`, `bin/` and `third_party/` are not in git. The models are [prism-ml/Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf), SHA-256 `3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3`, and [bartowski/SmolLM2-360M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF), SHA-256 `2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2`.

## Files

| Path | What |
|---|---|
| `smolchat.cpp` | interactive chat: repeat penalty, 2-thread default, drops the oldest turns when the context fills |
| `../tools/make_ipad_svgs.py` | draws `assets/scoreboard-ipad.svg`, `devices.svg` and `tradeoff-ipad.svg` from the numbers in §21 |
| `tinyppl.cpp` | perplexity on the device with `llama.h` only, one token at a time (llama.cpp's own tool needs `std::filesystem`) |
| `probes/` | `fbench.c` (uncached sequential flash reads), `rrbench.c` (uncached random reads, as the expert pager does them) and `vmprobe.c` (largest mapping a process can make) |
| `web.h` | the `--serve` HTTP server and its Safari 12 chat page |
| `service.sh`, `smolchat.plist` | optional background service for the web chat |
| `build.sh` | CPU-only cross-compile for iOS 12 |
| `deploy.sh` | copies binaries and models to the iPad and signs them |
| `chat.sh` | opens a chat on the iPad over SSH |
| `bench.sh` | the fixed three-turn test, with memory and time limits |
| `ent.xml` | the entitlements binaries need to launch |
| `kbench.c` | checks the new 1-bit kernel against the old one for exact equality, and times both |
| `patches/llama.cpp-ios12.patch` | compiles out `std::filesystem` behind `GGML_NO_FS`, adds `LLAMA_THREADS` to `llama-simple`, the two 1-bit kernels (`GGML_Q1_LUT=0` turns the LUT one off), pruned attention, the expert pager (`GGML_MOE_PAGER`), and pruned LM heads for `bailingmoe2` and `bailingmoe3` |

## What matters on this device

- **Use 2 threads.** llama.cpp's default of 4 on these 2 cores made `stories15M` 540× slower (0.54 vs 290 tok/s).
- **Sign with `ent.xml`.** Ad-hoc signing or a plain `ldid -S` gets the process killed at launch (exit 137).
- **Don't time runs right after copying a large file.** Speeds were about 3× lower until the device settled.
- **The model has to fit in RAM.** Roughly 650 MB is available with the iPad's normal apps running. A 508 MiB model that also needs working memory pages constantly.
