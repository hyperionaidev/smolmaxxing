# 📲 Device #6: Realme 5 Pro

Snapdragon 710: 2x Kryo 360 Gold ("Cortex-A76-based") at 2.3 GHz and 6x Kryo 360 Silver (Cortex-A55) at 1.7 GHz, **6 GB of RAM** (about 3.2-3.6 GB available at idle with the phone's own apps running), eMMC 5.1 flash, Android 10, kernel 4.9, no root. Everything runs on the CPU as the adb `shell` user, over wifi ADB (`adb tcpip 5555`), everything in `/data/local/tmp/llm`.

[← back to the main README](../README.md) · [build recipe](tv-box-setup.md) (one flag change, below)

## The finding: this "A76" has no dotprod

`/proc/cpuinfo` Features on the Kryo 360 Gold:

```
fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp
```

`asimddp` is missing. Qualcomm's semi-custom Kryo 360 carries the fp16 vector extension (`fphp asimdhp`) but **not the dot-product extension**, unlike the stock Cortex-A55 of the TV box and the stock A78 of the OnePlus. The stock `zigcc` wrapper (`-mcpu=cortex_a55`, dotprod+fp16) builds a binary that dies with `Illegal instruction` on the first matmul on this chip.

The fix is a wrapper with the feature list matching the kernel's own report:

```sh
exec zig cc -target aarch64-linux-musl -mcpu=generic+fullfp16 "$@"
```

(LLVM spells the feature `fullfp16`; `fp16` is rejected.) Configure with `-DGGML_CPU_ARM_ARCH=armv8-a` and otherwise the [tv-box-setup.md](tv-box-setup.md) recipe unchanged. The resulting binary is the safe superset: plain NEON v8.0 plus fp16, no `sdot`/`udot`/`smmla`, so it also runs on every dotprod device - it just doesn't take the dot-product fast path there.

One more build note: on a fresh v0.8.0 checkout the UI-assets provisioning falls back to the `latest` HF bundle, which no longer ships `workbox[hash].js`; the check in `scripts/ui-assets.cmake` that requires it needs dropping before the build completes.

## Results

Generation speed, `llama-bench -p 0 -n 128`, mean of the repo default 2 runs, patched llama.cpp v0.8.0 (the merged armv7 patch rev):

| Model | File | 2x Gold pinned (`-t 2 -C 0xC0`) | all 8 cores (`-t 8`) |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | 604 MB | 9.99 | **14.57** |
| Qwen3-0.6B Q4_K_M | 402 MB | 9.10 | 14.46 |
| Qwen3-1.7B Q4_K_M | 1.03 GB | 3.38 | 6.18 |
| Qwen3-1.7B Q8_0 | 1.70 GB | 3.71 | 5.68 |
| Llama-3.2-3B-Instruct Q4_K_M | 1.87 GB | 1.86 | 3.49 |
| Qwen3-4B Q4_K_M | 2.32 GB | 1.48 | 2.81 |
| Ling-mini-2.0 16B MoE (pager, pool 128, bias 0.3) | 5.5 GB | **5.13** | 4.73 |

Ling-mini-2.0 runs with the [phone.md](phone.md) pager recipe (pool 80 -> 96 -> 128 gave 4.90 -> 4.99 -> 5.13 tok/s; 3.5 GB available was enough for pool 128). The 5.5 GB model on eMMC, served to the LAN, was the biggest model tried - and then passed over: it buys 16B-class answers at the speed of the 1.7B dense models, so the deployment stays with the 0.6B + 1.7B pair below.

## What the missing dotprod costs

- **Quant level is nearly free.** Q8_0 and Q4_K_M decode at the same speed (14.57 vs 14.46 tok/s on the 0.6B). With `sdot` the int8 path is the fast one; without it both quants go through the same NEON lanes. Run the higher quant - the quality is free.
- **All 8 cores beat the 2 fast ones for dense models** - the opposite of the OnePlus A78 result. Without a dot-product instruction a token needs more (parallelizable) vector work, and the six 1.7 GHz A55s contribute more than they cost; with dotprod the A78s alone were the better team because barriers stalled on the slow cores.
- **Big vocabularies hurt twice.** Llama-3.2-3B is half the speed of the smaller-vocab Qwen3-1.7B despite the same weight class. A 128K-token output head without `sdot` is proportionally the most expensive part of the model.
- The MoE pager works fine off eMMC, but the flash is the cap: pool 128 only bought +5 % over pool 80, where the OnePlus (UFS 2.2 at ~940 MB/s) keeps gaining.

## Which devices this applies to

dotprod is optional in ARMv8.2 and only became mandatory in ARMv8.4. So the no-dotprod class includes:

- stock Cortex-A53/A73 and everything older (the 2016-2019 midrange: Snapdragon 660-class, Helio P60/P70, most E- and mid-range chipsets)
- semi-custom cores that don't carry it even when the marketing says "A76-based": Kryo 360 (Snapdragon 710/712 - this device) is measured; other Kryos need measuring
- the repo's own iPad Air (Apple A7, also no dot-product instructions) already runs this class of kernels

Check any Android device with:

```
adb shell grep -o asimddp /proc/cpuinfo
```

Empty answer: use the `generic+fullfp16` wrapper. `asimddp` present: the stock `zigcc` is fine. Either way the no-dotprod binary runs - it is the safe superset.

## Everyday setup

Everything as the adb shell user, nothing rooted, nothing outside `/data/local/tmp/llm`:

- **Everyday model: Qwen3-0.6B Q8_0 at 14.6 tok/s** (`-t 8`), 0.7 GB RSS, 3+ GB left for the phone's own apps. Qwen3's thinking mode is on by default; clients that want instant answers pass `"chat_template_kwargs": {"enable_thinking": false}`.
- **Quality alternative: Qwen3-1.7B Q8_0 at 5.7-6.2 tok/s**, 1.9 GB RSS.
- wifi ADB (`adb tcpip 5555`) survives screen-off; after a reboot it needs one USB replug to re-arm. A 5-minute health-check loop on the workstation restarts the server if it dies.
