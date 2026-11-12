# Models: downloads and rebuilds

Model files (0.5-5.5 GB each) are not in git. You can download the finished files from Hugging Face, or rebuild every one from public sources.

[← back to the main README](../README.md)

## TV box

The best file of each size class, exactly as the TV runs it. They need the patched llama.cpp ([setup](tv-box-setup.md)); stock llama.cpp can't load the pruned head. Each card has the run command and the measured numbers.

| Model | Speed on the TV | Hugging Face | File |
|---|---|---|---|
| Qwen3.5-0.8B | 16.8 tok/s | [hyperionaidev/Qwen3.5-0.8B-tvbox-GGUF](https://huggingface.co/hyperionaidev/Qwen3.5-0.8B-tvbox-GGUF) | `q08-q4k-h65p8.gguf` |
| **Qwen3.5-2B** (default) | 7.7 tok/s | [hyperionaidev/Qwen3.5-2B-tvbox-GGUF](https://huggingface.co/hyperionaidev/Qwen3.5-2B-tvbox-GGUF) | `w-q4k-h65p8.gguf` |
| Ternary-Bonsai-4B | 4.3 tok/s | [hyperionaidev/Ternary-Bonsai-4B-tvbox-GGUF](https://huggingface.co/hyperionaidev/Ternary-Bonsai-4B-tvbox-GGUF) | `tb4-q20-hml.gguf` |
| Moon-Beam-8B-A1B (MoE) | 7.2 tok/s | [hyperionaidev/Moon-Beam-8B-A1B-tvbox-GGUF](https://huggingface.co/hyperionaidev/Moon-Beam-8B-A1B-tvbox-GGUF) | `mb-q2-h60.gguf` + `moe-warm-moonbeam.txt` |
| Bonsai-8B (1-bit) | 5.2 tok/s | [hyperionaidev/Bonsai-8B-tvbox-GGUF](https://huggingface.co/hyperionaidev/Bonsai-8B-tvbox-GGUF) | `b8-q1-hml.gguf` |
| Ling-mini-2.0 (16B MoE) | 5.1 tok/s | [hyperionaidev/Ling-mini-2.0-tvbox-GGUF](https://huggingface.co/hyperionaidev/Ling-mini-2.0-tvbox-GGUF) | `lm2-q2-h60.gguf` + `moe-warm-ling.txt` |

Download with `hf download hyperionaidev/<repo> --local-dir models/`, then `adb push` the files to `/data/local/tmp/llm/`, with `moe-warm-*.txt` next to them. That replaces the `08` / `2b` / `4b` / `mb` / `b8` / `lm` rebuild targets. For the 8B and 16B models, the first start writes the repack cache.

## iPad Air

The three mixture-of-experts files are our own builds, laid out for the iPad's expert pager and lookup-table kernel. They need the [iOS 12 patch](../ipad-air/README.md); any other llama.cpp loads them but gets scrambled experts. The other models are used exactly as published, and the Bonsai models' fast layouts are built in RAM at load time.

| Model | Download | Speed on the iPad (tok/s) | Peak RSS | Experiments |
|---|---|---|---|---|
| **Bonsai-1.7B (1-bit), the default** | [prism-ml/Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) | **23.7-25.8**; 18-22 through long chats | 605 MB | 11 (E118-E128) |
| Bonsai-4B (1-bit), `-c 512 -b 64` | [prism-ml/Bonsai-4B-gguf](https://huggingface.co/prism-ml/Bonsai-4B-gguf) | **9.4-11.4**; 9.2-10.3 through long chats | 615-682 MB | 5 (E129-E133) |
| Moon-Beam-8B-A1B (MoE) | [hyperionaidev/Moon-Beam-8B-A1B-ipad-GGUF](https://huggingface.co/hyperionaidev/Moon-Beam-8B-A1B-ipad-GGUF): `mb-q2-h60-lut.gguf` (SHA-256 `acf1cb56…94b976`) | **3.5-3.6**; 3.7-4.0 with 7 experts (+0.4-1.9 %); 5.0 with 4 experts (+27 %) | 600-625 MB | 19 (E138-E156) |
| ↳ without the lookup-table repack, about 4 % slower | [hyperionaidev/Moon-Beam-8B-A1B-ipad-xl-GGUF](https://huggingface.co/hyperionaidev/Moon-Beam-8B-A1B-ipad-xl-GGUF): `mb-q2-h60-xl.gguf` (`aa919400…871d7e`) | | | |
| Ling-mini-2.0 (16B MoE) | [hyperionaidev/Ling-mini-2.0-ipad-GGUF](https://huggingface.co/hyperionaidev/Ling-mini-2.0-ipad-GGUF): `lm2-q2-h60-lut.gguf` (SHA-256 `3e464310…4e50bd`) | **2.6** unbiased; 3.3 with bias 0.1 (+2.6 % perplexity); 4.5 with bias 0.3; 5.2 adding `--experts 6` | 556-576 MB | 4 (E157-E160) |
| Bonsai-8B (1-bit, dense) | [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | 0.08: doesn't fit in RAM, read from flash | 629 MB | 4 (E134-E137, with the first Moon-Beam attempts) |
| SmolLM2-360M-Instruct Q4_K_M | [bartowski/SmolLM2-360M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF) | 10.9-12.5 | 269-316 MB | 10 (E108-E117: first runs, the small models and the web chat) |
| Gemma 3 270M-it Q4_K_M | [unsloth/gemma-3-270m-it-GGUF](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF) | 15.4-19.0 | 248-294 MB | |
| SmolLM2-135M-Instruct Q4_K_M | stock Q4_K_M | 18.6-29.3 | 141 MB | |
| Qwen3.5-0.8B Q4_K_M | [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | about 0.2: thrashes | 506 MB | |
| stories15M q4_0 | stock | about 290 in `llama-simple` | | |

53 experiments in all (E108-E160, §21 of [EXPERIMENT.md](../EXPERIMENT.md)).

## Rebuilding the TV files

[`tools/rebuild-models.sh`](../tools/rebuild-models.sh) recreates every TV file from public Hugging Face files, streaming them through `tools/gguf_build.py` onto the TV. The SHA-256s below are of the files used in the experiments. Rebuilds of the 4B, Bonsai-8B and 9B files were checked **byte-identical**.

| File on the TV | Target | Built from | SHA-256 |
|---|---|---|---|
| `w-q4k-h65.gguf` (Qwen3.5-2B) | `2b` | `unsloth/Qwen3.5-2B-GGUF` BF16 + `imatrix_unsloth`, MTP block from `Qwen/Qwen3.5-2B`, quantized on the TV, head `data/head-ids-ml60k.txt` | `16057be3…3285b` |
| `w-q4k-h65p8.gguf` (Qwen3.5-2B, **default**) | `2b` | the file above re-cut on the TV with `data/head-ids-ml60k-p8.txt` (65,672 rows, so the head is repacked, §17) | `631a27ae…54fd11` |
| `w-q4k-h45.gguf` (option) | `2b45` | the same, 45,654-row head `data/head-ids-ml40k.txt` (byte-identical to E29's H45 file) | `a50eadc5…664d5b` |
| `q08-q4k-h65.gguf` (Qwen3.5-0.8B) | `08` | `unsloth/Qwen3.5-0.8B-GGUF` BF16 + its `imatrix_unsloth`, pure Q4_K, head `data/head-ids-ml60k.txt`, quantized on the Mac | `901f3838…8523a6b` |
| `q08-q4k-h65p8.gguf` | `08` | the file above re-cut on the Mac with `data/head-ids-ml60k-p8.txt` (§17) | `7dd2aac5…52fc232` |
| `tb4-q20-hml.gguf` | `4b` | `prism-ml/Ternary-Bonsai-4B-gguf` Q2_0_g64, head `data/head-ids-qwen3-ml64k.txt` | `a02d805f…3fe24` |
| `b8-q1-hml.gguf` | `b8` | `prism-ml/Bonsai-8B-gguf` Q1_0, same head | `fc297cb3…0d42a` |
| `tb8-q20-hml.gguf` | `tb8` | `prism-ml/Ternary-Bonsai-8B-gguf` Q2_0_g64, same head | `01b1e0ef…93e06` |
| `mb-q2-h60.gguf` | `mb` | `tablinum/Moon-Beam-8B-A1B-GGUF-Preview` Q8_0, requantized on the Mac: experts Q2_K, the rest Q4_K, own imatrix from `data/corpus`, head `data/head-ids-ling-60k.txt`; expert bytes hole-punched on the TV after the repack cache (`cache/mbq2.*`) is written | `2d78fcdd…1c34ee` |
| `lm2-q2-h60.gguf` | `lm` | `bartowski/inclusionAI_Ling-mini-2.0-GGUF` Q4_K_M + its imatrix: experts requantized to Q2_K on the Mac, the rest Q4_K, head `data/head-ids-ling-60k.txt`; punched on the TV after the repack cache (`cache/lmq2.*`) is written | `c089aeb0…e8d7` |
| `q9-iq2-h65-mtp.gguf` | `q9` | `unsloth/Qwen3.5-9B-GGUF` UD-IQ2_XXS, MTP block from `Qwen/Qwen3.5-9B` (Q4_0), 65K head | `7766478e…8dcbc` |
| `cache/b8c.*`, `cache/tb8.*` | `caches` | written by the first run with `GGML_REPACK_CACHE` | |

The head-id lists in `data/` were made by `tools/vocab_select.py` (Qwen3.5 tokenizer) and `tools/head_select_tok.py` (any tokenizer, via `llama-tokenize`), from the corpus `tools/fetch_corpus.py` downloads (Wikipedia + Project Gutenberg).

## Rebuilding the iPad files

The iPad's mixture-of-experts files are built from the TV ones and published sources. The steps are in the [iPad README](../ipad-air/README.md#moon-beam-8b-a1b): `tools/gguf_experts_last.py` and `tools/gguf_q2lut.py` for Moon-Beam, and `tools/lm-ipad.sh` for Ling-mini-2.0.
