# Lessons

What 165 experiments on three devices taught us, including the things that didn't work. Each one points to the experiments in [EXPERIMENT.md](../EXPERIMENT.md).

[← back to the main README](../README.md)

## Models and quantization

- **On the TV box, the 2B at 4 bits beats every bigger model at 1-2 bits.** Post-training quantization below about 2.5 bits breaks normal models (E22, E23, E35). Models trained for 1-bit or ternary hold up, but at 4-8 B they don't fit without streaming.
- **Small models are mostly vocabulary.** Qwen3.5-0.8B's 248K-row output head is a third of its weights; pruning it to 65K rows plus all-Q4_K took it from 7.5 to 12.1 tok/s (E55). Small models also lose more to 4 bits (+7 % perplexity vs +2.3 % for the 2B).
- **Bigger isn't automatically better.** Ling-mini-2.0 (16B) is twice Moon-Beam's (8B) size on both devices, runs no faster, and scores worse per byte.
- **Perplexity can't see loops.** It scores one token at a time on clean text; the 2-bit model scored fine and still fell into loops in 12 of 13 long answers. Score long free-running generations too (`tools/loop_stats.py`), and compare models with different tokenizers in bits per byte, never in perplexity.

- **A pruned head that's fine for a 2B can break a 35B.** The 60K-row list made Qwen3.6-35B-A3B spell "palindrome" as "palinate": its whole-word token wasn't kept. The full head costs the phone only 3 % (E162, E163).
- **Pruning experts by how often they're used needs far more calibration than it seems.** Keeping the 192 most-used of 256 experts, which carried 98.6 % of 3,840 calibration tokens' uses, cost Qwen3.6 +34 % perplexity (E164).

## Speed

- **On an in-order core, count NEON ops as well as bytes.** Tighter packing saved 20 % of the bytes and cost 2.5× the ops; undoing it made the 4B 33 % faster in chat (E76, E77). The 4-bit kernel is the opposite case, limited by RAM (E27), so the trick doesn't help the 2B or 0.8B.
- **Check that every big matrix reaches the fast kernel.** It happened on both devices. The TV's 2B head (65,668 rows, about 10 % of each token's weights) silently ran on the generic kernel for weeks because ggml only repacks row counts divisible by 8; four extra rows gave +5.8 % (E71, E72). On the iPad, Bonsai-1.7B's 151,669-row head spent 45 of every 72 ms on the slow path. The load log's repack buffer size gives it away.
- **Question a kernel's requirements.** The TV's 1-bit lookup-table kernel was written for dotprod CPUs, but it didn't need dotprod: ported to the iPad's base NEON, it took Bonsai-1.7B from 3.45 to about 11 tok/s.
- **Simulate before you write a kernel.** Activation sparsity (E58) and a low-rank head (E60) both looked promising on paper; a quality simulation on the Mac killed each in under an hour.

- **On big.LITTLE, leave the little cores out.** On the phone, 2 threads on the two A78s ran Ling-mini-2.0 at 12.0 tok/s; adding A55s only slowed it (7.9 at 4 threads, 6.7 at 8), because every barrier waits for the slowest core (E161).
- **Once the experts are cached, RAM bandwidth is the wall.** Qwen3.6-35B-A3B's non-expert Q4_K weights stream at 13.9 GB/s and take 41 % of a token on the phone; more readers or a cheaper predictor changed nothing (E164).

## Memory and flash

- **Past RAM, flash bandwidth sets the speed.** The kernel's LRU is the worst possible policy for a cyclic scan bigger than the cache: it re-reads everything every token. Keep a fixed resident set and stream the rest yourself.
- **Batching is free when you're flash-bound.** Prompts and speculative verification cost about one weight pass, whatever their size.
- **For a mixture-of-experts model, flash is a cache miss, not a stream.** A ~6 ms miss per expert means the pool's hit rate decides the speed; routing toward cached experts and prefetching what the next router will pick turned 4.4 into 7.2 tok/s on the TV.
- **Past 9B, count resident bytes, not parameters.** The ceiling for a mixture-of-experts model is the non-expert weights that must stay in RAM plus a useful expert pool. Ling-mini-2.0 (16B) keeps 279 MB resident and runs at 5 tok/s on the TV; Qwen3.6-35B-A3B would need about 870 MB before caching a single expert (§20).
- **Address space can be the wall before RAM is.** iOS 12 lets a process map only about 1.7 GB, so stock llama.cpp can't even open the iPad's 2.6 and 5.5 GB files. Mapping only the non-expert part and reading experts with `pread` got around it (§21.8).
- **Most of the RAM was Android's, not the model's.** Force-stopping the launchers and Google services for the duration (nothing persistent) let Bonsai-8B sit fully in RAM: 1.2 → 3.6 tok/s before any kernel work (E80). An animated screensaver and a guard that forked 25 times a second each cost another 10-15 % (E82, E88).

- **Scale the routing bias to the router.** Ling's sigmoid scores live in 0-1 and take a bias of 0.25; Qwen's softmax over 256 experts gives most experts well under 0.05, where 0.02 already costs 7 % perplexity and 0.005 halves the misses for +0.7 % (E162, E163).
- **Flash wants several large reads in flight.** The phone's UFS 2.2 gives 515 MB/s to one reader at 512 KB and ~940 MB/s to four; the opposite of the UFS 4.0 phone in the PowerInfer-2 paper (§22.1).

## Working on the hardware

- **Measure, then verify.** Two streaming bugs gave plausible speeds and wrong numbers; only comparing tensors with the file, and perplexity with a Mac reference, caught them.
- **Guard the device.** After two early kernel panics on the TV, every heavy run goes through a wrapper that kills it below 180 MB of free memory. On the iPad, a pool of 16 experts and wiring weights in RAM each caused a kernel panic and a reboot.
- **Measure noise, and alternate.** Both devices vary by about 10 % between runs, and the iPad reads up to 3× low right after copying a large file. A/B tests alternate repeats back to back.
- **A batch can blow the expert pool.** A perplexity run with 128-token batches made the pager hold every expert the batch touched and restarted Android on the phone; small micro-batches (`-ub 8`) fixed it (E162). Long heavy runs also dropped the phone's USB link twice while the phone kept running, so check the cable before assuming a crash.
