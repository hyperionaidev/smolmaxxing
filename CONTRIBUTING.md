# Contributing

smolmaxxing is about one question: what's the biggest model a small device can really run? The more devices we have numbers for, the better the answer. If you've run a model on an old phone, a tablet, a TV box, a single-board computer or a router, we'd like to hear about it.

## Submit your device

Open a [**Submit your device**](https://github.com/hyperionaidev/smolmaxxing/issues/new?template=submit-device.yml) issue. It asks for four things:

| Field | Example |
|---|---|
| **Device** | Raspberry Pi Zero 2 W (Cortex-A53, 4 cores) |
| **RAM** | 512 MB (about 380 MB free) |
| **Model** | SmolLM2-360M-Instruct, Q4_K_M |
| **Generation speed (tok/s)** | 3.1 |

There are optional fields for the OS, the command you ran, peak memory and a log. Fill in what you have. A rough number with a note is more useful than no number at all.

Stock llama.cpp results are welcome. You don't need our patches.

### Measuring tok/s

Report the **generation** (decode) speed, not the prompt-processing speed. Either of these works:

```sh
# llama-bench: 128 generated tokens, no prompt
llama-bench -m model.gguf -p 0 -n 128 -t <cores>

# or the "eval time" line that llama-cli prints when it exits
llama-cli -m model.gguf -p "Why does the sea have tides?" -n 128 -t <cores>
```

A few things we learned the hard way (more in [docs/lessons.md](docs/lessons.md)):

- **Set threads to the number of physical cores.** More threads than cores can be hundreds of times slower.
- **Don't time right after copying a model.** Flash is still busy and speeds read low. Wait a minute first.
- **Run it three times and report the middle one.** Run-to-run noise of ±10 % is normal on small devices.
- **Watch the memory.** If the device starts swapping or the OS kills other apps, note it. A number that only holds for ten seconds isn't the real number.

### What happens next

We'll check the numbers, ask a question if something looks off, and add the device to the results. Devices that look promising may get their own page, like the [iPad Air](ipad-air/README.md).

## Code and docs

Pull requests are welcome. A few conventions:

- **Measure on the device.** Every number in this repo comes from real hardware. Say which device, model and command produced it.
- **Write down failures too.** An experiment that didn't work goes in [EXPERIMENT.md](EXPERIMENT.md) with its numbers, just like one that did.
- **Keep the front page short.** New results go in `docs/` or the device's own README, not the root README.
- **llama.cpp changes go in `patches/`** (or `ipad-air/patches/` for the iPad), as a patch on a named upstream commit.

## License

By contributing you agree that your contribution is released under the [MIT License](LICENSE).
