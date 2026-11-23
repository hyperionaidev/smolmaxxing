# 📺 Running it on the TV box

How to build the patched llama.cpp, get models onto the box and open the chat page. For the results, see [tv-box.md](tv-box.md).

[← back to the main README](../README.md)

## What you need

- The TV box, or another ARMv8.2 Android device, with network ADB turned on
- On your computer: `adb`, `zig` (for static aarch64 musl builds), `cmake` + `ninja`, `python3` and `uv` (for numpy)

Everything on the box lives in `/data/local/tmp/llm` and runs as the adb `shell` user. The only system setting touched is stay-awake in headless mode (`svc power stayon`), which `tv-llm-mode.sh off` turns back off. `adb shell rm -rf /data/local/tmp/llm` removes it all.

## 1. Build llama.cpp with the patch

```sh
export ADB_SERIAL=<tv-ip>:5555          # every script talks to the box through this
adb connect $ADB_SERIAL

git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
cd third_party/llama.cpp && git checkout 8642541 && git apply ../../patches/llama.cpp-v0.8.0-tvbox.patch
cmake -S . -B build-tv -G Ninja -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$PWD/../../zigcc/cc -DCMAKE_CXX_COMPILER=$PWD/../../zigcc/c++ -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXE_LINKER_FLAGS=-static -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF
ninja -C build-tv llama-server llama-cli llama-bench llama-perplexity llama-quantize
cd ../..
adb -s $ADB_SERIAL shell mkdir -p /data/local/tmp/llm/patched25 /data/local/tmp/llm/patched
adb -s $ADB_SERIAL push third_party/llama.cpp/build-tv/bin/llama-{server,cli,bench,perplexity} /data/local/tmp/llm/patched25/
adb -s $ADB_SERIAL push third_party/llama.cpp/build-tv/bin/llama-quantize /data/local/tmp/llm/patched/   # used by tools/tv-quant.sh
zig cc -target aarch64-linux-musl -O2 -static -o gguf_recut tools/gguf-recut/gguf_recut.c && adb -s $ADB_SERIAL push gguf_recut /data/local/tmp/llm/
```

## 2. Get a model

Either download a ready-made file (see [models.md](models.md#tv-box)) and `adb push` it to `/data/local/tmp/llm/`, or rebuild it from its Hugging Face sources straight onto the box:

```sh
tools/rebuild-models.sh 2b          # also: 08 4b b8 tb8 q9 mb lm caches
```

## 3. Chat

```sh
tv-llm-ui/start.sh          # the 2B with fp16 state; open http://<tv-ip>:8090 from any device on the network
tv-llm-ui/start.sh stop
```

## Offline: chat on the TV itself

The **SmolChat** app runs the chat on the box alone: it starts the model server at every boot, and opening SmolChat from the TV home screen shows the chat page. No Mac, adb or internet is needed once it's installed.

```sh
tv-llm-ui/android/build.sh       # once: JDK 17 + Android build-tools 34 and platform 34 in third_party/android-sdk (see the script)
tv-llm-ui/android/install.sh     # installs the app and copies Qwen3.5-0.8B into its storage (from the box, or MODEL_FILE=<file>)
```

How it works:

- **The model:** Qwen3.5-0.8B (`q08-q4k-h65p8.gguf`). The 2B doesn't fit next to the chat page: the memory guard killed it at about 110 MB free, even in headless mode at 640×360 (E7 found the same with the TV browser).
- **Speed:** about 15-16 tok/s with the chat on screen, and about 520 MB stays free.
- **The binaries:** the app carries the patched `llama-server` and `tv-watch` as `lib*.so`, because an app may only run files from its native library folder.
  - This box runs a 32-bit Android on a 64-bit kernel, so the static aarch64 binaries go in the app's `armeabi-v7a` folder. The kernel runs them fine.
- **The service:** a foreground service runs the server under the memory guard and restarts it if it exits.
- **Settings:** the same as `start.sh`: `patched25`, 3 threads, fp16 DeltaNet state.
- **Not used:** headless mode. An app can't stop other apps, and the 0.8B fits without it.
- **To remove it:** `adb uninstall com.smolmaxxing.tvchat`.

Without the app, `chat.sh` does the same from an adb shell. It also turns on headless mode and opens the page in the TV's own browser, at 11-12 tok/s at 1080p or about 16.7 with `LEVEL=2` (640×360). A reboot stops it.

```sh
tv-llm-ui/install-offline.sh                              # once, from the Mac
adb -s $ADB_SERIAL shell sh /data/local/tmp/llm/chat.sh   # start; "chat.sh stop" stops it
```

For the 2B, run `BROWSER=0 MODEL=w-q4k-h65p8.gguf sh chat.sh` and open `http://<tv-ip>:8090/` from a phone on the same Wi-Fi; a router with no internet works.

If `adb connect` fails with "No route to host" while `ping` and `nc -z <tv-ip> 5555` work, macOS Local Network privacy is blocking adb:

- Allow your terminal under System Settings → Privacy & Security → Local Network.
- Or run `tools/adb-relay.sh <tv-ip>` in a Terminal window, then `adb connect 127.0.0.1:15555`.

## Other models

`start.sh` takes the model file in `MODEL`. For the mixture-of-experts models it turns on headless mode, the expert pager, routing bias, prefetch and the repeat penalty; `POOL`, `BIAS` and `EXPERTS` override them.

```sh
# Moon-Beam-8B-A1B MoE, ~7.2 tok/s (first: `tools/rebuild-models.sh mb`, needs a native Mac build and ~9 GB on the Mac while it runs)
MODEL=mb-q2-h60.gguf tv-llm-ui/start.sh

# Ling-mini-2.0 16B MoE, ~5.1 tok/s (first: `tools/rebuild-models.sh lm`)
MODEL=lm2-q2-h60.gguf tv-llm-ui/start.sh

# Bonsai-8B 1-bit fully resident, ~5.2 tok/s (first: `tools/rebuild-models.sh b8`; the repack cache is written on first start)
MODEL=b8-q1-hml.gguf tv-llm-ui/start.sh

# Qwen3.5-0.8B, ~16 tok/s (first: `tools/rebuild-models.sh 08`, needs a native Mac build of llama-quantize)
MODEL=q08-q4k-h65p8.gguf tv-llm-ui/start.sh
```

Headless mode stays on after the server stops. To get the launcher and screensaver back, run this on the TV:

```sh
sh /data/local/tmp/llm/tv-llm-mode.sh off
```

<details>
<summary>The streaming setups (Bonsai-8B without headless mode, Qwen3.5-9B)</summary>

These numbers were measured with the build of that time (`patched16`); the current build has the same code plus §15-17. `GGML_CHUNK_GUIDED=0` in `SENV` restores the old matmul scheduling. There's no fp16 state with MTP.

```sh
# Bonsai-8B 1-bit with streaming and no headless mode, ~1.2 tok/s (first run: `tools/rebuild-models.sh b8 caches`)
MODEL=b8-q1-hml.gguf BIN=patched25 EXTRA="-ctk q8_0 -ctv q8_0 -fa on -cram 0" \
  SENV="GGML_REPACK_CACHE=/data/local/tmp/llm/cache/b8c GGML_REPACK_RESIDENT_MB=800 GGML_STREAM_MODE=cache GGML_STREAM_WINDOW_MB=64" tv-llm-ui/start.sh

# Qwen3.5-9B with MTP, ~0.27 tok/s
MODEL=q9-iq2-h65-mtp.gguf BIN=patched25 EXTRA="-nr -ctxcp 0 -cram 0 -ub 16 -b 64 --spec-type draft-mtp --spec-draft-n-max 1" \
  SENV="LLAMA_STREAM_RESIDENT_MB=600 LLAMA_STREAM_PIN=output.,blk.32. GGML_STREAM_WINDOW_MB=64" tv-llm-ui/start.sh
```

</details>
