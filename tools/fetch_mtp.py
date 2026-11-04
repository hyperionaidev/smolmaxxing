#!/usr/bin/env python3
"""Fetch only the MTP (multi-token prediction) tensors of a Qwen3.5 checkpoint from Hugging Face, via HTTP
range requests on the safetensors file, and convert them to GGUF-ready tensors (llama.cpp naming, layer n_layer).

Mirrors llama.cpp conversion/qwen.py (_QwenMtpMixin + Qwen3NextModel.modify_tensors):
  mtp.layers.0.<x>            -> blk.<n_layer>.<x>   (standard tensor map)
  mtp.fc                      -> blk.<n_layer>.nextn.eh_proj
  mtp.pre_fc_norm_embedding   -> blk.<n_layer>.nextn.enorm
  mtp.pre_fc_norm_hidden      -> blk.<n_layer>.nextn.hnorm
  mtp.norm                    -> blk.<n_layer>.nextn.shared_head_norm
  every *norm.weight gets +1 (Qwen stores zero-centered RMSNorm gains)
Matrices are quantized to Q8_0 (ggml reference rounding), or kept as raw BF16 with --bf16 (so llama-quantize
can quantize them together with the rest of the model); norms stay F32.

usage: python3 -I fetch_mtp.py <hf_repo> <n_layer> <outdir> [--bf16 | --q4]   (--q4: matrices as Q4_0)
writes <outdir>/manifest.json + <outdir>/<name>.bin (manifest paths are relative to <outdir>)
"""
import json, os, struct, sys, urllib.request
import numpy as np

UA = "tvbox-mtp-fetch/0.1"

def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": UA})
    data = urllib.request.urlopen(req, timeout=120).read()
    assert len(data) == b - a, f"range {a}-{b} returned {len(data)} bytes"
    return data

def q8_0(x):
    """x: float32 [rows, cols], cols % 32 == 0 -> ggml block_q8_0 bytes (fp16 d + 32 x int8), row-major."""
    rows, cols = x.shape
    blk = x.reshape(rows, cols // 32, 32)
    amax = np.abs(blk).max(axis=2, keepdims=True)
    d = amax / 127.0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.round(blk / d), 0).astype(np.int8)
    out = np.empty((rows, cols // 32), dtype=[("d", "<f2"), ("qs", "i1", 32)])
    out["d"] = d[..., 0].astype(np.float16)
    out["qs"] = q
    return out.tobytes()

def q4_0(x):
    """x: float32 [rows, cols] -> ggml block_q4_0 bytes (fp16 d + 16 bytes of nibbles), ggml reference rounding."""
    rows, cols = x.shape
    blk = x.reshape(rows, cols // 32, 32)
    idx = np.abs(blk).argmax(axis=2)[..., None]
    mx = np.take_along_axis(blk, idx, axis=2)                     # signed value of largest magnitude
    d = mx / -8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        idv = np.where(d != 0, 1.0 / d, 0.0)
    q = np.clip(np.trunc(blk * idv + 8.5), 0, 15).astype(np.uint8)
    out = np.empty((rows, cols // 32), dtype=[("d", "<f2"), ("qs", "u1", 16)])
    out["d"] = d[..., 0].astype(np.float16)
    out["qs"] = q[..., :16] | (q[..., 16:] << 4)
    return out.tobytes()

NAME_MAP = {
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
}
TOP_MAP = {"fc.weight": "nextn.eh_proj.weight", "pre_fc_norm_embedding.weight": "nextn.enorm.weight",
           "pre_fc_norm_hidden.weight": "nextn.hnorm.weight", "norm.weight": "nextn.shared_head_norm.weight"}

def main():
    repo, n_layer, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    bf16 = "--bf16" in sys.argv[4:]
    q4 = "--q4" in sys.argv[4:]
    os.makedirs(outdir, exist_ok=True)
    idx = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json", headers={"User-Agent": UA})).read())
    files = sorted({f for k, f in idx["weight_map"].items() if k.startswith("mtp.")})
    manifest = []
    for fn in files:
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        n = struct.unpack("<Q", rng(url, 0, 8))[0]
        header = json.loads(rng(url, 8, 8 + n))
        base = 8 + n
        for name, info in sorted(header.items()):
            if not name.startswith("mtp."):
                continue
            a, b = info["data_offsets"]
            raw = rng(url, base + a, base + b)
            assert info["dtype"] == "BF16", info["dtype"]
            x = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
            sub = name[len("mtp."):]
            if sub.startswith("layers.0."):
                gname = f"blk.{n_layer}." + NAME_MAP[sub[len("layers.0."):]]
            else:
                gname = f"blk.{n_layer}." + TOP_MAP[sub]
            if gname.endswith("norm.weight"):
                x = x + 1.0
            if x.ndim == 1:
                data, gtype, dims = x.astype("<f4").tobytes(), 0, [x.shape[0]]
            elif bf16:
                data, gtype, dims = raw, 30, [x.shape[1], x.shape[0]]
            elif q4:
                data, gtype, dims = q4_0(np.ascontiguousarray(x, dtype=np.float32)), 2, [x.shape[1], x.shape[0]]
            else:
                data, gtype, dims = q8_0(np.ascontiguousarray(x, dtype=np.float32)), 8, [x.shape[1], x.shape[0]]
            out = os.path.join(outdir, gname + ".bin")
            open(out, "wb").write(data)
            manifest.append({"name": gname, "dims": dims, "type": gtype, "file": gname + ".bin", "bytes": len(data)})
            tn = {0: "F32 ", 2: "Q4_0", 8: "Q8_0", 30: "BF16"}[gtype]
            print(f"{name:45s} {str(info['shape']):14s} -> {gname:38s} {tn} {len(data)/1e6:7.2f} MB")
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    print(f"total {sum(m['bytes'] for m in manifest)/1e6:.1f} MB in {len(manifest)} tensors")

if __name__ == "__main__":
    main()
