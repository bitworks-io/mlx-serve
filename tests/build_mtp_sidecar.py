"""Build an MTP sidecar (mtp/weights.safetensors) from the ORIGINAL
Qwen/Qwen3.6-27B bf16 weights — no third-party intermediate.

Range-downloads only the 15 mtp.* tensors (~840 MB of the 54 GB repo) via
HTTP Range requests against the safetensors shards, quantizes the seven
attention/MLP linears to INT4 group-32 with mlx (standard affine layout,
same geometry the published sidecars use), and keeps fc + norms bf16.

Usage (any python with mlx + numpy + requests/urllib):
  python3 tests/build_mtp_sidecar.py [REPO] [OUT_DIR]
Defaults: Qwen/Qwen3.6-27B -> /tmp/mtp-sidecar-qwen-original/
"""

import json
import struct
import sys
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.6-27B"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/mtp-sidecar-qwen-original")
BASE = f"https://huggingface.co/{REPO}/resolve/main"
QUANT_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                  "o_proj.weight", "gate_proj.weight", "up_proj.weight",
                  "down_proj.weight")


def fetch(url: str, start: int | None = None, end: int | None = None, retries: int = 4) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1}")
            return data
        except Exception as err:  # noqa: BLE001 — retry CDN hiccups
            last_err = err
            print(f"    retry {attempt + 1} after: {err}", flush=True)
    raise last_err


def shard_header(shard: str):
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", fetch(url, 0, 7))[0]
    hdr = json.loads(fetch(url, 8, 8 + n - 1))
    return url, 8 + n, hdr


def main() -> None:
    index = json.loads(fetch(f"{BASE}/model.safetensors.index.json").decode())
    weight_map = index["weight_map"]
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    assert len(mtp_keys) == 15, f"expected 15 mtp tensors, found {len(mtp_keys)}"

    by_shard: dict[str, list[str]] = {}
    for k in mtp_keys:
        by_shard.setdefault(weight_map[k], []).append(k)

    out: dict[str, mx.array] = {}
    for shard, keys in sorted(by_shard.items()):
        url, data_base, hdr = shard_header(shard)
        for key in keys:
            meta = hdr[key]
            assert meta["dtype"] == "BF16", (key, meta["dtype"])
            begin, end = meta["data_offsets"]
            print(f"  {key}  {meta['shape']}  ({(end - begin) / 1e6:.0f} MB)", flush=True)
            raw = fetch(url, data_base + begin, data_base + end - 1)
            u16 = np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])
            w = mx.array(u16).view(mx.bfloat16)
            if key.endswith(QUANT_SUFFIXES):
                wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
                out[key] = wq
                out[key.replace(".weight", ".scales")] = scales.astype(mx.bfloat16)
                out[key.replace(".weight", ".biases")] = biases.astype(mx.bfloat16)
            elif len(meta["shape"]) == 1:
                # Qwen stores RMS-norm weights delta-encoded: the layer
                # computes (1 + w). MLX-converted checkpoints (and our
                # runtime) expect the +1 baked in — verified by diffing the
                # published sidecar against the raw repo tensors (every norm
                # off by exactly 1.0, fc byte-identical).
                out[key] = (w.astype(mx.float32) + 1.0).astype(mx.bfloat16)
            else:
                out[key] = w
            mx.eval(*[v for v in out.values()])

    (OUT / "mtp").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(OUT / "mtp" / "weights.safetensors"), out)
    total = sum(v.nbytes for v in out.values())
    print(f"wrote {len(out)} tensors ({total / 1e6:.0f} MB) to {OUT}/mtp/weights.safetensors")


if __name__ == "__main__":
    main()
