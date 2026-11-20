#!/usr/bin/env python3
"""Convert the NVFP4 (compressed-tensors) teacher checkpoint to plain bf16 safetensors,
one transformer layer at a time so RAM stays bounded (~0.5 GiB per layer).

The NVFP4 checkpoint cannot be run on an 8 GiB GPU: `compressed-tensors` decompresses
all 8.79 B parameters to bf16 on the first forward pass (~17.6 GiB). This script produces
a bf16 checkpoint that can then be loaded with bitsandbytes NF4 (research-teacher-selection
§10.5.2 mandatory arm config), which fits in ~4.4 GiB VRAM.

No model weights are executed; this is a pure tensor format conversion.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.quantization import QuantizationScheme

SRC = Path("models/teacher/granite-4.2-8b-nvfp4")
DST = Path("models/teacher/granite-4.2-8b-bf16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--only-layer", type=int, default=None, help="convert only this layer (debug)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    config = json.loads((src / "config.json").read_text())
    group0 = config["quantization_config"]["config_groups"]["group_0"]
    scheme = QuantizationScheme(**group0)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]

    # Group tensor names by owning shard (per transformer layer, or "other").
    def shard_of(name: str) -> str:
        m = re.search(r"model\.layers\.(\d+)\.", name)
        return f"layer-{int(m.group(1)):02d}" if m else "other"

    shards: dict[str, list[str]] = {}
    for name in weight_map:
        shards.setdefault(shard_of(name), []).append(name)

    new_weight_map: dict[str, str] = {}
    total_size = 0

    for shard, names in sorted(shards.items()):
        if args.only_layer is not None and shard != f"layer-{args.only_layer:02d}":
            continue
        # Quantized prefixes in this shard
        prefixes = sorted({n[: -len(".weight_packed")] for n in names if n.endswith(".weight_packed")})
        prefix_set = set(prefixes)
        out: dict[str, torch.Tensor] = {}
        # Files needed for this shard
        files = sorted({weight_map[n] for n in names})
        for fname in files:
            with safe_open(src / fname, framework="pt", device="cpu") as f:
                for prefix in prefixes:
                    if weight_map[f"{prefix}.weight_packed"] != fname:
                        continue
                    sd = {
                        "weight_packed": f.get_tensor(f"{prefix}.weight_packed"),
                        "weight_scale": f.get_tensor(f"{prefix}.weight_scale"),
                        "weight_global_scale": f.get_tensor(f"{prefix}.weight_global_scale"),
                    }
                    dec = NVFP4PackedCompressor.decompress(sd, scheme)
                    w = dec["weight"].to(torch.bfloat16).contiguous()
                    out[f"{prefix}.weight"] = w
                for n in names:
                    if weight_map[n] != fname:
                        continue
                    if any(n.startswith(p) for p in prefix_set):
                        continue
                    out[n] = f.get_tensor(n).to(torch.bfloat16).contiguous()

        out_fname = f"model-{shard}.safetensors"
        save_file(out, dst / out_fname)
        for n, t in out.items():
            new_weight_map[n] = out_fname
            total_size += t.numel() * t.element_size()
        print(f"[{shard}] wrote {out_fname}: {len(out)} tensors, "
              f"{sum(t.numel()*t.element_size() for t in out.values())/2**20:.1f} MiB", flush=True)
        del out

    # Write the new index and a de-quantized config
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }, indent=2) + "\n")

    cfg = dict(config)
    cfg.pop("quantization_config", None)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # Copy tokenizer / generation config / chat template
    for extra in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "generation_config.json", "chat_template.jinja"):
        p = src / extra
        if p.exists():
            shutil.copy2(p, dst / extra)

    print(f"Done. Total bf16 size: {total_size/2**30:.2f} GiB across {len(set(new_weight_map.values()))} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
