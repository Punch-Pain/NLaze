#!/usr/bin/env python3
"""Phase 5.2 — generate teacher soft targets over the training corpus (Method D).

RESUMABLE BY DESIGN: the corpus is processed in fixed-order shards, each written to a
`.tmp` file and atomically renamed on completion. A completed shard is never recomputed;
a crash loses at most one shard. Restart the same command and it continues.

Teacher scoring (Method D, chosen by the bake-off on 2026-11-20):
  - options are presented as a numbered list in the USER turn,
  - the model is asked to "Answer with the number of the best option",
  - one forward per rotation; the n single-token answer numbers are read at the answer
    position, so `teacher_probs = softmax(number_logprobs / T)`.
  This scored 0.820 overall (typo 0.923, intent 0.852, path 1.0, flag 1.0) vs 0.458 raw
  list-scoring and 0.579 PMI_DC. Danger stays 0.0 because a general model cannot prefer
  NLaze's made-up `confirm_with_user()` sentinel; the deterministic danger table owns
  that slice.

Output rows (one shard file per `--shard-size` corpus rows):
  {"input", "options", "gold", "slice", "teacher_probs", "teacher_argmax", "stable"}

No command string is ever executed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.bakeoff import number_user_prompt, score_numbers


def contiguous_completed(rows_n: int, shard_size: int, out_dir: Path) -> int:
    """Number of leading shards that are complete and valid."""
    n_shards = (rows_n + shard_size - 1) // shard_size
    for s in range(n_shards):
        p = out_dir / f"train_v1-{s:05d}.jsonl"
        if not p.exists():
            return s
        expected = min(shard_size, rows_n - s * shard_size)
        try:
            n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            return s
        if n != expected:
            return s
    return n_shards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/combined/train.jsonl")
    ap.add_argument("--out-dir", default="data/teacher_out")
    ap.add_argument("--model", default="models/teacher/granite-4.2-8b-bf16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--stability-rotations", type=int, default=2,
                    help="extra rotations beyond the stored order (2 => 3 total)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ns = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(line) for line in Path(ns.train).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if ns.limit > 0:
        rows = rows[: ns.limit]
    out_dir = Path(ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_shard = contiguous_completed(len(rows), ns.shard_size, out_dir)
    n_shards = (len(rows) + ns.shard_size - 1) // ns.shard_size
    print(f"[gen] {len(rows)} rows, {n_shards} shards, resuming at shard {start_shard}", flush=True)
    if start_shard >= n_shards:
        print("[gen] all shards complete", flush=True)
        return 0

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head"],
    )
    tok = AutoTokenizer.from_pretrained(ns.model)
    model = AutoModelForCausalLM.from_pretrained(
        ns.model, quantization_config=bnb, device_map=ns.device,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[gen] loaded {ns.model}", flush=True)

    def row_options(r: dict) -> list[str]:
        return list(r["questions"]["correction"]["criteria"].values())

    max_rot = 1 + max(0, ns.stability_rotations)
    t0 = time.perf_counter()
    done = 0
    for s in range(start_shard, n_shards):
        chunk = rows[s * ns.shard_size:(s + 1) * ns.shard_size]
        tmp = out_dir / f"train_v1-{s:05d}.jsonl.tmp"
        final = out_dir / f"train_v1-{s:05d}.jsonl"
        f = open(tmp, "w", encoding="utf-8")
        try:
            for r in chunk:
                text = r["state"]["text"]
                opts = row_options(r)
                n = len(opts)
                gold = r["meta"]["correct"]

                rot_winners: list[int] = []
                base_scores: list[float] | None = None
                for j in range(min(max_rot, n)):
                    order = opts[j:] + opts[:j]
                    prefix = tok.encode(number_user_prompt(tok, text, order),
                                        add_special_tokens=False)
                    scores = score_numbers(model, tok, prefix, n, "cuda")
                    if j == 0:
                        base_scores = scores
                    # map the rotated winner back to the canonical option index
                    rot_winners.append((j + int(np.argmax(scores))) % n)
                assert base_scores is not None

                z = np.array(base_scores, dtype=np.float64) / ns.temperature
                z -= z.max()
                probs = np.exp(z) / np.exp(z).sum()
                stable = len(set(rot_winners)) == 1

                f.write(json.dumps({
                    "input": text,
                    "options": opts,
                    "gold": gold,
                    "slice": r["meta"].get("type", "unknown"),
                    "teacher_probs": [round(float(p), 6) for p in probs],
                    "teacher_argmax": int(np.argmax(probs)),
                    "stable": stable,
                }) + "\n")
                done += 1
            f.flush()
            os.fsync(f.fileno())
        finally:
            f.close()
        tmp.replace(final)
        el = time.perf_counter() - t0
        print(f"[gen] shard {s + 1}/{n_shards} done ({len(chunk)} rows), "
              f"{done} rows this run, {el:.0f}s ({done / max(el, 1e-9):.2f} rows/s)", flush=True)

    print(f"[gen] all shards complete in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
