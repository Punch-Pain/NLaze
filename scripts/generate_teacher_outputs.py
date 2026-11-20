#!/usr/bin/env python3
"""Phase 5.2 — generate teacher soft targets over the training corpus (frozen G1 format).

RESUMABLE BY DESIGN: the corpus is processed in fixed-order shards, each written to a
`.tmp` file and atomically renamed on completion. A completed shard is never recomputed;
a crash loses at most one shard. Restart the same command and it continues.

For each training item:
  - chat prefix + numbered 7-option list (stored order),
  - one batched forward over the stored order + `--stability-rotations` extra rotations,
  - PMI_DC = score - null("Answer:", option),
  - probs = softmax(PMI_DC / T); also stores raw and length-normalised variants,
  - stable = all rotations agree on the argmax.

Output rows (one shard file per `--shard-size` corpus rows):
  {"input", "options", "gold", "slice", "teacher_probs", "teacher_argmax",
   "raw_probs", "pmi_len_probs", "stable"}

No command string is ever executed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.bakeoff import (
    chat_prefix,
    encode_prompt,
    score_null_batch,
    score_rotations_batch,
)


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
    ap.add_argument("--stability-rotations", type=int, default=5,
                    help="extra rotations beyond the stored order (5 => 6 total, matching §3.5)")
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

    remaining = rows[start_shard * ns.shard_size:]
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
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"[gen] loaded {ns.model}", flush=True)

    def row_options(r: dict) -> list[str]:
        return list(r["questions"]["correction"]["criteria"].values())

    distinct = sorted({o for r in remaining for o in row_options(r)})
    tn = time.perf_counter()
    null_scores = score_null_batch(tok, model, distinct, ns.device)
    torch.cuda.empty_cache()
    print(f"[gen] null term: {len(distinct)} options in {time.perf_counter() - tn:.1f}s", flush=True)

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
                prefix = chat_prefix(tok, text)

                orders: list[list[str]] = []
                seqs: list[list[int]] = []
                spans_list: list[list[tuple[int, int]]] = []
                for j in range(min(max_rot, n)):
                    order = opts[j:] + opts[:j]
                    ids, spans = encode_prompt(tok, prefix, order)
                    orders.append(order)
                    seqs.append(ids)
                    spans_list.append(spans)
                rot_scores = score_rotations_batch(model, seqs, spans_list, ns.device, pad_id=pad_id)

                def dist(row_scores: list[float], order: list[str]) -> np.ndarray:
                    z = np.array(
                        [row_scores[i] - null_scores[order[i]] for i in range(n)],
                        dtype=np.float64,
                    ) / ns.temperature
                    z -= z.max()
                    return np.exp(z) / np.exp(z).sum()

                probs = dist(rot_scores[0], orders[0])
                raw = np.array(rot_scores[0], dtype=np.float64)
                raw -= raw.max()
                raw_probs = np.exp(raw) / np.exp(raw).sum()
                lens = np.array(
                    [max(1, len(tok.encode(orders[0][i], add_special_tokens=False))) for i in range(n)],
                    dtype=np.float64,
                )
                zl = np.array(
                    [rot_scores[0][i] - null_scores[orders[0][i]] for i in range(n)],
                    dtype=np.float64,
                ) / (lens * ns.temperature)
                zl -= zl.max()
                pmi_len_probs = np.exp(zl) / np.exp(zl).sum()

                winners = []
                for j, order in enumerate(orders):
                    sj = np.array(
                        [rot_scores[j][i] - null_scores[order[i]] for i in range(n)],
                        dtype=np.float64,
                    )
                    winners.append(int(np.argmax(sj)))
                stable = len(set(winners)) == 1

                f.write(json.dumps({
                    "input": text,
                    "options": opts,
                    "gold": gold,
                    "slice": r["meta"].get("type", "unknown"),
                    "teacher_probs": [round(float(p), 6) for p in probs],
                    "teacher_argmax": int(np.argmax(probs)),
                    "raw_probs": [round(float(p), 6) for p in raw_probs],
                    "pmi_len_probs": [round(float(p), 6) for p in pmi_len_probs],
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
