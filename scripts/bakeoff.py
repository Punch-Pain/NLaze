#!/usr/bin/env python3
"""Phase 1.3 & Phase 5.1 — Teacher Bake-Off Harness (research-teacher-selection §10.5).

SAFETY CONTRACT:
  Every option string in this script is INERT DATA. No option or model output is ever
  executed, shelled out, or eval'd.

Subcommands:
  draw   Build the frozen 2,000-item stratified held-out draw at `data/bakeoff/items_v1.jsonl`
         (asserting zero overlap with `data/combined/train.jsonl`) and record its SHA-256 in
         `data/bakeoff/MANIFEST.sha256`.
  score  Score a Laya checkpoint (`--kind laya`) or a HuggingFace causal LM (`--kind causal`)
         using rotation-averaged logprobs + PMI_DC null normalization (§10.5.2) and report all
         four pre-registered metrics (§10.5.3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np

from nlaze.candidates import INSTRUCTIONS, QID

ROOT = Path(__file__).resolve().parents[1]
BAKEOFF_DIR = ROOT / "data" / "bakeoff"
ITEMS_PATH = BAKEOFF_DIR / "items_v1.jsonl"
MANIFEST_PATH = BAKEOFF_DIR / "MANIFEST.sha256"

SEED = 20261013
# Stratified quotas summing to 2,000 (§10.5.1):
SLICE_QUOTAS = {
    "intent": 1133,
    "typo": 431,
    "path": 296,
    "flag": 60,
    "danger": 80,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_draw(force: bool = False) -> dict:
    if ITEMS_PATH.exists() and MANIFEST_PATH.exists() and not force:
        digest = sha256_file(ITEMS_PATH)
        recorded = MANIFEST_PATH.read_text(encoding="utf-8").split()[0]
        assert digest == recorded, f"SHA-256 mismatch on {ITEMS_PATH}: {digest} != {recorded}"
        return {"status": "already_frozen", "path": str(ITEMS_PATH), "sha256": digest}

    train_inputs: set[str] = set()
    train_path = ROOT / "data" / "combined" / "train.jsonl"
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_inputs.add(json.loads(line)["state"]["text"].strip())

    pools: dict[str, list[dict]] = {k: [] for k in SLICE_QUOTAS}
    seen_inputs: set[str] = set()

    # 1. Load held-out val.jsonl and test.jsonl
    for split in ("val", "test"):
        p = ROOT / "data" / "combined" / f"{split}.jsonl"
        with open(p, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                text = r["state"]["text"].strip()
                if text in train_inputs or text in seen_inputs:
                    continue
                crit = r["questions"][QID]["criteria"]
                opts = list(crit.values())
                gold = r["meta"]["correct"]
                if gold not in opts or len(set(opts)) != len(opts):
                    continue
                sl = r["meta"]["type"]
                if sl not in pools:
                    continue
                seen_inputs.add(text)
                pools[sl].append({
                    "source": f"combined/{split}:{idx}",
                    "slice": sl,
                    "text": text,
                    "options": opts,
                    "gold": gold,
                })

    # 2. Load held-out rows from data/flag_v5/pairs.jsonl (§10.5.1 adapter)
    flag_path = ROOT / "data" / "flag_v5" / "pairs.jsonl"
    if flag_path.exists():
        with open(flag_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                text = r["input"].strip()
                if text in train_inputs or text in seen_inputs:
                    continue
                gold = r["correct"]
                distractors = [d for d in r["distractors"] if d != gold]
                # Deduplicate while preserving order
                uniq_d = list(dict.fromkeys(distractors))
                opts = [gold] + uniq_d[:6]
                if len(opts) < 5:
                    continue
                seen_inputs.add(text)
                pools["flag"].append({
                    "source": f"flag_v5:{idx}",
                    "slice": "flag",
                    "text": text,
                    "options": opts,
                    "gold": gold,
                })

    # 3. Load held-out rows from data/safety_v5/pairs.jsonl into danger pool
    safety_path = ROOT / "data" / "safety_v5" / "pairs.jsonl"
    if safety_path.exists():
        with open(safety_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                text = r["input"].strip()
                if text in train_inputs or text in seen_inputs:
                    continue
                gold = r["correct"]
                if gold == "confirm_with_user":
                    gold = "confirm_with_user()"
                distractors = [
                    d for d in r["distractors"]
                    if d != gold and not (gold == "confirm_with_user()" and d == "confirm_with_user")
                ]
                uniq_d = list(dict.fromkeys(distractors))
                opts = [gold] + uniq_d[:6]
                if len(opts) < 5:
                    continue
                seen_inputs.add(text)
                pools["danger"].append({
                    "source": f"safety_v5:{idx}",
                    "slice": "danger",
                    "text": text,
                    "options": opts,
                    "gold": gold,
                })

    rng = random.Random(SEED)
    selected: list[dict] = []
    for sl, quota in SLICE_QUOTAS.items():
        pool = pools[sl]
        assert len(pool) >= quota, f"Slice {sl} pool has {len(pool)} < {quota}"
        drawn = rng.sample(pool, quota)
        for item in drawn:
            opts = list(item["options"])
            rng.shuffle(opts)
            gold = item["gold"]
            selected.append({
                "slice": sl,
                "source": item["source"],
                "text": item["text"],
                "options": opts,
                "gold": gold,
                "gold_idx": opts.index(gold),
            })

    rng.shuffle(selected)
    BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
    with open(ITEMS_PATH, "w", encoding="utf-8") as out:
        for idx, it in enumerate(selected):
            opts = it["options"]
            row = {
                "id": f"bakeoff_{idx:04d}",
                "slice": it["slice"],
                "source": it["source"],
                "text": it["text"],
                "n_options": len(opts),
                "options": opts,
                "gold": it["gold"],
                "gold_idx": it["gold_idx"],
                "state": {"text": it["text"]},
                "questions": {
                    QID: {
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "criteria": {f"option_{i}": c for i, c in enumerate(opts)},
                    }
                },
            }
            out.write(json.dumps(row) + "\n")

    digest = sha256_file(ITEMS_PATH)
    MANIFEST_PATH.write_text(f"{digest}  items_v1.jsonl\n", encoding="utf-8")
    return {
        "status": "created",
        "path": str(ITEMS_PATH),
        "n_items": len(selected),
        "slice_counts": {sl: sum(1 for x in selected if x["slice"] == sl) for sl in SLICE_QUOTAS},
        "sha256": digest,
    }


def compute_metrics(items: list[dict], per_item_rot_scores: list[np.ndarray]) -> dict:
    """Compute the four §10.5.3 metrics from per-item [n_rot, n_opt] score arrays
    (where column j corresponds to `item["options"][j]` across every rotation r)."""
    by_slice: dict[str, dict[str, list[float]]] = {}
    gaps_correct: list[float] = []
    gaps_wrong: list[float] = []
    pb_ranges: list[float] = []
    flips = 0

    for it, rot_mat in zip(items, per_item_rot_scores):
        # rot_mat shape: [n_opt, n_opt] where row r is rotation r, col j is canonical option j
        sl = it["slice"]
        gold_j = it["gold_idx"]
        n_opt = len(it["options"])

        # Rotation winners
        rot_winners = [int(np.argmax(rot_mat[r])) for r in range(rot_mat.shape[0])]
        if len(set(rot_winners)) > 1:
            flips += 1

        # Position-bias range per option across rotations
        pb_ranges.append(float(np.mean(np.max(rot_mat, axis=0) - np.min(rot_mat, axis=0))))

        # Rotation-averaged probability distribution
        avg_scores = np.mean(rot_mat, axis=0)
        # Normalize to probabilities if not already
        if np.all(avg_scores >= 0.0) and abs(float(np.sum(avg_scores)) - 1.0) < 1e-3:
            probs = avg_scores
        else:
            shifted = avg_scores - np.max(avg_scores)
            ex = np.exp(shifted)
            probs = ex / np.sum(ex)

        order_desc = np.argsort(-probs)
        top1_ok = float(int(order_desc[0]) == gold_j)
        top3_ok = float(gold_j in [int(x) for x in order_desc[: min(3, n_opt)]])

        p_gold = float(probs[gold_j])
        p_best_rival = float(np.max([probs[j] for j in range(n_opt) if j != gold_j]))
        gap = p_gold - p_best_rival
        if top1_ok:
            gaps_correct.append(gap)
        else:
            gaps_wrong.append(gap)

        sdict = by_slice.setdefault(sl, {"top1": [], "top3": [], "gap": []})
        sdict["top1"].append(top1_ok)
        sdict["top3"].append(top3_ok)
        sdict["gap"].append(gap)

    per_slice = {}
    for sl, d in by_slice.items():
        per_slice[sl] = {
            "n": len(d["top1"]),
            "top1": round(float(np.mean(d["top1"])), 4),
            "top3": round(float(np.mean(d["top3"])), 4),
            "mean_gap": round(float(np.mean(d["gap"])), 4),
        }

    # §10.5.3 metric 1: equal-weighted macro average across slices
    macro_top1 = round(float(np.mean([v["top1"] for v in per_slice.values()])), 4)
    macro_top3 = round(float(np.mean([v["top3"] for v in per_slice.values()])), 4)
    micro_top1 = round(
        float(sum(sum(d["top1"]) for d in by_slice.values()) / max(1, len(items))), 4
    )

    return {
        "n_items": len(items),
        "macro_top1": macro_top1,
        "macro_top3": macro_top3,
        "micro_top1": micro_top1,
        "per_slice": per_slice,
        "confidence_gap": {
            "overall": round(float(np.mean(gaps_correct + gaps_wrong)), 4),
            "when_correct": round(float(np.mean(gaps_correct)) if gaps_correct else 0.0, 4),
            "when_wrong": round(float(np.mean(gaps_wrong)) if gaps_wrong else 0.0, 4),
        },
        "position_bias": {
            "mean_range": round(float(np.mean(pb_ranges)), 4),
            "flip_rate_rot": round(float(flips / max(1, len(items))), 4),
        },
    }


def score_laya(model_dir: str, items: list[dict]) -> tuple[dict, float]:
    import laya

    t0 = time.perf_counter()
    agent = laya.Agent(str(model_dir), device="cuda")
    per_item_mats: list[np.ndarray] = []

    for it in items:
        opts = it["options"]
        n = len(opts)
        questions = {}
        for r in range(n):
            rot = opts[r:] + opts[:r]
            questions[f"r{r}"] = {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {f"option_{i}": c for i, c in enumerate(rot)},
            }
        ans = agent.predict({"text": it["text"]}, questions)["answers"]
        mat = np.zeros((n, n), dtype=np.float64)
        for r in range(n):
            probs = ans[f"r{r}"]["probabilities"]
            for i in range(n):
                orig_j = (r + i) % n
                mat[r, orig_j] = float(probs[f"option_{i}"])
        per_item_mats.append(mat)

    wall_s = time.perf_counter() - t0
    metrics = compute_metrics(items, per_item_mats)
    metrics["wall_s"] = round(wall_s, 2)
    metrics["ms_per_item"] = round(wall_s * 1000.0 / max(1, len(items)), 2)
    return metrics, wall_s


# ---------------------------------------------------------------------------
# Causal-LM scoring (research-teacher-selection §10.5.2)
# ---------------------------------------------------------------------------

def chat_prefix(tok, text: str) -> str:
    """chat_template(system=INSTRUCTIONS, user=text) + assistant prefix.

    `enable_thinking=False` is passed where the template supports it (Granite): a
    non-reasoning distribution is the theoretically best setting for PMI_DC over
    bare command strings (research-bracket-beaters §Rank 3).
    """
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": text},
    ]
    try:
        return tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except Exception:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def encode_prompt(tok, prefix: str, order: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `prefix + numbered option list` and locate each option's token span.

    Uses ISOLATED option tokenisation everywhere (research-teacher-selection §7.2 tokenisation
    hygiene): the option's in-context span is replaced by `tok.encode(option)`, so the scored
    tokens are exactly the option's own ids. The PMI_DC null term uses the same convention.
    """
    full = prefix + "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(order))
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    raw_spans: list[tuple[int, int]] = []
    pos = len(prefix)
    for i, opt in enumerate(order):
        label = f"{i + 1}. "
        start = pos + len(label)
        end = start + len(opt)
        idx = [j for j, (s, e) in enumerate(offsets) if s < end and e > start]
        if not idx:
            raise ValueError(f"could not locate option {i} ({opt!r}) in prompt")
        raw_spans.append((min(idx), max(idx) + 1 - min(idx)))
        pos = end + 1  # skip the newline separator

    # Build the final ids by splicing the isolated option tokenisation in place of each
    # in-context span. Track spans as we go so length changes cannot shift later options.
    out_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    prev_end = 0
    n_replaced = 0
    for i, (t, n) in enumerate(raw_spans):
        iso = tok.encode(order[i], add_special_tokens=False)
        if len(iso) != n:
            n_replaced += 1
        out_ids.extend(ids[prev_end:t])
        spans.append((len(out_ids), len(iso)))
        out_ids.extend(iso)
        prev_end = t + n
    out_ids.extend(ids[prev_end:])
    return out_ids, spans


def _logsumexp_rows(x, chunk: int = 16384):
    """Row-wise logsumexp without a full fp32 copy of `x` (chunked over the vocab)."""
    import torch

    parts = [
        torch.logsumexp(x[:, i:i + chunk].float(), dim=-1)
        for i in range(0, x.size(1), chunk)
    ]
    if len(parts) == 1:
        return parts[0]
    return torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)


def score_spans(model, input_ids: list[int], spans: list[tuple[int, int]], device: str) -> list[float]:
    """Sum logP(option tokens | prefix) for each span, slicing hidden states before lm_head.

    For option tokens at positions t..t+n-1, the predicting logits are at t-1..t+n-2.
    """
    import torch

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.model(input_ids=ids, use_cache=False)
    h = out.last_hidden_state[0]  # [seq, d]

    pred_pos: list[int] = []
    opt_slice: list[tuple[int, int]] = []
    for (t, n) in spans:
        opt_slice.append((len(pred_pos), len(pred_pos) + n))
        pred_pos.extend(range(t - 1, t - 1 + n))
    pos_t = torch.tensor(pred_pos, dtype=torch.long, device=device)
    logits = model.lm_head(h[pos_t])  # [total, vocab]
    tok_ids = torch.cat([ids[0, t:t + n] for (t, n) in spans])
    target = logits.gather(1, tok_ids.unsqueeze(1)).squeeze(1).float()
    logZ = _logsumexp_rows(logits)
    lp = target - logZ
    del logits

    scores: list[float] = []
    for (s, e) in opt_slice:
        scores.append(float(lp[s:e].sum().item()))
    return scores


def score_null(tok, model, option: str, device: str) -> float:
    """PMI_DC null term: logP(option | "Answer:") — one forward, no rotations.

    Same isolated-option convention as `encode_prompt`.
    """
    prefix_ids = tok.encode("Answer:", add_special_tokens=False)
    opt_ids = tok.encode(option, add_special_tokens=False)
    ids = prefix_ids + opt_ids
    span = (len(prefix_ids), len(opt_ids))
    return score_spans(model, ids, [span], device)[0]


def score_rotations_batch(
    model,
    seqs: list[list[int]],
    spans_list: list[list[tuple[int, int]]],
    device: str,
    pad_id: int,
) -> list[list[float]]:
    """Score several rotation prompts for one item in a single batched forward.

    Right-pads to the batch max length; causal attention masks the padding, so the real
    option-token positions are unaffected.
    """
    import torch

    maxlen = max(len(s) for s in seqs)
    ids = [s + [pad_id] * (maxlen - len(s)) for s in seqs]
    attn = [[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs]
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
    attn_t = torch.tensor(attn, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.model(input_ids=ids_t, attention_mask=attn_t, use_cache=False)
    h = out.last_hidden_state  # [B, maxlen, d]

    b_idx: list[int] = []
    pos_list: list[int] = []
    tok_list: list[int] = []
    slices: list[list[tuple[int, int]]] = []
    for b, spans in enumerate(spans_list):
        sb: list[tuple[int, int]] = []
        for (t, n) in spans:
            sb.append((len(pos_list), len(pos_list) + n))
            for k in range(n):
                b_idx.append(b)
                pos_list.append(t - 1 + k)
                tok_list.append(seqs[b][t + k])
        slices.append(sb)

    b_t = torch.tensor(b_idx, dtype=torch.long, device=device)
    p_t = torch.tensor(pos_list, dtype=torch.long, device=device)
    h_slice = h[b_t, p_t]
    logits = model.lm_head(h_slice)
    t_t = torch.tensor(tok_list, dtype=torch.long, device=device)
    target = logits.gather(1, t_t.unsqueeze(1)).squeeze(1).float()
    logZ = _logsumexp_rows(logits)
    lp = target - logZ
    del logits
    return [[float(lp[s:e].sum().item()) for (s, e) in sb] for sb in slices]


def score_null_batch(
    tok, model, options: list[str], device: str, batch_size: int = 8
) -> dict[str, float]:
    """Batched PMI_DC null term: logP(option | "Answer:") for many options at once.

    Left-pads to the batch max length; the option tokens are always the final tokens, so the
    predicting logits are at positions `opt_start-1 .. opt_start+len(opt)-2`.
    """
    import torch

    prefix_ids = tok.encode("Answer:", add_special_tokens=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    scores: dict[str, float] = {}
    for i in range(0, len(options), batch_size):
        batch = options[i:i + batch_size]
        seqs = [prefix_ids + tok.encode(o, add_special_tokens=False) for o in batch]
        maxlen = max(len(s) for s in seqs)
        ids = [[pad_id] * (maxlen - len(s)) + s for s in seqs]
        attn = [[0] * (maxlen - len(s)) + [1] * len(s) for s in seqs]
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        attn_t = torch.tensor(attn, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.model(input_ids=ids_t, attention_mask=attn_t, use_cache=False)
        h = out.last_hidden_state  # [B, seq, d]

        b_idx: list[int] = []
        pos_list: list[int] = []
        tok_list: list[int] = []
        slices: list[tuple[int, int]] = []
        for b, s in enumerate(seqs):
            pad = maxlen - len(s)
            opt_start = pad + len(prefix_ids)
            opt_len = len(s) - len(prefix_ids)
            slices.append((len(pos_list), len(pos_list) + opt_len))
            for k in range(opt_len):
                b_idx.append(b)
                pos_list.append(opt_start - 1 + k)
                tok_list.append(s[len(prefix_ids) + k])

        b_t = torch.tensor(b_idx, dtype=torch.long, device=device)
        p_t = torch.tensor(pos_list, dtype=torch.long, device=device)
        h_slice = h[b_t, p_t]
        logits = model.lm_head(h_slice)
        t_t = torch.tensor(tok_list, dtype=torch.long, device=device)
        target = logits.gather(1, t_t.unsqueeze(1)).squeeze(1).float()
        logZ = _logsumexp_rows(logits)
        all_lp = target - logZ
        del logits
        for j, (s0, s1) in enumerate(slices):
            scores[batch[j]] = float(all_lp[s0:s1].sum().item())
        print(f"[causal] null batch {i + len(batch)}/{len(options)}", flush=True)
    return scores


SYSTEM_NUMBER = "You are a shell command corrector. Answer with the number of the best option."


def number_user_prompt(tok, text: str, opts: list[str]) -> str:
    user = text + "\n\nOptions:\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(opts))
    msgs = [{"role": "system", "content": SYSTEM_NUMBER},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def score_numbers(model, tok, prompt_ids: list[int], n: int, device: str) -> list[float]:
    """One forward; read logprobs of the n single-token answer numbers at the answer position."""
    import torch

    sp = tok.encode(" ", add_special_tokens=False)
    ids = torch.tensor([prompt_ids + sp], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.model(input_ids=ids, use_cache=False)
        h = out.last_hidden_state[0, -1]
        logits = model.lm_head(h)
        logZ = _logsumexp_rows(logits.unsqueeze(0))[0]
    scores: list[float] = []
    for i in range(1, n + 1):
        nid = tok.encode(str(i), add_special_tokens=False)
        if len(nid) != 1:
            raise ValueError(f"answer number {i} is not a single token: {nid}")
        scores.append(float(logits[nid[0]].float() - logZ))
    return scores


def score_causal_number(
    model_dir: str,
    items: list[dict],
    temperature: float = 1.0,
    device: str = "cuda",
    max_memory: dict | None = None,
) -> tuple[dict, float]:
    """Method D: numbered options in the user turn, score the answer-number token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.perf_counter()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head"],
    )
    tok = AutoTokenizer.from_pretrained(model_dir)
    kwargs: dict = dict(quantization_config=bnb, torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True)
    if max_memory is not None:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = max_memory
    else:
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    print(f"[causal-number] loaded {model_dir} in {time.perf_counter() - t0:.1f}s", flush=True)

    per_item_mats: list[np.ndarray] = []
    ts = time.perf_counter()
    for k, it in enumerate(items):
        opts = it["options"]
        n = len(opts)
        mat = np.zeros((n, n), dtype=np.float64)
        for r in range(n):
            order = opts[r:] + opts[:r]
            prefix = tok.encode(number_user_prompt(tok, it["text"], order),
                                add_special_tokens=False)
            scores = score_numbers(model, tok, prefix, n, device)
            for i in range(n):
                mat[r, (r + i) % n] = scores[i] / temperature
        per_item_mats.append(mat)
        if (k + 1) % 100 == 0:
            el = time.perf_counter() - ts
            print(f"[causal-number] {k + 1}/{len(items)} items, {el:.1f}s", flush=True)

    wall_s = time.perf_counter() - t0
    metrics = compute_metrics(items, per_item_mats)
    metrics["wall_s"] = round(wall_s, 2)
    metrics["ms_per_item"] = round(wall_s * 1000.0 / max(1, len(items)), 2)
    metrics["method"] = "number"
    return metrics, wall_s


def score_causal(
    model_dir: str,
    items: list[dict],
    temperature: float = 1.0,
    device: str = "cuda",
    max_memory: dict | None = None,
) -> tuple[dict, float]:
    """Score a HuggingFace causal LM with the mandatory §10.5.2 arm config.

    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head"]).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.perf_counter()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head"],
    )
    tok = AutoTokenizer.from_pretrained(model_dir)
    kwargs: dict = dict(
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = max_memory
    else:
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"[causal] loaded {model_dir} in {load_s:.1f}s", flush=True)

    # PMI_DC null term, once per distinct option string (batched)
    distinct = sorted({opt for it in items for opt in it["options"]})
    tn = time.perf_counter()
    null_scores = score_null_batch(tok, model, distinct, device)
    torch.cuda.empty_cache()
    print(f"[causal] null term: {len(distinct)} options in {time.perf_counter() - tn:.1f}s", flush=True)

    per_item_mats: list[np.ndarray] = []
    raw_mats: list[np.ndarray] = []
    pmi_len_mats: list[np.ndarray] = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ts = time.perf_counter()
    for k, it in enumerate(items):
        opts = it["options"]
        n = len(opts)
        mat = np.zeros((n, n), dtype=np.float64)
        raw_mat = np.zeros((n, n), dtype=np.float64)
        pmi_len_mat = np.zeros((n, n), dtype=np.float64)
        prefix = chat_prefix(tok, it["text"])
        orders: list[list[str]] = []
        seqs: list[list[int]] = []
        spans_list: list[list[tuple[int, int]]] = []
        for r in range(n):
            order = opts[r:] + opts[:r]
            ids, spans = encode_prompt(tok, prefix, order)
            orders.append(order)
            seqs.append(ids)
            spans_list.append(spans)
        rot_scores = score_rotations_batch(model, seqs, spans_list, device, pad_id=pad_id)
        opt_lens = [max(1, len(tok.encode(o, add_special_tokens=False))) for o in opts]
        for r in range(n):
            order = orders[r]
            scores = rot_scores[r]
            for i in range(n):
                orig_j = (r + i) % n
                pmi = scores[i] - null_scores[order[i]]
                mat[r, orig_j] = pmi / temperature
                raw_mat[r, orig_j] = scores[i] / temperature
                pmi_len_mat[r, orig_j] = pmi / (opt_lens[orig_j] * temperature)
        per_item_mats.append(mat)
        raw_mats.append(raw_mat)
        pmi_len_mats.append(pmi_len_mat)
        if (k + 1) % 100 == 0:
            el = time.perf_counter() - ts
            print(f"[causal] {k + 1}/{len(items)} items, {el:.1f}s", flush=True)

    wall_s = time.perf_counter() - t0
    metrics = compute_metrics(items, per_item_mats)
    metrics["raw"] = compute_metrics(items, raw_mats)
    metrics["pmi_len"] = compute_metrics(items, pmi_len_mats)
    metrics["wall_s"] = round(wall_s, 2)
    metrics["ms_per_item"] = round(wall_s * 1000.0 / max(1, len(items)), 2)
    metrics["n_distinct_options"] = len(distinct)
    return metrics, wall_s


def main() -> int:
    ap = argparse.ArgumentParser(prog="bakeoff")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draw", help="create/verify frozen 2,000-item stratified draw")
    d.add_argument("--force", action="store_true")

    s = sub.add_parser("score", help="score a candidate model on the frozen bake-off draw")
    s.add_argument("--model", required=True)
    s.add_argument("--kind", choices=["laya", "causal"], default="laya")
    s.add_argument("--method", choices=["list", "number"], default="number")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", default=None)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--device", default="cuda")
    s.add_argument("--max-memory", default=None,
                    help='e.g. "0:7GiB,cpu:4GiB" — enables device_map=auto')

    ns = ap.parse_args()
    if ns.cmd == "draw":
        info = build_draw(force=ns.force)
        print(json.dumps(info, indent=2))
        return 0

    if ns.cmd == "score":
        build_draw(force=False)
        items = []
        with open(ITEMS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        if ns.limit > 0:
            items = items[: ns.limit]

        max_memory = None
        if ns.max_memory:
            max_memory = {}
            for part in ns.max_memory.split(","):
                key, val = part.split(":", 1)
                max_memory[int(key) if key.isdigit() else key] = val

        if ns.kind == "causal":
            fn = score_causal_number if ns.method == "number" else score_causal
            metrics, _ = fn(
                ns.model, items, temperature=ns.temperature,
                device=ns.device, max_memory=max_memory,
            )
        else:
            metrics, _ = score_laya(ns.model, items)
        metrics["model"] = ns.model
        metrics["kind"] = ns.kind
        if ns.out:
            out_p = Path(ns.out)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
