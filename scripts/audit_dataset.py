#!/usr/bin/env python3
"""Audit a generated NLaze dataset (teacher outputs in the frozen G1 format).

Checks (pure string inspection — nothing is executed):
  1. Structure: 5-7 options, all unique, gold in options, probs length matches, sum ~= 1,
     argmax matches teacher_probs.
  2. Command validity: every option resolves to a real head; no raw-block-device writes.
     Skipped for the danger slice (gold may be `confirm_with_user()`).
  3. Duplicates: no duplicate `input` (whitespace-normalised).
  4. Danger safety: for the danger slice, a dangerous option must never be the gold, and
     the gold must be `confirm_with_user()` when any option is dangerous.
  5. Slice balance and length distribution.

Usage:
  python scripts/audit_dataset.py --path data/teacher_out/train_v1.jsonl [--sample 500] [--json]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from nlaze.danger import HIGH, classify

DANGER_GOLD_FORMS = {"confirm_with_user()", "confirm_with_user"}

_HEAD_RE = re.compile(r"^[A-Za-z0-9_./~][A-Za-z0-9_.+/~:-]*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def structurally_valid_option(o: str) -> bool:
    """A well-formed command string: non-empty, not a bare flag, plausible head.

    Deliberately does NOT require the binary to be installed — the corpus contains
    legitimate tools (`tree`, `nmap`, `gifsicle`, …) that are not present on the
    auditing machine. Installation is a verifier concern at serve time, not a
    dataset-integrity concern.
    """
    o = o.strip()
    if not o or o.startswith("-"):
        return False
    head = o.split()[0]
    return bool(_HEAD_RE.match(head))


def audit(rows: list[dict], sample: int = 0) -> dict:
    if sample and sample < len(rows):
        rows = random.Random(20261120).sample(rows, sample)

    failures: list[dict] = []
    seen_inputs: dict[str, int] = {}
    slices: Counter = Counter()
    problems = Counter()
    n_opts: Counter = Counter()
    argmax_matches = 0
    argmax_checked = 0

    for idx, r in enumerate(rows):
        sl = r.get("slice", "unknown")
        slices[sl] += 1
        inp = _norm(r.get("input", ""))
        opts = r.get("options", [])
        gold = r.get("gold", "")
        probs = r.get("teacher_probs")
        n_opts[len(opts)] += 1

        def fail(kind: str, detail: str):
            problems[kind] += 1
            if len(failures) < 50:
                failures.append({"idx": idx, "slice": sl, "input": inp[:120],
                                 "kind": kind, "detail": detail})

        if not inp:
            fail("empty_input", "")
            continue
        if inp in seen_inputs:
            fail("duplicate_input", f"first at {seen_inputs[inp]}")
        seen_inputs[inp] = idx

        if not (5 <= len(opts) <= 7):
            fail("bad_option_count", str(len(opts)))
            continue
        if len(set(opts)) != len(opts):
            fail("duplicate_options", "")
        if gold not in opts:
            fail("gold_not_in_options", gold[:80])

        if probs is not None:
            if len(probs) != len(opts):
                fail("probs_length_mismatch", f"{len(probs)} vs {len(opts)}")
            elif abs(sum(probs) - 1.0) > 0.05:
                fail("probs_not_normalised", f"{sum(probs):.3f}")
            else:
                am = int(max(range(len(probs)), key=lambda i: probs[i]))
                argmax_checked += 1
                if opts[am] == gold:
                    argmax_matches += 1

        # Command validity (skip danger slice: gold may be confirm_with_user())
        if sl != "danger":
            for o in opts:
                if o in DANGER_GOLD_FORMS:
                    fail("confirm_form_outside_danger", o)
                    continue
                if not structurally_valid_option(o):
                    fail("invalid_option", o[:80])
                if any(p.startswith("of=/dev/") for p in o.split()):
                    fail("blockdev_write_option", o[:80])

        # Danger safety
        if sl == "danger":
            dangerous = [o for o in opts if classify(o) is HIGH]
            if dangerous and gold not in DANGER_GOLD_FORMS:
                fail("dangerous_gold", gold[:80])

    total = max(1, len(rows))
    return {
        "n_audited": len(rows),
        "slices": dict(slices),
        "option_counts": dict(n_opts),
        "problems": dict(problems),
        "passed": all(v == 0 for v in problems.values()),
        "teacher_argmax_rate": round(argmax_matches / max(1, argmax_checked), 4),
        "failure_examples": failures,
        "checks": {
            "structure": problems["bad_option_count"] == 0 and problems["duplicate_options"] == 0,
            "gold_present": problems["gold_not_in_options"] == 0,
            "no_duplicates": problems["duplicate_input"] == 0,
            "options_valid": problems["invalid_option"] == 0 and problems["blockdev_write_option"] == 0,
            "danger_safe": problems["dangerous_gold"] == 0,
        },
    }


def load_rows(path: Path) -> list[dict]:
    if path.is_dir():
        rows: list[dict] = []
        for f in sorted(path.glob("*.jsonl")):
            rows.extend(json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        return rows
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args()

    p = Path(ns.path)
    rows = load_rows(p)
    report = audit(rows, sample=ns.sample)
    if ns.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"audited {report['n_audited']} rows; passed={report['passed']}, "
              f"teacher_argmax_rate={report['teacher_argmax_rate']}")
        for k, v in report["checks"].items():
            print(f"  {'OK ' if v else 'FAIL'} {k}")
        if report["problems"]:
            print("problems:", report["problems"])
        for f in report["failure_examples"][:10]:
            print("  -", f)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
