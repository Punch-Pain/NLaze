"""Tests for the Phase 1.3 bake-off draw and rotation-averaged metric computation."""
from __future__ import annotations

import json
import numpy as np
import pytest

from nlaze.candidates import INSTRUCTIONS, QID

bakeoff = pytest.importorskip("scripts.bakeoff")


@pytest.mark.skipif(
    not (bakeoff.ITEMS_PATH.exists() and bakeoff.MANIFEST_PATH.exists()),
    reason="internal data/bakeoff not present",
)
def test_bakeoff_frozen_draw_integrity_and_schema():
    info = bakeoff.build_draw(force=False)
    assert info["sha256"] == bakeoff.sha256_file(bakeoff.ITEMS_PATH)
    recorded = bakeoff.MANIFEST_PATH.read_text(encoding="utf-8").split()[0]
    assert info["sha256"] == recorded

    counts = {k: 0 for k in bakeoff.SLICE_QUOTAS}
    seen_ids = set()
    with open(bakeoff.ITEMS_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2000
    for r in rows:
        seen_ids.add(r["id"])
        counts[r["slice"]] += 1
        opts = r["options"]
        assert 5 <= len(opts) <= 7
        assert len(set(opts)) == len(opts)
        assert opts[r["gold_idx"]] == r["gold"]
        q = r["questions"][QID]
        assert q["instructions"] == INSTRUCTIONS
        assert list(q["criteria"].keys()) == [f"option_{i}" for i in range(len(opts))]
    assert len(seen_ids) == 2000
    assert counts == bakeoff.SLICE_QUOTAS


def test_bakeoff_compute_metrics_rotation_averaging():
    items = [
        {"slice": "typo", "options": ["git status", "git stash", "ls"], "gold_idx": 0},
        {"slice": "danger", "options": ["confirm_with_user()", "rm -rf /", "ls"], "gold_idx": 0},
    ]
    mat0 = np.array([
        [0.80, 0.15, 0.05],
        [0.75, 0.20, 0.05],
        [0.85, 0.10, 0.05],
    ])
    mat1 = np.array([
        [0.70, 0.20, 0.10],
        [0.40, 0.50, 0.10],
        [0.70, 0.20, 0.10],
    ])
    m = bakeoff.compute_metrics(items, [mat0, mat1])
    assert m["macro_top1"] == 1.0
    assert m["macro_top3"] == 1.0
    assert m["position_bias"]["flip_rate_rot"] == 0.5
    assert m["confidence_gap"]["when_correct"] > 0.3
