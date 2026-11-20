"""Tests for candidate retrieval, short-input priors (§1.3), ordering (§3.1/§3.2),
and Gate G1 option format freeze."""
from __future__ import annotations

import pytest

from nlaze.candidates import (
    INSTRUCTIONS,
    QID,
    TRAIN_K,
    ScoredCandidate,
    format_question,
    is_harness_confident,
    order_for_model,
    prime_candidate,
    retrieve_candidates,
    rotations_for_model,
)
from nlaze.usage import Store
from nlaze.verifier import command_is_safe_to_suggest, valid_as_typed


def test_retrieve_git_subcommand_typo():
    cands = retrieve_candidates("git statts")
    cmds = [c.cmd for c in cands]
    assert len(cands) == TRAIN_K
    assert len(set(cmds)) == len(cmds), "no duplicate candidates (§3.2)"
    assert cmds[0] == "git status"
    assert all(command_is_safe_to_suggest(c) for c in cmds)
    assert all(valid_as_typed(c)[0] for c in cmds)


def test_retrieve_git_head_and_subcommand_typo_preserves_flags():
    cands = retrieve_candidates("gti statts -s")
    cmds = [c.cmd for c in cands]
    assert cmds[0] == "git status -s"


def test_retrieve_cd_directory_typo(tmp_path):
    (tmp_path / "documents").mkdir()
    (tmp_path / "downloads").mkdir()
    (tmp_path / "desktop").mkdir()
    cands = retrieve_candidates("cd docments", cwd=tmp_path, use_zoxide=False)
    cmds = [c.cmd for c in cands]
    assert cmds[0] == "cd documents"
    assert "cd downloads" in cmds


def test_retrieve_short_inputs_use_prefix_and_frequency_prior():
    """Spec §1.3: short inputs (<= 3 chars) need prefix, keyboard-distance, and frequency prior."""
    cands_icp = [c.cmd for c in retrieve_candidates("icp")]
    assert "cp" in cands_icp and "ip" in cands_icp

    cands_mdir = [c.cmd for c in retrieve_candidates("mdir")]
    assert "mkdir" in cands_mdir and "rmdir" in cands_mdir

    cands_tat = [c.cmd for c in retrieve_candidates("tat")]
    assert "stat" in cands_tat and "tar" in cands_tat


def test_retrieve_user_history_has_highest_priority(tmp_path):
    """Spec §3: user-history frequency >= zoxide/path >= edit distance >= corpus prior."""
    s = Store(tmp_path / "usage.sqlite")
    for _ in range(5):
        s.record_accept("tat", "stat -c '%a %n'")
    cands = retrieve_candidates("tat", store=s)
    assert cands[0].cmd == "stat -c '%a %n'"
    assert cands[0].source == "history"


def test_prime_candidate_is_top_belief_never_duplicated():
    """Spec §3.2: prime candidate is top belief, never duplicated."""
    cands = retrieve_candidates("git statts")
    prime = prime_candidate(cands)
    assert prime is not None
    assert prime.cmd == "git status"
    ordered = order_for_model(cands, confident=True)
    assert ordered.count(prime.cmd) == 1


def test_ordering_confident_places_prime_last():
    """Spec §3.1: when confident, place top belief in last slot (slot 6) to exploit recency bias."""
    cands = [
        ScoredCandidate("git status", 0.96, "family"),
        ScoredCandidate("git stash", 0.72, "family"),
        ScoredCandidate("git stage", 0.65, "family"),
        ScoredCandidate("git show", 0.58, "family"),
        ScoredCandidate("git switch", 0.52, "family"),
        ScoredCandidate("git shortlog", 0.48, "family"),
        ScoredCandidate("git submodule", 0.44, "family"),
    ]
    assert is_harness_confident(cands)
    ordered = order_for_model(cands)
    assert len(ordered) == 7
    assert ordered[-1] == "git status"  # slot 6 (last slot, adjacent to input)
    assert ordered[0] == "git submodule"  # lowest-scoring rival in slot 0


def test_ordering_uncertain_rotates():
    """Spec §3.1: when not confident, rotate order rather than rubber-stamping slot 6."""
    cands = [
        ScoredCandidate("cp", 0.74, "exec"),
        ScoredCandidate("ip", 0.72, "exec"),
        ScoredCandidate("scp", 0.65, "exec"),
        ScoredCandidate("cmp", 0.55, "exec"),
        ScoredCandidate("cat", 0.40, "prior"),
        ScoredCandidate("ls", 0.35, "prior"),
        ScoredCandidate("pwd", 0.30, "prior"),
    ]
    assert not is_harness_confident(cands)
    rots = rotations_for_model(cands, n_rotations=3)
    assert len(rots) == 3
    # Each rotation has the same 7 unique candidates in a different cyclic shift
    assert len({tuple(r) for r in rots}) == 3
    for r in rots:
        assert sorted(r) == sorted(c.cmd for c in cands)


def test_gate_g1_frozen_option_format():
    """Gate G1: keys must be option_0..option_6 and instructions must match training schema."""
    cands = retrieve_candidates("git statts")
    ordered = order_for_model(cands, confident=True)
    q = format_question(ordered)
    assert list(q.keys()) == [QID]
    inner = q[QID]
    assert inner["type"] == "choice"
    assert inner["instructions"] == INSTRUCTIONS
    assert list(inner["criteria"].keys()) == [f"option_{i}" for i in range(TRAIN_K)]
    assert inner["criteria"][f"option_{TRAIN_K - 1}"] == "git status"
