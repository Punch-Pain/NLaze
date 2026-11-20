"""Tests for HarnessEngine, direct serve (§6), postexec flow (§1-§5),
Unix socket daemon (§7), and fish hook integration."""
from __future__ import annotations

import json
import os
import stat
import threading
import time
import pytest

from nlaze.daemon import (
    DaemonServer,
    HarnessEngine,
    query_daemon_or_local,
)
from nlaze.lanes import Lane
from nlaze.policy import Policy
from nlaze.usage import Store


def _make_engine(tmp_path, predict_fn=None):
    store = Store(tmp_path / "usage.sqlite")
    policy = Policy(tmp_path / "policy.sqlite")
    return HarnessEngine(store=store, policy=policy, predict_fn=predict_fn)


def test_postexec_success_ticks_and_passthroughs(tmp_path):
    eng = _make_engine(tmp_path)
    res = eng.handle_postexec("ls -la", exit_status=0, cwd=tmp_path)
    assert res["action"] == "passthrough"
    assert eng.store.tick_now() == 1


def test_postexec_failed_valid_command_passthroughs(tmp_path):
    """Rule 0: a valid command that fails at runtime (e.g. grep with no match) never triggers correction."""
    eng = _make_engine(tmp_path)
    res = eng.handle_postexec("grep definitely_missing_pattern /etc/hosts", exit_status=1, cwd=tmp_path)
    assert res["action"] == "passthrough"
    assert res["lane"] == Lane.PASSTHROUGH.value


def test_postexec_typo_suggests_correction_and_records_offer(tmp_path):
    eng = _make_engine(tmp_path)
    res = eng.handle_postexec("git statts", exit_status=1, cwd=tmp_path)
    assert res["correction"] == "git status"
    assert res["lane"] == Lane.CONFIRM.value
    assert res["action"] == "confirm"
    assert eng.reconditioner.offer_count() == 1


def test_postexec_unallowlisted_head_triggers_certify_head(tmp_path):
    """Spec §4.1: a proposed correction whose head is not on the allowlist triggers certification."""
    eng = _make_engine(tmp_path)
    # 'pytest' or 'python3' is not in SEED_HEADS; let's check 'python3 --version' from typo 'pythno3 --version'
    res = eng.handle_postexec("pythno3 --version", exit_status=127, cwd=tmp_path)
    assert res["correction"] == "python3 --version"
    assert res["action"] == "certify_head"
    assert res["head"] == "python3"


def test_postexec_tier_b_destructive_flag_triggers_certify_flag(tmp_path):
    """Spec §2.2 & §4.1: allowlisted Tier B head with a destructive flag triggers flag prompt."""
    eng = _make_engine(tmp_path)
    eng.policy.allow_head("python3")
    res = eng.handle_postexec("pythno3 --remove-cache", exit_status=127, cwd=tmp_path)
    assert res["correction"] == "python3 --remove-cache"
    assert res["action"] == "certify_flag"
    assert res["head"] == "python3"
    assert res["flag"] == "--remove-cache"

    # Rejecting the flag prompt puts (python3, --remove-cache) on the deny list -> always confirm
    eng.handle_feedback(
        input_text="pythno3 --remove-cache",
        correction="python3 --remove-cache",
        accepted=False,
        certify_kind="flag",
        head="python3",
        flag="--remove-cache",
    )
    assert eng.policy.is_denied("python3", "--remove-cache")
    res2 = eng.handle_postexec("pythno3 --remove-cache", exit_status=127, cwd=tmp_path)
    assert res2["action"] == "confirm"


def test_postexec_dangerous_correction_never_auto_runs(tmp_path):
    """Spec §2: dangerous candidate gets confirm_high even if allowlisted and abundant history."""
    eng = _make_engine(tmp_path)
    eng.policy.allow_head("git")
    for _ in range(25):
        eng.store.record_accept("git pusch --force origin main", "git push --force origin main")
    for _ in range(110):
        eng.store.tick()
    res = eng.handle_postexec("git pusch --force origin main", exit_status=1, cwd=tmp_path)
    assert res["correction"] == "git push --force origin main"
    assert res["lane"] == Lane.DANGER_HIGH.value
    assert res["action"] == "confirm_high"


def test_direct_serve_under_5ms_without_calling_model(tmp_path):
    """Spec §6: after Auto gate is satisfied, serve directly from cache in <= 5 ms without loading model."""
    model_called = False

    def boom_predict(*args, **kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError("model should not be called on direct serve")

    eng = _make_engine(tmp_path, predict_fn=boom_predict)
    for _ in range(20):
        eng.store.record_accept("gti status", "git status")
    for _ in range(105):
        eng.store.tick()

    t0 = time.perf_counter()
    res = eng.handle_postexec("gti status", exit_status=127, cwd=tmp_path)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert not model_called
    assert res["action"] == "auto"
    assert res["source"] == "direct_serve"
    assert res["correction"] == "git status"
    assert dt_ms <= 5.0, f"direct serve took {dt_ms:.2f} ms (> 5 ms)"


def test_direct_serve_stays_alive_under_continued_use(tmp_path):
    """Regression: repeated direct-serve auto-runs must refresh pair evidence so an actively
    used shortcut does not decay below AUTO_NEFF while being used."""
    eng = _make_engine(tmp_path)
    for _ in range(18):
        eng.store.record_accept("gti status", "git status")
    for _ in range(105):
        eng.store.tick()
    for _ in range(250):
        res = eng.handle_postexec("gti status", exit_status=127, cwd=tmp_path)
        assert res["action"] == "auto" and res["source"] == "direct_serve"


def test_strike_demotes_direct_serve_to_confirm(tmp_path):
    """Spec §5.2 & §6: running an alternative from a recent offer demotes AUTO/DIRECT to CONFIRM."""
    eng = _make_engine(tmp_path)
    for _ in range(20):
        eng.store.record_accept("gti status", "git status")
    for _ in range(105):
        eng.store.tick()

    # 1. Auto-runs via direct serve and records offer containing rivals (e.g. git stash)
    res1 = eng.handle_postexec("gti status", exit_status=127, cwd=tmp_path)
    assert res1["action"] == "auto"
    offered = res1["offered"]
    rival = next(c for c in offered if c != "git status")

    # 2. Next command: user runs rival (`git stash`), causing 1 strike on ("gti status", "git status")
    eng.handle_postexec(rival, exit_status=0, cwd=tmp_path)

    # 3. Next time user types `gti status`, it is demoted from AUTO to CONFIRM!
    res2 = eng.handle_postexec("gti status", exit_status=127, cwd=tmp_path)
    assert res2["action"] == "confirm"


def test_unix_socket_daemon_roundtrip_and_permissions(tmp_path):
    """Spec §7: per-user socket has 0600 permissions and answers postexec queries."""
    sock_path = tmp_path / "run" / "daemon.sock"
    eng = _make_engine(tmp_path)
    srv = DaemonServer(sock_path, eng)
    srv.start_background()
    try:
        mode = stat.S_IMODE(os.stat(sock_path).st_mode)
        assert mode == 0o600
        resp = query_daemon_or_local(
            {"op": "postexec", "cmd": "git statts", "status": 1, "cwd": str(tmp_path)},
            sock_path=sock_path,
            fallback_engine=None,
        )
        assert resp["correction"] == "git status"
        assert resp["action"] == "confirm"
    finally:
        srv.stop()
