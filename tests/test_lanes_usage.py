import os, tempfile
import pytest
from nlaze.lanes import classify_input, classify_candidate, Lane
from nlaze.usage import Store
from nlaze.policy import Policy

@pytest.mark.parametrize("cmd,lane", [
    ("", Lane.PASSTHROUGH),
    ("ls -la", Lane.PASSTHROUGH),
    ("grep -r foo .", Lane.PASSTHROUGH),
    ("git push origin main", Lane.PASSTHROUGH),
    ("sudo -n true", Lane.DANGER_MILD),
    ("sudo whoami", Lane.DANGER_MILD),
    ("rm -rf / --no-preserve-root", Lane.DANGER_HIGH),
    (":(){ :|:& };:", Lane.DANGER_HIGH),
    ("not_a_command_xyz_abc", Lane.CONFIRM),
])
def test_lane_input(cmd, lane):
    assert classify_input(cmd).lane is lane

def test_candidate_never_auto_on_danger():
    assert classify_candidate("rm -rf / --no-preserve-root", history_pairs_clean=999).lane is Lane.DANGER_HIGH

def test_mild_candidate_never_auto():
    assert classify_candidate("sudo whoami", history_pairs_clean=999).lane is Lane.DANGER_MILD

def test_candidate_auto_only_after_history():
    assert classify_candidate("git status", history_pairs_clean=0).lane is Lane.CONFIRM
    assert classify_candidate("git status", history_pairs_clean=17).lane is Lane.AUTO
    assert classify_candidate("git status", history_pairs_clean=100).lane is Lane.AUTO

def test_store_tick_monotonic(tmp_path):
    s = Store(tmp_path / "u.sqlite")
    for _ in range(5):
        assert s.tick() == s.tick_now()

def test_store_decay_and_auto(tmp_path):
    s = Store(tmp_path / "u.sqlite")
    for _ in range(18):
        s.record_accept("gti status", "git status")
    # spec rule: ≥100 ticks from first accept for the pair to promote to Auto
    for _ in range(100):
        s.tick()
    assert s.auto_eligible("gti status", "git status")

def test_store_reject_halves_and_quarantine(tmp_path):
    s = Store(tmp_path / "u.sqlite")
    s.record_accept("gti status", "git status")
    s.record_reject("gti status", "git status")
    assert not s.auto_eligible("gti status", "git status")

def test_rejected_ambiguous_state(tmp_path):
    s = Store(tmp_path / "u.sqlite")
    s.record_accept("cd prjects", "cd projects")
    s.record_reject("cd prjects", "cd projects")
    # low-confidence reject marks pair as not-auto-eligible
    assert not s.auto_eligible("cd prjects", "cd projects")

def test_quarantine_flag(tmp_path):
    s = Store(tmp_path / "u.sqlite")
    s.record_reject("dd if=/dev/zero of=/dev/sda", "dd of=/dev/sda")
    assert s.quarantined("dd if=/dev/zero of=/dev/sda", "dd of=/dev/sda")

def test_hash_input_deterministic(tmp_path):
    s1, s2 = Store(tmp_path / "a.sqlite"), Store(tmp_path / "b.sqlite")
    assert s1.hash_input("git status") == s2.hash_input("git status")


# --- policy integration (spec §4.2 precedence) ------------------------------

def test_candidate_denied_by_policy(tmp_path):
    p = Policy(tmp_path / "p.sqlite")
    p.deny("opencode", "--remove-database")
    v = classify_candidate("opencode --remove-database", history_pairs_clean=999, policy=p)
    assert v.lane is Lane.CONFIRM


def test_candidate_allowlisted_auto(tmp_path):
    p = Policy(tmp_path / "p.sqlite")
    assert classify_candidate("git status", history_pairs_clean=999, policy=p).lane is Lane.AUTO


def test_candidate_unallowlisted_confirms(tmp_path):
    p = Policy(tmp_path / "p.sqlite")
    # even with abundant history, an unallowlisted head must not auto-run
    assert classify_candidate("opencode foo", history_pairs_clean=999, policy=p).lane is Lane.CONFIRM


def test_danger_beats_allowlist(tmp_path):
    p = Policy(tmp_path / "p.sqlite")
    p.allow_head("rm")
    v = classify_candidate("rm -rf /", history_pairs_clean=999, policy=p)
    assert v.lane is Lane.DANGER_HIGH
