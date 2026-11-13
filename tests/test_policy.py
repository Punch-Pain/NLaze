"""Per-user policy: allowlist + deny list (spec §4)."""
import pytest

from nlaze.policy import Policy, Decision, policy_tokens


@pytest.fixture
def policy(tmp_path):
    return Policy(tmp_path / "policy.sqlite")


# --- token extraction -------------------------------------------------------

@pytest.mark.parametrize("cmd,head,pos,flags", [
    ("ls -la", "ls", [], []),
    ("git status", "git", ["status"], []),
    ("git push --force origin main", "git", ["push", "origin", "main"], ["--force"]),
    ("opencode --remove-database", "opencode", [], ["--remove-database"]),
    ("docker system prune -a", "docker", ["system", "prune"], []),
])
def test_policy_tokens(cmd, head, pos, flags):
    h, p, f = policy_tokens(cmd)
    assert h == head
    assert p == pos
    assert f == flags


# --- seeded allowlist -------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "ls -la",
    "cd /tmp",
    "cat /etc/passwd",
    "rg foo src/",
    "git status",
    "git log --oneline",
    "df -h",
    "mount -l",
])
def test_seeded_commands_allowed(policy, cmd):
    assert policy.check(cmd) is Decision.ALLOW, cmd


def test_unseeded_head_prompts(policy):
    assert policy.check("opencode --remove-database") is Decision.HEAD_PROMPT
    assert policy.check("docker system prune -a") is Decision.HEAD_PROMPT


# --- head certification -----------------------------------------------------

def test_head_certification_then_auto(policy):
    assert policy.check("opencode foo") is Decision.HEAD_PROMPT
    policy.allow_head("opencode")
    assert policy.check("opencode foo") is Decision.ALLOW


def test_tier_b_head_still_scans_flags(policy):
    policy.allow_head("opencode")
    # head allowed, but the flag trips the heuristic
    assert policy.check("opencode --remove-database") is Decision.FLAG_PROMPT


def test_tier_a_head_does_not_scan_flags(policy):
    # `find` is seeded Tier A; `find / -delete` is caught by the danger table,
    # not by the flag heuristic — the policy itself must not prompt.
    policy.allow_head("find")
    assert policy.check("find / -delete") is Decision.ALLOW


# --- flag certification -----------------------------------------------------

def test_flag_certification_then_auto(policy):
    policy.allow_head("opencode")
    assert policy.check("opencode --remove-database") is Decision.FLAG_PROMPT
    policy.allow_flag("opencode", "--remove-database")
    assert policy.check("opencode --remove-database") is Decision.ALLOW


# --- deny list --------------------------------------------------------------

def test_denied_flag_always_confirms(policy):
    policy.allow_head("opencode")
    policy.deny("opencode", "--remove-database")
    assert policy.check("opencode --remove-database") is Decision.DENY


def test_deny_wins_over_allow(policy):
    policy.allow_head("opencode")
    policy.allow_flag("opencode", "--remove-database")
    policy.deny("opencode", "--remove-database")
    assert policy.check("opencode --remove-database") is Decision.DENY


def test_deny_is_per_flag(policy):
    policy.allow_head("opencode")
    policy.deny("opencode", "--remove-database")
    assert policy.check("opencode --other-flag") is Decision.ALLOW


# --- audit ------------------------------------------------------------------

def test_list_and_remove(policy):
    policy.allow_head("opencode")
    policy.allow_flag("opencode", "--remove-database")
    policy.deny("opencode", "--drop-data")
    heads, flags = policy.list_allow()
    assert "opencode" in heads
    assert ("opencode", "--remove-database") in flags
    assert ("opencode", "--drop-data") in policy.list_deny()

    policy.remove_allow_flag("opencode", "--remove-database")
    assert ("opencode", "--remove-database") not in policy.list_allow()[1]

    policy.remove_deny("opencode", "--drop-data")
    assert policy.list_deny() == []
