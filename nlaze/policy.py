"""Per-user policy: the allowlist and the deny list (spec §4).

Auto runs only commands the user has certified safe. The allowlist makes Auto safe by
construction rather than by pattern coverage.

Precedence: (danger table ∪ deny list) > allowlist > model confidence. This module owns the
allowlist and the deny list; the danger table lives in `nlaze.danger` and is checked by the
caller before the policy.

Stored per-user at `$XDG_DATA_HOME/nlaze/policy.sqlite` (spec §7).
"""
from __future__ import annotations

import shlex
import sqlite3
from enum import Enum
from pathlib import Path

from nlaze.danger import flag_doubt


class Decision(Enum):
    ALLOW = "allow"              # head allowlisted, no flag concern -> AUTO
    DENY = "deny"                # on the deny list -> CONFIRM, always
    HEAD_PROMPT = "head_prompt"  # head not allowlisted -> certification prompt
    FLAG_PROMPT = "flag_prompt"  # Tier B flag trips the heuristic/doubt -> flag prompt


# Tier A heads: safe regardless of flags (spec §4.3).
TIER_A_HEADS = frozenset({
    "ls", "cd", "pwd", "cat", "head", "tail", "grep", "rg", "find", "df", "du",
    "ps", "free", "which", "echo", "wc", "file", "stat", "lsblk", "blkid", "mount", "z",
})

# Seeded allowlist (spec §4.4): read-only / navigational, so the prompt is rare and
# first-time users are not spammed into habitual approval.
SEED_HEADS = frozenset({
    "ls", "cd", "pwd", "cat", "head", "tail", "grep", "rg", "find", "df", "du",
    "ps", "free", "which", "echo", "wc", "file", "stat", "lsblk", "blkid", "mount", "z",
})
SEED_FLAGS = frozenset({
    ("git", "status"), ("git", "log"), ("git", "diff"), ("git", "show"),
})

DDL = """
CREATE TABLE IF NOT EXISTS allowlist (
    head TEXT NOT NULL,
    token TEXT NOT NULL,      -- '*' for a head-level entry
    PRIMARY KEY (head, token)
);
CREATE TABLE IF NOT EXISTS deny (
    head TEXT NOT NULL,
    token TEXT NOT NULL,
    PRIMARY KEY (head, token)
);
"""


def policy_tokens(cmd_line: str) -> tuple[str, list[str], list[str]]:
    """Extract the head, the positional args, and the long flags.

    Positional args are how subcommands enter the policy (`git status`, `docker system`);
    long flags (`--foo`, `--foo=bar` normalized to `--foo`) are how flags enter it. Short
    flags are ignored: they are too noisy to key on.
    """
    try:
        parts = shlex.split(cmd_line, posix=True)
    except ValueError:
        return "", [], []
    if not parts:
        return "", [], []
    head = parts[0]
    positionals: list[str] = []
    flags: list[str] = []
    for p in parts[1:]:
        if p.startswith("--"):
            flags.append(p.split("=", 1)[0])
        elif not p.startswith("-"):
            positionals.append(p)
    return head, positionals, flags


class Policy:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(DDL)
        self._seed()

    def _seed(self) -> None:
        for head in SEED_HEADS:
            self._add_allow(head, "*")
        for head, token in SEED_FLAGS:
            self._add_allow(head, token)

    def _add_allow(self, head: str, token: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO allowlist(head, token) VALUES (?,?)", (head, token))
        self.db.commit()

    # --- queries ------------------------------------------------------------

    def tier(self, head: str) -> str:
        return "A" if head in TIER_A_HEADS else "B"

    def is_allowlisted(self, head: str, token: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM allowlist WHERE head=? AND token=?", (head, token)).fetchone()
        return row is not None

    def is_denied(self, head: str, token: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM deny WHERE head=? AND token=?", (head, token)).fetchone()
        return row is not None

    # --- mutations ----------------------------------------------------------

    def allow_head(self, head: str) -> None:
        self._add_allow(head, "*")

    def allow_flag(self, head: str, token: str) -> None:
        self._add_allow(head, token)

    def deny(self, head: str, token: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO deny(head, token) VALUES (?,?)", (head, token))
        self.db.commit()

    def remove_allow_head(self, head: str) -> None:
        self.db.execute("DELETE FROM allowlist WHERE head=? AND token='*'", (head,))
        self.db.commit()

    def remove_allow_flag(self, head: str, token: str) -> None:
        self.db.execute("DELETE FROM allowlist WHERE head=? AND token=?", (head, token))
        self.db.commit()

    def remove_deny(self, head: str, token: str) -> None:
        self.db.execute("DELETE FROM deny WHERE head=? AND token=?", (head, token))
        self.db.commit()

    # --- audit --------------------------------------------------------------

    def list_allow(self) -> tuple[list[str], list[tuple[str, str]]]:
        heads = [r[0] for r in self.db.execute(
            "SELECT DISTINCT head FROM allowlist WHERE token='*' ORDER BY head")]
        flags = [(r[0], r[1]) for r in self.db.execute(
            "SELECT head, token FROM allowlist WHERE token!='*' ORDER BY head, token")]
        return heads, flags

    def list_deny(self) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.db.execute(
            "SELECT head, token FROM deny ORDER BY head, token")]

    # --- the decision -------------------------------------------------------

    def check(self, cmd_line: str) -> Decision:
        head, positionals, flags = policy_tokens(cmd_line)
        if not head:
            return Decision.HEAD_PROMPT

        # 1. the deny list is a hard veto (both subcommands and flags)
        for tok in positionals + flags:
            if self.is_denied(head, tok):
                return Decision.DENY

        # 2. allowlist: a head-level entry, or the subcommand explicitly allowed
        if not self.is_allowlisted(head, "*"):
            if not positionals or not self.is_allowlisted(head, positionals[0]):
                return Decision.HEAD_PROMPT

        # 3. Tier A heads are flag-insensitive
        if self.tier(head) == "A":
            return Decision.ALLOW

        # 4. Tier B heads are flag-scanned
        for flag in flags:
            if self.is_denied(head, flag):
                return Decision.DENY
            if self.is_allowlisted(head, flag):
                continue
            if flag_doubt(flag)[0]:
                return Decision.FLAG_PROMPT
        return Decision.ALLOW
