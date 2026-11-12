"""Danger classification for candidate command lines (deterministic table, no model).

Severity:
  mild  — privileged but read-only / reversible: confirm normally, Auto allowed.
  high  — irreversible, trust-boundary crossing: typed acknowledgment, never Auto.

A command matching `high_patterns` is NEVER promoted to Auto or shown as an Auto candidate,
regardless of user's history. Danger is a property of the candidate, not the user.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Severity:
    name: str
    detail: str

MILD = Severity("mild", "privileged but read-only / reversible")
HIGH = Severity("high", "irreversible, trust-boundary crossing")
NONE = Severity("none", "not dangerous")

def _pat(rx: str) -> re.Pattern[str]:
    return re.compile(rx, re.IGNORECASE)

# Ordered: first match wins. Each tuple: (pattern, severity, detail)
TABLE: list[tuple[re.Pattern[str], Severity, str]] = [
    # --- HIGH: catastrophic, irreversible ---
    (_pat(r"\brm\s+(-[a-z]*[rRf][a-z]*\s+)*(--no-preserve-root)?\s*(/|\*|/home|~|/usr|/etc|/var|/boot)(\s|$|/)"),
     HIGH, "rm targeting system/home root"),
    (_pat(r"\bdd\s+.*\bof=/dev/(sd|hd|nvme|mmcblk)"), HIGH, "dd writes to a raw block device"),
    (_pat(r"\b(mkfs|wipefs|shred|flashrom|forket|forkbomb|:\(\))\b"), HIGH, "device-erase / format / forkbomb"),
    (_pat(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), HIGH, "fork bomb"),
    (_pat(r"\bmkfs\.[a-z0-9]+\s"), HIGH, "filesystem format"),
    (_pat(r"\bgit\s+push\s+.*(-force|--force|--force-with-lease)(\s|$)"), HIGH, "git force push"),
    (_pat(r"\bgit\s+reset\s+--hard\b"), HIGH, "git hard reset discards work"),
    (_pat(r"\bcrontab\s+-r\b"), HIGH, "crontab removal"),
    (_pat(r"\bwipefs\b"), HIGH, "filesystem signature wipe"),
    (_pat(r"\bchmod\s+-R\s+777\s+(/|\*|/home|/usr|/etc|/var|/boot)(\s|$|/)"), HIGH, "recursive 777 on system path"),
    (_pat(r"\bmke2fs\b|\bfdisk\s+-w\b"), HIGH, "partition write"),
    (_pat(r"\bchown\s+-R\s+[^\s]+\s+(/|\*|/home|/usr|/etc|/var|/boot)(\s|$|/)"), HIGH, "recursive chown system path"),
    (_pat(r"\bkill\s+-9\s+-1\b"), HIGH, "kill all user processes"),
    (_pat(r"\bhdparm\s+.*--security-erase\b"), HIGH, "drive security erase"),
    # --- MILD: privileged but read-only / reversible ---
    (_pat(r"\bsudo\s+(whoami|id|hostname|date|uname|cat\s+/etc/os-release|systemctl\s+(status|is-active|list-units)\b|journalctl\b|mount\s+-l\b|ls\b)"),
     MILD, "read-only privileged query"),
    (_pat(r"\bsudo\s+-n\s+true\b"), MILD, "sudo token probe"),
    (_pat(r"\bsudo\s+id\b"), MILD, "privileged id"),
    (_pat(r"\bsudo\s+true\b|\bsudo\s+-v\b"), MILD, "sudo cache refresh"),
    (_pat(r"\bsudo\s+(apt|dnf|pacman|yay)(\s+-[a-zA-Z]+)?\s+(update|upgrade|check|list)\b"), MILD, "package index refresh, no changes"),
]

def classify(cmd_line: str) -> Severity:
    """Return the severity of a candidate command line. NONE if no match."""
    line = cmd_line.strip()
    if not line:
        return NONE
    for pat, sev, _detail in TABLE:
        if pat.search(line):
            return sev
    return NONE

def is_high(cmd_line: str) -> bool:
    return classify(cmd_line) is HIGH

def is_mild(cmd_line: str) -> bool:
    return classify(cmd_line) is MILD
