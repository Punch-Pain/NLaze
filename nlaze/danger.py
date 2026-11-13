"""Danger classification for candidate command lines (deterministic table, no model).

Severity:
  mild  — privileged but read-only / reversible: confirm normally, Auto allowed.
  high  — irreversible, trust-boundary crossing: typed acknowledgment, never Auto.

A command matching a HIGH pattern is NEVER promoted to Auto or shown as an Auto candidate,
regardless of the user's history. Danger is a property of the candidate, not the user.

v6 (spec §2.1, §2.2):
  - **Privileged commands fail closed**: sudo/doas/pkexec default HIGH unless they match the
    explicit mild list.
  - **Expanded table**: covers all 32 of arm4's canonical destructive base commands.
  - **Deterministic flag heuristic** (`flag_doubt`) for Tier B heads — flag *names* are a
    smaller, more regular space than command lines, so a name-level heuristic is cheaper and
    more complete than a command-level table.
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
    (_pat(r"\brm\s+(-[a-z]*[rRf][a-z]*\s+)*(--no-preserve-root\s+)?\s*(/|\*|~|/home|/usr|/etc|/var|/boot|/lib|/bin|/sbin|/opt|/srv)(?=\s|$|/|\*)"),
     HIGH, "rm targeting system/home root"),
    (_pat(r"\bdd\s+.*\bof=/dev/(sd|hd|nvme|mmcblk|mapper/)"),
     HIGH, "dd writes to a raw block device"),
    (_pat(r"\b(mkfs|wipefs|shred|flashrom|forket|forkbomb)\b"),
     HIGH, "device-erase / format"),
    (_pat(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
     HIGH, "fork bomb"),
    (_pat(r"\bmkfs\.[a-z0-9]+\s"), HIGH, "filesystem format"),
    (_pat(r"\b(parted\s+.*\bmklabel\b|sgdisk\s+--zap-all|gdisk\s+.*\bmklabel\b)"),
     HIGH, "disk label rewrite"),
    (_pat(r"(>>?|\btee\s+-a\s+)\s*/etc/(passwd|shadow|sudoers)\b"),
     HIGH, "writing to an auth database"),
    (_pat(r">\s*/proc/sysrq-trigger"), HIGH, "sysrq trigger"),
    (_pat(r"\b(modprobe\s+-r\b|\brmmod\b)"), HIGH, "kernel module removal"),
    (_pat(r"\b(lvremove|vgreduce|pvremove)\b"), HIGH, "LVM volume removal"),
    (_pat(r"\bzpool\s+(destroy|labelclear)\b|\bmdadm\s+--zero-superblock\b"),
     HIGH, "zfs/mdraid destruction"),
    (_pat(r"\biptables\s+(-F|-X|-P)\b|\bufw\s+disable\b"), HIGH, "firewall flush/disable"),
    (_pat(r"\bfind\b.*\s-delete\b"), HIGH, "find -delete"),
    (_pat(r"\btar\b[^|;]*-C\s*/(etc|usr|var|lib|bin|sbin|boot|opt|srv|root|home)?(?=\s|$)(?:.*--overwrite)?"),
     HIGH, "tar extracting into a system path"),
    (_pat(r"\b(docker|podman)\s+(system|image|volume|container)\s+prune\b|\b(docker|podman)\s+volume\s+rm\b|\b(docker|podman)\s+rm\s+-f\b"),
     HIGH, "container prune/force-remove"),
    (_pat(r"\b(userdel|groupdel)\b"), HIGH, "user/group deletion"),
    (_pat(r"\bpacman\s+-R(dd|ns?|sc)\b"), HIGH, "pacman recursive remove"),
    (_pat(r"\bgit\s+push\s+.*(-force|--force|--force-with-lease)(\s|$)"), HIGH, "git force push"),
    (_pat(r"\bgit\s+reset\s+--hard\b"), HIGH, "git hard reset discards work"),
    (_pat(r"\bcrontab\s+-r\b"), HIGH, "crontab removal"),
    (_pat(r"\bchmod\s+-R\s+777\s+(/|\*|/home|/usr|/etc|/var|/boot)(?=\s|$|/)"), HIGH, "recursive 777 on system path"),
    (_pat(r"\bmke2fs\b|\bfdisk\s+-w\b"), HIGH, "partition write"),
    (_pat(r"\bchown\s+-R\s+[^\s]+\s+(/|\*|/home|/usr|/etc|/var|/boot)(?=\s|$|/)"), HIGH, "recursive chown system path"),
    (_pat(r"\bkill\s+-9\s+-1\b"), HIGH, "kill all user processes"),
    (_pat(r"\bhdparm\b.*--security-erase\b"), HIGH, "drive security erase"),
    # --- MILD: privileged but read-only / reversible (the explicit list, §2.1) ---
    (_pat(r"\bsudo\s+(whoami|id|hostname|date|uname|cat\s+/etc/os-release|systemctl\s+(status|is-active|list-units)\b|journalctl\b|mount\s+-l\b)\b"),
     MILD, "read-only privileged query"),
    (_pat(r"\bsudo\s+-n\s+true\b"), MILD, "sudo token probe"),
    (_pat(r"\bsudo\s+true\b|\bsudo\s+-v\b"), MILD, "sudo cache refresh"),
]

# §2.1 — anything privileged that is not on the explicit mild list fails closed.
_PRIVILEGED = _pat(r"(?:^|[\s;&|])(sudo|doas|pkexec)(?:\s|$)")

# §2.2 — deterministic flag heuristic. Flag names matching a destructive prefix, or one of
# the exact names, require certification regardless of the model.
_FLAG_PREFIXES = ("remove", "delete", "destroy", "wipe", "erase", "purge", "drop",
                  "kill", "reset", "clean", "auto")
_FLAG_EXACT = frozenset({"--force", "--yes", "--sudo", "--root"})


def classify(cmd_line: str) -> Severity:
    """Return the severity of a candidate command line. NONE if no match."""
    line = cmd_line.strip()
    if not line:
        return NONE
    for pat, sev, _detail in TABLE:
        if pat.search(line):
            return sev
    if _PRIVILEGED.search(line):
        return HIGH
    return NONE


def flag_doubt(cmd_line: str) -> tuple[bool, str]:
    """§2.2 deterministic flag heuristic.

    True iff any long flag matches a destructive prefix or one of the exact names. Cheap,
    name-level, and deliberately over-inclusive: it triggers a *prompt*, never a block.
    """
    for tok in cmd_line.split():
        if not tok.startswith("--"):
            continue
        name = tok.split("=", 1)[0].lower()
        if name in _FLAG_EXACT:
            return True, f"destructive flag {name}"
        body = name[2:]
        for prefix in _FLAG_PREFIXES:
            if body.startswith(prefix):
                return True, f"destructive flag {name}"
    return False, ""


def is_high(cmd_line: str) -> bool:
    return classify(cmd_line) is HIGH


def is_mild(cmd_line: str) -> bool:
    return classify(cmd_line) is MILD
