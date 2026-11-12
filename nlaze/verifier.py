"""Verifier — primary gate. A candidate is never suggested unless this passes."""
from __future__ import annotations
import os
import shutil
import subprocess

def head_resolves(head: str) -> bool:
    """True iff `head` is an existing command (PATH or shell builtin)."""
    if not head:
        return False
    # torch of shells built in: check common builtins first
    if head in {"cd", "exit", "echo", "true", "false", "export", "set", "unset", "source", ".", "local", "return", "exec", "command", "builtin", "enable"}:
        return True
    return shutil.which(head) is not None

def path_exists(path: str) -> bool:
    return os.path.exists(os.path.expanduser(path))

def command_is_safe_to_suggest(cmd_line: str) -> bool:
    """Verifier for a *candidate* (not the user's raw input — that uses zero model anyway)."""
    parts = cmd_line.split()
    if not parts:
        return False
    if not head_resolves(parts[0]):
        return False
    # last resort safety floor: never suggest a command that overwrites a raw block device
    if any(p.startswith("of=/dev/") for p in parts[1:]):
        return False
    return True
