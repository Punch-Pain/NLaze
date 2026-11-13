"""Verifier — primary gate. A candidate is never suggested unless this passes.

Also owns the front door's first-argument validity check (spec §1.1): a command is
valid as typed only if its head resolves AND, for known families, the first argument
is real. This is what stops `cd docments` / `git statts` / `sudo git commmit` from
passing through without ever reaching the corrector.

The check is deterministic and cheap: static subcommand tables, no shelling out to the
command being validated.
"""
from __future__ import annotations
import os
import re
import shlex
import shutil

_BUILTINS = {
    "cd", "exit", "echo", "true", "false", "export", "set", "unset", "source", ".",
    "local", "return", "exec", "command", "builtin", "enable",
}

# Static tables (spec §1.1): "valid iff <sub> is a real subcommand".
GIT_SUBCOMMANDS = frozenset("""
add am annotate apply archive bisect blame branch bundle checkout cherry-pick citool
clean clone column commit config describe diff difftool fetch format-patch gc grep gui
init log ls-files ls-tree merge mergetool mv notes pull push range-diff rebase reflog
remote repack replace request-pull reset restore revert rm shortlog show show-branch
stash status submodule switch tag worktree verify-commit verify-tag version
""".split())

CONTAINER_SUBCOMMANDS = frozenset("""
attach build builder commit container context cp create diff events exec export history
image images import info inspect kill load login logout logs manifest network node pause
plugin port ps pull push rename restart rm rmi run save search secret service stack
start stats stop swarm system tag top unpause update version volume wait compose
""".split())

# sudo flags that consume the following token as their argument.
_SUDO_FLAGS_WITH_ARG = frozenset({"-u", "-g", "-h", "-p", "-C", "-U", "-r", "-R", "-t", "-T"})

_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def head_resolves(head: str) -> bool:
    """True iff `head` is an existing command (PATH or shell builtin)."""
    if not head:
        return False
    if head in _BUILTINS:
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


def _split(cmd_line: str) -> list[str] | None:
    try:
        return shlex.split(cmd_line, posix=True)
    except ValueError:
        return None


def _strip_sudo(parts: list[str]) -> list[str]:
    """Drop sudo's own flags so the *real* command head is what gets validated."""
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "--":
            return parts[i + 1:]
        if tok.startswith("-") and tok != "-":
            i += 2 if (tok in _SUDO_FLAGS_WITH_ARG and i + 1 < len(parts)) else 1
            continue
        return parts[i:]
    return []


def _strip_prefixes(parts: list[str]) -> list[str]:
    """Resolve the effective head through sudo / env / exec-like wrappers."""
    while parts:
        head = parts[0]
        if head == "sudo":
            parts = _strip_sudo(parts[1:])
            continue
        if head == "env":
            parts = parts[1:]
            while parts and (parts[0].startswith("-") or _ENV_ASSIGN.match(parts[0])):
                parts = parts[1:]
            continue
        if head in {"exec", "command", "nohup", "builtin"}:
            parts = parts[1:]
            continue
        break
    return parts


def valid_as_typed(cmd_line: str) -> tuple[bool, str]:
    """Front door (spec §1.1). True iff the command is valid as typed.

    Narrow but argument-aware:
      - `cd <arg>` valid iff `<arg>` exists after expansion.
      - `git <sub>` valid iff `<sub>` is a real git subcommand.
      - `docker|podman <sub>` valid iff `<sub>` is a real subcommand.
      - general: head resolves -> valid.
    """
    parts = _split(cmd_line)
    if parts is None:
        return False, "unparseable quoting"
    if not parts:
        return False, "empty input"
    parts = _strip_prefixes(parts)
    if not parts:
        return False, "no command after prefixes"
    head = parts[0]
    if not head_resolves(head):
        return False, f"head {head!r} does not resolve"
    arg = parts[1] if len(parts) > 1 else None

    if head == "cd":
        if arg is None or arg == "-":
            return True, "cd with no target / cd -"
        target = os.path.expanduser(os.path.expandvars(arg))
        if os.path.isdir(target):
            return True, "cd target exists"
        return False, f"cd target {arg!r} does not exist"

    if head == "git":
        if arg is None or arg.startswith("-"):
            return True, "git with no/option-only invocation"
        if arg in GIT_SUBCOMMANDS:
            return True, "git subcommand is real"
        return False, f"git subcommand {arg!r} is unknown"

    if head in {"docker", "podman"}:
        if arg is None or arg.startswith("-"):
            return True, f"{head} with no/option-only invocation"
        if arg in CONTAINER_SUBCOMMANDS:
            return True, f"{head} subcommand is real"
        return False, f"{head} subcommand {arg!r} is unknown"

    return True, "head resolves; no family-specific check"
