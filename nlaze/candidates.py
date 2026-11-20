"""Candidate retrieval, ordering, and Gate G1 option format (spec §1.3, §3, §3.1, §3.2).

Pool assembly and deterministic cut to <= 7 concrete, fully-formed command strings
(never flags-only, never fragments). Pool sources, in priority order (§3):
  1. User-history frequencies ($XDG_DATA_HOME/nlaze/usage.sqlite)
  2. Path adapter / zoxide resolver (fallback: complete from cwd)
  3. PATH executables + shell builtins + family subcommands (edit distance + prefix +
     keyboard-distance + frequency prior for short inputs, §1.3)
  4. Corpus priors (published prevalence; fallback for unseen inputs)

Ordering rule (§3.1, §3.2):
  - Prime candidate = harness's top belief (never duplicated).
  - Harness confident -> place prime in the LAST slot (slot 6) to exploit the v5 model's
    measured +11.70 pp recency bias (slot 0 -> slot 6, adjacent to input text).
  - Harness not confident -> rotate the order and treat the model's pick as a real signal.

Gate G1 (frozen option format):
  - Instructions: "What did the user mean to type?"
  - Question ID: "correction"
  - Criteria keys: "option_0" .. "option_6"
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from nlaze.usage import Store
from nlaze.verifier import (
    CONTAINER_SUBCOMMANDS,
    GIT_SUBCOMMANDS,
    command_is_safe_to_suggest,
    head_resolves,
    valid_as_typed,
)

INSTRUCTIONS = "What did the user mean to type?"
QID = "correction"
TRAIN_K = 7

# QWERTY keyboard neighbors for short-input and keystroke-typo scoring (§1.3).
_KEY_NEIGHBORS: dict[str, str] = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol",
    "q": "wa", "r": "edft", "s": "wedxza", "t": "rfgy", "u": "yhji",
    "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

# Published corpus prevalence prior [0.0, 0.12] so common commands beat obscure binaries.
_COMMAND_PRIOR: dict[str, float] = {
    "git": 0.12, "ls": 0.11, "cd": 0.11, "cat": 0.10, "grep": 0.10, "rg": 0.10,
    "find": 0.09, "cp": 0.09, "mv": 0.09, "rm": 0.08, "mkdir": 0.09, "rmdir": 0.08,
    "stat": 0.08, "tar": 0.08, "ip": 0.08, "docker": 0.09, "podman": 0.08,
    "python3": 0.08, "python": 0.08, "pytest": 0.07, "pip": 0.07, "npm": 0.07,
    "cargo": 0.07, "make": 0.07, "micro": 0.08, "nano": 0.08, "nvim": 0.08,
    "vim": 0.08, "less": 0.07, "head": 0.07, "tail": 0.07, "echo": 0.07,
    "pwd": 0.07, "df": 0.07, "du": 0.07, "ps": 0.07, "free": 0.07, "top": 0.06,
    "htop": 0.06, "kill": 0.06, "chmod": 0.07, "chown": 0.06, "ln": 0.06,
    "touch": 0.07, "sed": 0.07, "awk": 0.07, "sort": 0.06, "uniq": 0.06,
    "wc": 0.06, "cut": 0.06, "curl": 0.08, "wget": 0.07, "ssh": 0.08, "scp": 0.07,
    "rsync": 0.07, "systemctl": 0.08, "journalctl": 0.07, "dmesg": 0.06,
    "mount": 0.06, "umount": 0.06, "lsblk": 0.07, "blkid": 0.06, "pacman": 0.08,
    "apt": 0.07, "which": 0.07, "file": 0.07, "diff": 0.06, "man": 0.07,
    "history": 0.06, "clear": 0.06, "env": 0.06, "export": 0.06, "z": 0.07,
}

_GIT_SUB_PRIOR: dict[str, float] = {
    "status": 0.12, "commit": 0.11, "checkout": 0.10, "push": 0.10, "pull": 0.10,
    "diff": 0.10, "log": 0.10, "branch": 0.09, "stash": 0.09, "add": 0.09,
    "reset": 0.08, "rebase": 0.08, "merge": 0.08, "fetch": 0.08, "clone": 0.08,
    "switch": 0.08, "restore": 0.08, "show": 0.08, "tag": 0.07, "remote": 0.07,
    "clean": 0.06, "cherry-pick": 0.06, "blame": 0.06, "init": 0.06,
}

_CONTAINER_SUB_PRIOR: dict[str, float] = {
    "ps": 0.12, "run": 0.11, "images": 0.10, "build": 0.10, "exec": 0.10,
    "logs": 0.10, "stop": 0.09, "start": 0.09, "restart": 0.08, "rm": 0.08,
    "rmi": 0.08, "pull": 0.08, "push": 0.08, "compose": 0.09, "inspect": 0.08,
    "volume": 0.07, "network": 0.07, "system": 0.07, "stats": 0.06, "top": 0.06,
}

_CORPUS_FILLERS: tuple[str, ...] = (
    "git status",
    "ls -la",
    "pwd",
    "df -h",
    "ps aux",
    "free -h",
    "cat /etc/os-release",
    "git diff",
    "git log --oneline -n 5",
    "lsblk",
    "whoami",
    "echo ok",
)


@dataclass(frozen=True)
class ScoredCandidate:
    cmd: str
    score: float
    source: str


def damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string alignment distance (insertion, deletion, substitution, transposition)."""
    la, lb = len(a), len(b)
    if a == b:
        return 0
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
    return dp[la][lb]


def _keyboard_bonus(a: str, b: str) -> float:
    """Bonus when a single edit is an adjacent-key substitution or transposition."""
    if len(a) == len(b):
        diffs = [(ca, cb) for ca, cb in zip(a, b) if ca != cb]
        if len(diffs) == 1:
            ca, cb = diffs[0]
            if cb in _KEY_NEIGHBORS.get(ca, ""):
                return 0.08
    return 0.0


def _subsequence_match(short: str, target: str) -> bool:
    """True iff characters of `short` appear in `target` in order (e.g. mdir -> mkdir)."""
    it = iter(target)
    return all(ch in it for ch in short)


def token_similarity(query: str, target: str, prior: float = 0.0) -> float:
    """Composite similarity for command heads, subcommands, and short inputs (§1.3, §3)."""
    q = query.lower()
    t = target.lower()
    if not q or not t:
        return 0.0
    dist = damerau_levenshtein(q, t)
    max_len = max(len(q), len(t))
    base = max(0.0, 1.0 - (dist / max_len))

    bonus = prior + _keyboard_bonus(q, t)
    if t.startswith(q):
        # §1.3 prefix matching for short inputs
        bonus += 0.18 if len(q) <= 3 else 0.10
    elif len(q) <= 4 and _subsequence_match(q, t):
        bonus += 0.08
    if q[0] == t[0]:
        bonus += 0.04
    if q[-1] == t[-1]:
        bonus += 0.03
    return min(0.94, base * 0.80 + bonus)


def _candidate_ok(cmd: str, raw_input: str, cwd: Path | None = None) -> bool:
    """Every candidate must differ from raw input and pass the verifier + first-arg check."""
    s = cmd.strip()
    if not s or s == raw_input.strip():
        return False
    if not command_is_safe_to_suggest(s):
        return False
    if cwd is not None and s.startswith("cd "):
        arg = s[3:].strip()
        target = Path(os.path.expanduser(os.path.expandvars(arg)))
        if not target.is_absolute():
            target = cwd / target
        return target.is_dir()
    ok, _ = valid_as_typed(s)
    return ok


def _path_candidates(
    arg: str,
    cwd: Path,
    *,
    use_zoxide: bool = True,
) -> list[ScoredCandidate]:
    raw_arg = arg.strip()
    if not raw_arg:
        return []
    expanded = os.path.expanduser(os.path.expandvars(raw_arg))
    p = Path(expanded)
    if "/" in raw_arg:
        parent_str = os.path.dirname(expanded)
        leaf_query = os.path.basename(expanded)
        prefix_for_cmd = os.path.dirname(raw_arg)
        search_dir = Path(parent_str) if os.path.isabs(expanded) else (cwd / parent_str)
    else:
        leaf_query = raw_arg
        prefix_for_cmd = ""
        search_dir = cwd

    out: list[ScoredCandidate] = []
    if search_dir.is_dir():
        try:
            entries = sorted(e.name for e in search_dir.iterdir() if e.is_dir())
        except OSError:
            entries = []
        for name in entries:
            if name.startswith(".") and not leaf_query.startswith("."):
                continue
            sim = token_similarity(leaf_query, name, prior=0.05)
            if leaf_query.lower() in name.lower():
                sim = min(0.94, sim + 0.08)
            rel = f"{prefix_for_cmd}/{name}" if prefix_for_cmd else name
            out.append(ScoredCandidate(f"cd {rel}", sim, "path"))

    if use_zoxide and shutil.which("zoxide") is not None:
        try:
            proc = subprocess.run(
                ["zoxide", "query", "-l", "--", *raw_arg.split()],
                capture_output=True,
                text=True,
                timeout=0.03,
                check=False,
            )
            if proc.returncode == 0:
                for rank, line in enumerate(proc.stdout.splitlines()[:5]):
                    zpath = line.strip()
                    if zpath and os.path.isdir(zpath):
                        out.append(
                            ScoredCandidate(f"cd {zpath}", max(0.60, 0.88 - 0.04 * rank), "zoxide")
                        )
        except (OSError, subprocess.SubprocessError):
            pass

    out.sort(key=lambda c: (-c.score, c.cmd))
    return out


def retrieve_candidates(
    cmd_line: str,
    *,
    store: Store | None = None,
    cwd: str | Path | None = None,
    k: int = TRAIN_K,
    use_zoxide: bool = True,
) -> list[ScoredCandidate]:
    """Assemble candidate pool from history, path/zoxide, executables/families, and priors,
    then make a deterministic cut to <= k verified concrete command strings (§3)."""
    raw = cmd_line.strip()
    cwd_path = Path(cwd) if cwd is not None else Path.cwd()
    pool: dict[str, ScoredCandidate] = {}

    def _add(cand: ScoredCandidate) -> None:
        if not _candidate_ok(cand.cmd, raw, cwd=cwd_path):
            return
        prev = pool.get(cand.cmd)
        if prev is None or cand.score > prev.score:
            pool[cand.cmd] = cand

    # 1. User-history frequencies (highest priority, §3)
    if store is not None and raw:
        for corr, n_eff in store.top_corrections(raw, limit=k):
            score = min(0.995, 0.95 + min(0.045, n_eff / 400.0))
            _add(ScoredCandidate(corr, score, "history"))

    # Parse tokens while preserving exact raw argument substrings (including quotes)
    raw_tokens = raw.split()
    if raw_tokens:
        prefix = ""
        body_str = raw
        if raw_tokens[0] == "sudo" and len(raw_tokens) > 1:
            prefix = "sudo "
            body_str = raw.split(None, 1)[1]

        body_parts = body_str.split(None, 2)
        head = body_parts[0]
        rest_after_head = (" " + body_str.split(None, 1)[1]) if len(body_parts) > 1 else ""
        first_arg = body_parts[1] if len(body_parts) > 1 else None
        rest_after_sub = (" " + body_parts[2]) if len(body_parts) > 2 else ""

        # 2. Path adapter / zoxide resolver (`cd <arg>`, `z <arg>`, or `dc <arg>`)
        if head in {"cd", "z", "dc", "cdd"} and first_arg is not None:
            arg_raw = body_str.split(None, 1)[1]
            for cand in _path_candidates(arg_raw, cwd_path, use_zoxide=use_zoxide):
                _add(cand)

        # 3a. Git family (`git <sub>` or typo'd `gti <sub>`, `igt <sub>`)
        git_dist = damerau_levenshtein(head.lower(), "git")
        if (head == "git" or (git_dist <= 1 and len(head) in {2, 3, 4})) and head_resolves("git"):
            if first_arg is not None and not first_arg.startswith("-"):
                sub_q = first_arg
                head_penalty = 0.0 if head == "git" else 0.03
                for sub in GIT_SUBCOMMANDS:
                    if damerau_levenshtein(sub_q.lower(), sub) <= max(3, len(sub_q)):
                        sc = token_similarity(sub_q, sub, prior=_GIT_SUB_PRIOR.get(sub, 0.02))
                        _add(
                            ScoredCandidate(
                                f"{prefix}git {sub}{rest_after_sub}",
                                max(0.10, sc - head_penalty),
                                "family",
                            )
                        )
            else:
                _add(ScoredCandidate(f"{prefix}git{rest_after_head}", 0.90, "family"))

        # 3b. Docker / Podman family (`docker <sub>`, `podman <sub>`, or typo'd head)
        for fam in ("docker", "podman"):
            if not head_resolves(fam):
                continue
            fdist = damerau_levenshtein(head.lower(), fam)
            if head == fam or fdist <= 2:
                if first_arg is not None and not first_arg.startswith("-"):
                    sub_q = first_arg
                    head_penalty = 0.0 if head == fam else 0.04 * fdist
                    for sub in CONTAINER_SUBCOMMANDS:
                        if damerau_levenshtein(sub_q.lower(), sub) <= max(3, len(sub_q)):
                            sc = token_similarity(
                                sub_q, sub, prior=_CONTAINER_SUB_PRIOR.get(sub, 0.02)
                            )
                            _add(
                                ScoredCandidate(
                                    f"{prefix}{fam} {sub}{rest_after_sub}",
                                    max(0.10, sc - head_penalty),
                                    "family",
                                )
                            )

        # 3c. General executable / builtin matching (§1.3 short inputs + edit distance)
        for cand_head, prior in _COMMAND_PRIOR.items():
            if not head_resolves(cand_head):
                continue
            dist = damerau_levenshtein(head.lower(), cand_head)
            if len(head) <= 3:
                # §1.3 short inputs: allow prefix match or distance <= 2
                if dist > 2 and not cand_head.startswith(head.lower()):
                    continue
            else:
                if dist > 3:
                    continue
            sc = token_similarity(head, cand_head, prior=prior)
            _add(ScoredCandidate(f"{prefix}{cand_head}{rest_after_head}", sc, "exec"))

    # 3d. Global user-history popular corrections as secondary prior
    if store is not None:
        for corr, n_eff in store.popular_corrections(limit=k):
            _add(ScoredCandidate(corr, min(0.55, 0.35 + n_eff / 100.0), "history_prior"))

    # 4. Corpus priors to guarantee k distinct valid options when needed
    for idx, filler in enumerate(_CORPUS_FILLERS):
        if len(pool) >= k and idx > 3:
            break
        _add(ScoredCandidate(filler, max(0.05, 0.25 - 0.015 * idx), "prior"))

    ranked = sorted(pool.values(), key=lambda c: (-c.score, c.cmd))
    return ranked[:k]


def prime_candidate(scored: list[ScoredCandidate]) -> ScoredCandidate | None:
    """Spec §3.2: the prime candidate is the harness's top belief (never duplicated)."""
    return scored[0] if scored else None


def is_harness_confident(
    scored: list[ScoredCandidate],
    *,
    min_score: float = 0.78,
    min_margin: float = 0.10,
) -> bool:
    """Spec §3.1: True iff top candidate is clearly ahead and verifier-passed."""
    if not scored:
        return False
    top = scored[0]
    if top.score < min_score:
        return False
    if not valid_as_typed(top.cmd)[0]:
        return False
    if len(scored) == 1:
        return True
    return (top.score - scored[1].score) >= min_margin


def order_for_model(
    scored: list[ScoredCandidate],
    *,
    confident: bool | None = None,
    rotation: int = 0,
) -> list[str]:
    """Spec §3.1 ordering rule:
    - Harness confident -> place prime (top belief) in the LAST slot (slot k-1 = slot 6)
      to exploit the model's +11.70 pp recency bias; rivals sorted ascending in 0..k-2.
    - Harness not confident -> cyclic rotation by `rotation` slots (no recency rubber-stamp).
    """
    if not scored:
        return []
    if confident is None:
        confident = is_harness_confident(scored)
    if confident:
        prime = scored[0].cmd
        rivals_asc = [c.cmd for c in reversed(scored[1:])]
        return rivals_asc + [prime]
    base = [c.cmd for c in scored]
    r = rotation % len(base)
    return base[r:] + base[:r]


def rotations_for_model(
    scored: list[ScoredCandidate],
    n_rotations: int = 3,
) -> list[list[str]]:
    """Return `n_rotations` cyclic rotations for probability averaging when not confident."""
    if not scored:
        return []
    n = len(scored)
    count = min(max(1, n_rotations), n)
    # Spread rotations across the candidate list (e.g. shifts 0, 2, 4 for k=7, count=3)
    shifts = [int(round(i * n / count)) % n for i in range(count)]
    # Ensure distinct shifts
    used: list[int] = []
    for s in shifts:
        while s in used:
            s = (s + 1) % n
        used.append(s)
    return [order_for_model(scored, confident=False, rotation=s) for s in used]


def format_question(ordered_cands: list[str]) -> dict:
    """Gate G1: frozen Laya choice question format (option_0..option_{k-1})."""
    return {
        QID: {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {f"option_{i}": c for i, c in enumerate(ordered_cands)},
        }
    }
