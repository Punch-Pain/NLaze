"""Phase 1 — lane classifier for the three-lane harness.

Lane enum order = precedence: PASSTHROUGH first, then DANGER_HIGH/MILD, then Auto/Confirm.
Anything the pre-check cannot classify lands in CONFIRM (never AUTO).
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from nlaze.verifier import head_resolves
from nlaze.danger import classify, HIGH, MILD

class Lane(Enum):
    PASSTHROUGH = "passthrough"   # already-correct input; nothing initializes
    CONFIRM     = "confirm"       # needs explicit user approval
    DANGER_MILD = "danger_mild"   # confirm UX, ordinary acknowledgment
    DANGER_HIGH = "danger_high"   # typed acknowledgment / hold-Enter; never Auto
    AUTO        = "auto"          # clean history → may serve without prompting

@dataclass(frozen=True)
class Verdict:
    lane: Lane
    reason: str
    detail: str = ""

def classify_input(cmd_line: str, known_correct: bool = False) -> Verdict:
    line = cmd_line.strip()
    if not line:
        return Verdict(Lane.PASSTHROUGH, "empty input → passthrough")
    sev = classify(line)
    if sev is HIGH:
        return Verdict(Lane.DANGER_HIGH, "matches high-danger pattern", sev.detail)
    if sev is MILD:
        return Verdict(Lane.DANGER_MILD, "matches mild-danger (privileged, reversible)", sev.detail)
    # valid as typed? deterministic, cheap, narrow
    parts = line.split()
    if head_resolves(parts[0]):
        return Verdict(Lane.PASSTHROUGH, "head resolves on PATH/builtin — valid as typed")
    return Verdict(Lane.CONFIRM, "unknown command — needs user intent elicitation")

def classify_candidate(cmd_line: str, *, history_pairs_clean: int = 0) -> Verdict:
    """Candidate verdict for a *suggested* correction (never called on the raw input)."""
    sev = classify(cmd_line)
    if sev is HIGH:
        return Verdict(Lane.DANGER_HIGH, "dangerous candidate — warning panel only")
    if sev is MILD:
        return Verdict(Lane.DANGER_MILD, "privileged-read candidate — ordinary confirm")
    return Verdict(Lane.AUTO if history_pairs_clean >= 17 else Lane.CONFIRM,
                   "auto-eligible by usage" if history_pairs_clean >= 17 else "no usage evidence — confirm")
