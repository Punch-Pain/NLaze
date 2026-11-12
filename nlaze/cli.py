"""nlaze CLI — minimal entrypoint so the fish hook / terminal can query the gate.

Usage:
  nlaze check "<command>"   # print decision: passthrough | confirm | danger_mild | danger_high
  nlaze --json check "<command>"  # machine-readable
"""
from __future__ import annotations
import argparse
import json
import sys

from nlaze.lanes import classify_input, Lane

def _hint(line: str) -> str:
    v = classify_input(line)
    return {
        Lane.PASSTHROUGH: "ok — passthrough, nothing runs",
        Lane.CONFIRM:     "needs confirmation",
        Lane.DANGER_MILD: "confirm — privileged but reversible",
        Lane.DANGER_HIGH: "refusal-tier warning — explicit acknowledgment required",
    }[v.lane]

def main() -> int:
    ap = argparse.ArgumentParser(prog="nlaze")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="classify a command line (dry-run, never executes)")
    c.add_argument("line")
    ns = ap.parse_args()
    if ns.cmd != "check":
        ap.print_help()
        return 2
    v = classify_input(ns.line)
    data = {"lane": v.lane.value, "reason": v.reason, "hint": _hint(ns.line)}
    if ns.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data["hint"])
        print(f"  lane: {v.lane.value}  ({v.reason})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
