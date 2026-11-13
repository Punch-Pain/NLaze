"""nlaze CLI — query the gate and audit the per-user policy.

Usage:
  nlaze check "<command>"                 # classify a command line (dry-run, never executes)
  nlaze --json check "<command>"          # machine-readable
  nlaze allowlist list                    # show certified heads and flags
  nlaze allowlist add <head> [--flag <f>]
  nlaze allowlist remove <head> [--flag <f>]
  nlaze danger list                       # show the deny list
  nlaze danger add <head> --flag <f>
  nlaze danger remove <head> --flag <f>
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from nlaze.lanes import classify_input, Lane
from nlaze.policy import Policy


def default_policy_path() -> Path:
    """$XDG_DATA_HOME/nlaze/policy.sqlite (spec §7). NLAZE_POLICY_PATH overrides (tests)."""
    override = os.environ.get("NLAZE_POLICY_PATH")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(xdg) / "nlaze" / "policy.sqlite"


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

    al = sub.add_parser("allowlist", help="audit the allowlist")
    al_sub = al.add_subparsers(dest="al_cmd")
    al_sub.add_parser("list")
    al_add = al_sub.add_parser("add")
    al_add.add_argument("head")
    al_add.add_argument("--flag")
    al_rm = al_sub.add_parser("remove")
    al_rm.add_argument("head")
    al_rm.add_argument("--flag")

    dn = sub.add_parser("danger", help="audit the deny list")
    dn_sub = dn.add_subparsers(dest="dn_cmd")
    dn_sub.add_parser("list")
    dn_add = dn_sub.add_parser("add")
    dn_add.add_argument("head")
    dn_add.add_argument("--flag", required=True)
    dn_rm = dn_sub.add_parser("remove")
    dn_rm.add_argument("head")
    dn_rm.add_argument("--flag", required=True)

    ns = ap.parse_args()

    if ns.cmd == "check":
        v = classify_input(ns.line)
        data = {"lane": v.lane.value, "reason": v.reason, "hint": _hint(ns.line)}
        if ns.json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print(data["hint"])
            print(f"  lane: {v.lane.value}  ({v.reason})")
        return 0

    if ns.cmd == "allowlist":
        return _allowlist(ns)
    if ns.cmd == "danger":
        return _danger(ns)

    ap.print_help()
    return 2


def _allowlist(ns) -> int:
    p = Policy(default_policy_path())
    if ns.al_cmd == "list":
        heads, flags = p.list_allow()
        print("heads:")
        for h in heads:
            print(f"  {h}")
        print("flags:")
        for h, f in flags:
            print(f"  {h} {f}")
        return 0
    if ns.al_cmd == "add":
        if ns.flag:
            p.allow_flag(ns.head, ns.flag)
            print(f"allowlisted flag: {ns.head} {ns.flag}")
        else:
            p.allow_head(ns.head)
            print(f"allowlisted head: {ns.head}")
        return 0
    if ns.al_cmd == "remove":
        if ns.flag:
            p.remove_allow_flag(ns.head, ns.flag)
            print(f"removed flag: {ns.head} {ns.flag}")
        else:
            p.remove_allow_head(ns.head)
            print(f"removed head: {ns.head}")
        return 0
    print("usage: nlaze allowlist list|add|remove ...")
    return 2


def _danger(ns) -> int:
    p = Policy(default_policy_path())
    if ns.dn_cmd == "list":
        entries = p.list_deny()
        if not entries:
            print("(empty)")
        for h, f in entries:
            print(f"  {h} {f}")
        return 0
    if ns.dn_cmd == "add":
        p.deny(ns.head, ns.flag)
        print(f"denied: {ns.head} {ns.flag}")
        return 0
    if ns.dn_cmd == "remove":
        p.remove_deny(ns.head, ns.flag)
        print(f"removed deny: {ns.head} {ns.flag}")
        return 0
    print("usage: nlaze danger list|add|remove ...")
    return 2


if __name__ == "__main__":
    sys.exit(main())
