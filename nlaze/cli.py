"""nlaze CLI — query the gate, run the postexec hook/daemon, and audit per-user policy.

Usage:
  nlaze check "<command>"                 # classify a command line (dry-run, never executes)
  nlaze --json check "<command>"          # machine-readable
  nlaze suggest "<command>"               # retrieve and rank candidates (dry-run)
  nlaze --json hook-postexec --status <rc> [--cwd <dir>] -- "<command>"
  nlaze feedback --input "<inp>" --correction "<corr>" (--accept | --reject)
  nlaze daemon [--socket <path>] [--model <dir>]
  nlaze fish-init                         # emit fish shell hook script
  nlaze purge                             # wipe per-user usage.sqlite learning state
  nlaze allowlist list | add <head> [--flag <f>] | remove <head> [--flag <f>]
  nlaze danger    list | add <head> --flag <f>   | remove <head> --flag <f>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from nlaze.daemon import (
    DaemonServer,
    HarnessEngine,
    default_socket_path,
    default_usage_path,
    query_daemon_or_local,
)
from nlaze.lanes import Lane, classify_input
from nlaze.policy import Policy
from nlaze.usage import Store


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


def _make_local_engine() -> HarnessEngine:
    return HarnessEngine(
        store=Store(default_usage_path()),
        policy=Policy(default_policy_path()),
    )


def main() -> int:
    ap = argparse.ArgumentParser(prog="nlaze")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="classify a command line (dry-run, never executes)")
    c.add_argument("line")

    sg = sub.add_parser("suggest", help="propose a correction for a failed command (dry-run)")
    sg.add_argument("line")
    sg.add_argument("--cwd", default=None)

    hp = sub.add_parser("hook-postexec", help="fish_postexec hook entrypoint")
    hp.add_argument("--status", type=int, required=True)
    hp.add_argument("--cwd", default=None)
    hp.add_argument("line")

    fb = sub.add_parser("feedback", help="record user accept/reject/certification feedback")
    fb.add_argument("--input", required=True, dest="input_text")
    fb.add_argument("--correction", required=True)
    grp = fb.add_mutually_exclusive_group(required=True)
    grp.add_argument("--accept", action="store_true")
    grp.add_argument("--reject", action="store_true")
    fb.add_argument("--certify", choices=["head", "flag"], default=None)
    fb.add_argument("--head", default=None)
    fb.add_argument("--flag", default=None)

    dm = sub.add_parser("daemon", help="run per-user Unix socket daemon")
    dm.add_argument("--socket", default=None)
    dm.add_argument("--model", default=None)

    sub.add_parser("fish-init", help="print fish shell integration script")
    sub.add_parser("purge", help="wipe per-user usage.sqlite learning store")

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

    if ns.cmd in {"suggest", "hook-postexec"}:
        status = 1 if ns.cmd == "suggest" else ns.status
        req = {"op": "postexec", "cmd": ns.line, "status": status, "cwd": ns.cwd}
        resp = query_daemon_or_local(req, fallback_engine=_make_local_engine())
        if ns.json:
            print(json.dumps(resp))
        else:
            if resp.get("action") == "passthrough":
                print("passthrough")
            else:
                print(f"{resp.get('action')}: {resp.get('correction')}")
        return 0

    if ns.cmd == "feedback":
        req = {
            "op": "feedback",
            "input": ns.input_text,
            "correction": ns.correction,
            "accepted": bool(ns.accept),
            "certify_kind": ns.certify,
            "head": ns.head,
            "flag": ns.flag,
        }
        resp = query_daemon_or_local(req, fallback_engine=_make_local_engine())
        if ns.json:
            print(json.dumps(resp))
        return 0

    if ns.cmd == "fish-init":
        fish_file = Path(__file__).resolve().parent / "fish" / "nlaze.fish"
        print(fish_file.read_text(encoding="utf-8"), end="")
        return 0

    if ns.cmd == "purge":
        up = default_usage_path()
        if up.exists():
            up.unlink()
        print("purged usage store")
        return 0

    if ns.cmd == "daemon":
        predict_fn = None
        if ns.model:
            import laya
            agent = laya.Agent(str(ns.model))
            predict_fn = agent.predict
        eng = HarnessEngine(
            store=Store(default_usage_path()),
            policy=Policy(default_policy_path()),
            predict_fn=predict_fn,
        )
        sock = Path(ns.socket) if ns.socket else default_socket_path()
        srv = DaemonServer(sock, eng)
        srv.serve_forever()
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
