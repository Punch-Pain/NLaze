"""NLaze v6 harness engine and per-user Unix socket daemon (spec §1–§7).

No candidate string is EVER executed by this module. The engine and daemon return
structured JSON decisions (`passthrough`, `auto`, `confirm`, `confirm_mild`,
`confirm_high`, `certify_head`, `certify_flag`); the user's shell hook renders
the prompt and executes only when authorized.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import threading
from typing import Callable

from nlaze.candidates import (
    INSTRUCTIONS,
    QID,
    ScoredCandidate,
    format_question,
    is_harness_confident,
    order_for_model,
    prime_candidate,
    retrieve_candidates,
    rotations_for_model,
)
from nlaze.danger import HIGH, MILD, classify, flag_doubt
from nlaze.lanes import Lane
from nlaze.policy import Decision, Policy, policy_tokens
from nlaze.recondition import Reconditioner
from nlaze.usage import Store
from nlaze.verifier import command_is_safe_to_suggest, is_complex_input, valid_as_typed


def default_usage_path() -> Path:
    override = os.environ.get("NLAZE_USAGE_PATH")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(xdg) / "nlaze" / "usage.sqlite"


def default_socket_path() -> Path:
    override = os.environ.get("NLAZE_SOCKET_PATH")
    if override:
        return Path(override)
    run_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/nlaze-{os.getuid()}"
    return Path(run_dir) / "nlaze" / "daemon.sock"


class HarnessEngine:
    """Core v6 decision engine around a frozen decision model (or deterministic ranker)."""

    def __init__(
        self,
        *,
        store: Store,
        policy: Policy,
        predict_fn: Callable[[dict, dict], dict] | None = None,
        use_zoxide: bool = True,
    ):
        self.store = store
        self.policy = policy
        self.reconditioner = Reconditioner(store)
        self.predict_fn = predict_fn
        self.use_zoxide = use_zoxide

    def _observe_and_tick(self, cmd_line: str) -> list[str]:
        """Increment command counter (§5.1) and apply any near-neighbor strikes (§5.2)."""
        t = self.store.tick()
        strikes = self.reconditioner.observe(cmd_line.strip(), tick=t)
        if strikes:
            self.reconditioner.apply(strikes, tick=t)
        return [s.correction for s in strikes]

    def _try_direct_serve(self, raw: str, cwd: Path) -> dict | None:
        """Spec §6: serve directly from cache in <= 5 ms without initializing the model."""
        top = self.store.top_corrections(raw, limit=1)
        if not top:
            return None
        corr, _neff = top[0]
        if not self.store.auto_eligible(raw, corr):
            return None
        ih = self.store.hash_input(raw)
        if not self.reconditioner.auto_allowed(ih, corr):
            return None
        # Rule 1: verifier is mandatory at every layer, including cache hits and direct serve.
        if not command_is_safe_to_suggest(corr) or not valid_as_typed(corr)[0]:
            return None
        if classify(corr) is not None and classify(corr) in (HIGH, MILD):
            return None
        if self.policy.check(corr) is not Decision.ALLOW:
            return None

        # Assemble fast offer list for the reconditioning ring buffer
        cands = retrieve_candidates(
            raw, store=self.store, cwd=cwd, use_zoxide=False
        )
        offered = [c.cmd for c in cands]
        if corr not in offered:
            offered = [corr] + offered[:6]
        self.reconditioner.record_offer(raw, corr, offered)
        return {
            "action": "auto",
            "lane": Lane.AUTO.value,
            "input": raw,
            "correction": corr,
            "confidence": 0.99,
            "source": "direct_serve",
            "offered": offered,
        }

    def _score_with_model(
        self, raw: str, scored: list[ScoredCandidate], confident: bool
    ) -> tuple[str, float, bool]:
        """Score candidates with `predict_fn` if attached, else use harness ranking."""
        prime = prime_candidate(scored)
        assert prime is not None
        if self.predict_fn is None:
            return prime.cmd, round(float(prime.score), 4), (not confident)

        if confident:
            ordered = order_for_model(scored, confident=True)
            q = format_question(ordered)
            res = self.predict_fn({"text": raw}, q)
            ans = res["answers"][QID]
            probs = ans["probabilities"]
            best_key = max(probs, key=lambda k: float(probs[k]))
            best_idx = int(best_key.split("_")[1])
            best_cmd = ordered[best_idx]
            conf = float(probs[best_key])
            override = best_cmd != prime.cmd
            return best_cmd, round(conf, 4), override

        # Harness not confident (§3.1) -> 3-rotation cyclic probability average
        rots = rotations_for_model(scored, n_rotations=3)
        questions = {}
        for r_idx, rot in enumerate(rots):
            questions[f"r{r_idx}"] = {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {f"option_{i}": c for i, c in enumerate(rot)},
            }
        res = self.predict_fn({"text": raw}, questions)
        agg: dict[str, float] = {c.cmd: 0.0 for c in scored}
        for r_idx, rot in enumerate(rots):
            probs = res["answers"][f"r{r_idx}"]["probabilities"]
            for i, cmd in enumerate(rot):
                agg[cmd] += float(probs[f"option_{i}"]) / len(rots)
        best_cmd = max(agg, key=lambda c: agg[c])
        conf = float(agg[best_cmd])
        return best_cmd, round(conf, 4), True

    def handle_postexec(
        self,
        cmd_line: str,
        *,
        exit_status: int,
        cwd: str | Path | None = None,
    ) -> dict:
        raw = cmd_line.strip()
        cwd_path = Path(cwd) if cwd is not None else Path.cwd()

        if not raw:
            return {"action": "passthrough", "lane": Lane.PASSTHROUGH.value, "reason": "empty"}

        strikes = self._observe_and_tick(raw)

        # Spec §1: activation on command failure only ($status != 0).
        if exit_status == 0:
            return {
                "action": "passthrough",
                "lane": Lane.PASSTHROUGH.value,
                "reason": "command succeeded",
                "strikes": strikes,
            }

        # Spec §1.2: complex inputs passthrough without loading model.
        complex_in, complex_why = is_complex_input(cmd_line)
        if complex_in:
            return {
                "action": "passthrough",
                "lane": Lane.PASSTHROUGH.value,
                "reason": f"complex input ({complex_why})",
            }

        # Spec §1 Rule 0 & §1.1: valid-as-typed commands passthrough (e.g. grep no match).
        valid_ok, valid_why = valid_as_typed(raw)
        if valid_ok and not (raw.startswith("cd ") and not (cwd_path / raw[3:].strip()).is_dir()):
            return {
                "action": "passthrough",
                "lane": Lane.PASSTHROUGH.value,
                "reason": valid_why,
            }

        # Spec §6: direct serve from cache if Auto gate is satisfied.
        direct = self._try_direct_serve(raw, cwd_path)
        if direct is not None:
            return direct

        # Spec §3: candidate retrieval and ordering.
        scored = retrieve_candidates(
            raw, store=self.store, cwd=cwd_path, use_zoxide=self.use_zoxide
        )
        if not scored:
            return {
                "action": "passthrough",
                "lane": Lane.PASSTHROUGH.value,
                "reason": "no valid candidates",
            }

        confident = is_harness_confident(scored)
        best_cmd, conf, force_confirm = self._score_with_model(raw, scored, confident)
        offered = [c.cmd for c in scored]

        # Mandatory verifier check on chosen candidate (§1 Rule 1).
        if not command_is_safe_to_suggest(best_cmd):
            return {
                "action": "passthrough",
                "lane": Lane.PASSTHROUGH.value,
                "reason": "chosen candidate rejected by verifier",
            }

        self.reconditioner.record_offer(raw, best_cmd, offered)

        # Spec §2 & §4.2: Precedence: danger table > deny list > allowlist > confidence.
        sev = classify(best_cmd)
        if sev is HIGH:
            return {
                "action": "confirm_high",
                "lane": Lane.DANGER_HIGH.value,
                "input": raw,
                "correction": best_cmd,
                "confidence": conf,
                "reason": sev.detail,
                "offered": offered,
            }
        if sev is MILD:
            return {
                "action": "confirm_mild",
                "lane": Lane.DANGER_MILD.value,
                "input": raw,
                "correction": best_cmd,
                "confidence": conf,
                "reason": sev.detail,
                "offered": offered,
            }

        head, _pos, flags = policy_tokens(best_cmd)
        dec = self.policy.check(best_cmd)
        if dec is Decision.DENY:
            return {
                "action": "confirm",
                "lane": Lane.CONFIRM.value,
                "input": raw,
                "correction": best_cmd,
                "confidence": conf,
                "reason": "on deny list",
                "offered": offered,
            }
        if dec is Decision.HEAD_PROMPT:
            return {
                "action": "certify_head",
                "lane": Lane.CONFIRM.value,
                "input": raw,
                "correction": best_cmd,
                "confidence": conf,
                "head": head,
                "offered": offered,
            }
        if dec is Decision.FLAG_PROMPT:
            flagged = next((f for f in flags if flag_doubt(f)[0]), flags[0] if flags else "")
            return {
                "action": "certify_flag",
                "lane": Lane.CONFIRM.value,
                "input": raw,
                "correction": best_cmd,
                "confidence": conf,
                "head": head,
                "flag": flagged,
                "offered": offered,
            }

        # Decision.ALLOW -> check Auto eligibility (§5.1, §5.2)
        ih = self.store.hash_input(raw)
        auto_ok = (
            (not force_confirm)
            and self.store.auto_eligible(raw, best_cmd)
            and self.reconditioner.auto_allowed(ih, best_cmd)
            and conf >= 0.85
        )
        lane = Lane.AUTO if auto_ok else Lane.CONFIRM
        return {
            "action": "auto" if auto_ok else "confirm",
            "lane": lane.value,
            "input": raw,
            "correction": best_cmd,
            "confidence": conf,
            "source": "model" if self.predict_fn is not None else "harness",
            "offered": offered,
        }

    def handle_feedback(
        self,
        *,
        input_text: str,
        correction: str,
        accepted: bool,
        certify_kind: str | None = None,
        head: str | None = None,
        flag: str | None = None,
    ) -> dict:
        """Record user response to a Confirm or Certification prompt (§4.1, §5.1)."""
        # Spec §5.1 Rule 2: danger-confirm answers are not written to popularity store.
        is_danger = classify(correction) in (HIGH, MILD)

        if certify_kind == "head" and head:
            if accepted:
                self.policy.allow_head(head)
        elif certify_kind == "flag" and head and flag:
            if accepted:
                self.policy.allow_flag(head, flag)
            else:
                self.policy.deny(head, flag)

        if not is_danger:
            if accepted:
                self.store.record_accept(input_text, correction)
            else:
                self.store.record_reject(input_text, correction)

        return {"ok": True, "accepted": accepted}

    def handle_request(self, req: dict) -> dict:
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "op": "pong"}
        if op == "postexec":
            return self.handle_postexec(
                req.get("cmd", ""),
                exit_status=int(req.get("status", 1)),
                cwd=req.get("cwd"),
            )
        if op == "feedback":
            return self.handle_feedback(
                input_text=req.get("input", ""),
                correction=req.get("correction", ""),
                accepted=bool(req.get("accepted", False)),
                certify_kind=req.get("certify_kind"),
                head=req.get("head"),
                flag=req.get("flag"),
            )
        return {"ok": False, "error": f"unknown op {op!r}"}


class DaemonServer:
    """Per-user Unix domain socket daemon with 0700/0600 permissions and UID verification (§7)."""

    def __init__(self, sock_path: str | Path, engine: HarnessEngine):
        self.sock_path = Path(sock_path).expanduser()
        self.engine = engine
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._uid = os.getuid()

    def _prepare_socket(self) -> socket.socket:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.sock_path.parent, 0o700)
        if self.sock_path.exists():
            self.sock_path.unlink()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(self.sock_path))
        os.chmod(self.sock_path, 0o600)
        s.listen(16)
        s.settimeout(0.2)
        return s

    def _verify_peer_uid(self, conn: socket.socket) -> bool:
        so_peercred = getattr(socket, "SO_PEERCRED", None)
        if so_peercred is None:
            return True
        try:
            creds = conn.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", creds)
            return uid == self._uid
        except OSError:
            return False

    def serve_forever(self) -> None:
        if self._sock is None:
            self._sock = self._prepare_socket()
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                if not self._verify_peer_uid(conn):
                    continue
                data = b""
                while not data.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    continue
                try:
                    req = json.loads(data.decode("utf-8"))
                    resp = self.engine.handle_request(req)
                except Exception as exc:  # noqa: BLE001
                    resp = {"ok": False, "error": str(exc)}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

    def start_background(self) -> None:
        self._sock = self._prepare_socket()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.sock_path.exists():
            try:
                self.sock_path.unlink()
            except OSError:
                pass


def query_daemon_or_local(
    req: dict,
    *,
    sock_path: str | Path | None = None,
    fallback_engine: HarnessEngine | None = None,
) -> dict:
    """Send request to the per-user Unix socket daemon; fallback to in-process engine."""
    target = Path(sock_path) if sock_path is not None else default_socket_path()
    if target.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(str(target))
                s.sendall((json.dumps(req) + "\n").encode("utf-8"))
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                if buf:
                    return json.loads(buf.decode("utf-8"))
        except OSError:
            pass
    if fallback_engine is not None:
        return fallback_engine.handle_request(req)
    raise RuntimeError(f"daemon socket {target} unavailable and no fallback_engine provided")
