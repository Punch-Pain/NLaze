"""Near-neighbor reconditioning (spec §5.2).

The Auto path has no explicit reject: the command runs and the user never chooses. The only
feedback channel is **what the user runs next.**

The harness stores the **offer** — the candidate set it presented for an input — in a short
ring buffer. A strike is counted when the user runs an alternative from a recent offer for
the same input:

    strike = within W commands of an auto-run of A for input X,
             the user runs B, where B was in the offer for X and B != A

Strikes are **weak**: they demote, they do not reset. 1 strike AUTO -> CONFIRM; 2 -> UNKNOWN.
Strikes decay with the same command-tick alpha as the usage store, so an old strike eventually
forgets itself.
"""
from __future__ import annotations

import json
from typing import NamedTuple

from nlaze.usage import ALPHA, Store

DDL = """
CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY,
    input_hash TEXT NOT NULL,
    correction TEXT NOT NULL,
    offered TEXT NOT NULL,      -- JSON array of the candidate set presented
    tick INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS strikes (
    input_hash TEXT NOT NULL,
    correction TEXT NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    last_tick INTEGER NOT NULL,
    PRIMARY KEY (input_hash, correction)
);
"""


class Strike(NamedTuple):
    input_hash: str
    correction: str


class Reconditioner:
    def __init__(self, store: Store, window: int = 10, max_offers: int = 64):
        self.store = store
        self.window = window
        self.max_offers = max_offers
        self.db = store.db
        self.db.executescript(DDL)

    # --- offer ring buffer --------------------------------------------------

    def record_offer(self, input_text: str, correction: str, offered: list[str],
                     tick: int | None = None) -> None:
        t = self.store.tick_now() if tick is None else tick
        ih = self.store.hash_input(input_text)
        self.db.execute(
            "INSERT INTO offers(input_hash, correction, offered, tick) VALUES (?,?,?,?)",
            (ih, correction, json.dumps(sorted(set(offered))), t))
        self.db.commit()
        self._trim()

    def _trim(self) -> None:
        n = self.db.execute("SELECT COUNT(*) FROM offers").fetchone()[0]
        if n > self.max_offers:
            self.db.execute(
                "DELETE FROM offers WHERE id IN "
                "(SELECT id FROM offers ORDER BY id ASC LIMIT ?)",
                (n - self.max_offers,))
            self.db.commit()

    def offer_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM offers").fetchone()[0]

    def _recent_offers(self, tick: int) -> list[tuple[str, str, list[str]]]:
        rows = self.db.execute(
            "SELECT input_hash, correction, offered FROM offers WHERE tick >= ? ORDER BY id DESC",
            (tick - self.window,)).fetchall()
        return [(r[0], r[1], json.loads(r[2])) for r in rows]

    # --- strikes ------------------------------------------------------------

    def observe(self, command: str, tick: int | None = None) -> list[Strike]:
        """Which recent auto-runs does this command contradict?"""
        t = self.store.tick_now() if tick is None else tick
        out: list[Strike] = []
        seen: set[tuple[str, str]] = set()
        for ih, corr, offered in self._recent_offers(t):
            if command in offered and command != corr and (ih, corr) not in seen:
                seen.add((ih, corr))
                out.append(Strike(ih, corr))
        return out

    def apply(self, strikes: list[Strike], tick: int | None = None) -> None:
        t = self.store.tick_now() if tick is None else tick
        for s in strikes:
            row = self.db.execute(
                "SELECT n FROM strikes WHERE input_hash=? AND correction=?",
                (s.input_hash, s.correction)).fetchone()
            n = (row[0] + 1) if row else 1
            self.db.execute(
                "INSERT INTO strikes(input_hash, correction, n, last_tick) VALUES (?,?,?,?) "
                "ON CONFLICT(input_hash, correction) DO UPDATE SET n=excluded.n, "
                "last_tick=excluded.last_tick",
                (s.input_hash, s.correction, n, t))
            if n >= 2:
                # 2 strikes -> UNKNOWN: wipe the pair's evidence entirely.
                self.store.reset_pair(s.input_hash, s.correction)
        self.db.commit()

    def strike_count(self, input_hash: str, correction: str, tick: int | None = None) -> float:
        t = self.store.tick_now() if tick is None else tick
        row = self.db.execute(
            "SELECT n, last_tick FROM strikes WHERE input_hash=? AND correction=?",
            (input_hash, correction)).fetchone()
        if not row:
            return 0.0
        return row[0] * (ALPHA ** (t - row[1]))

    def auto_allowed(self, input_hash: str, correction: str, tick: int | None = None) -> bool:
        """A pair with an active strike is demoted out of Auto."""
        return self.strike_count(input_hash, correction, tick) < 1.0
