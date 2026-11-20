"""Usage store — per-user pair memory, command-tick decay. Pure SQLite, zero net."""
from __future__ import annotations
import sqlite3, time, hashlib
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS pairs (
    id INTEGER PRIMARY KEY,
    input_hash TEXT NOT NULL,
    correction TEXT NOT NULL,
    n_eff REAL NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    rejected INTEGER NOT NULL DEFAULT 0,
    first_tick INTEGER NOT NULL,
    last_tick INTEGER NOT NULL,
    quarantined_until_tick INTEGER NOT NULL DEFAULT 0,
    ambiguous INTEGER NOT NULL DEFAULT 0,
    UNIQUE(input_hash, correction)
);
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY, v INTEGER NOT NULL
);
"""

ALPHA = 0.99931               # per-command-tick decay
AUTO_NEFF = 16.6            # ≈17 accepts
REJECT_HALVE = 0.5

def _d(n_eff: float, last_tick: int, now_tick: int) -> float:
    return n_eff * (ALPHA ** max(0, now_tick - last_tick))

class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.executescript(DDL)

    def tick(self) -> int:
        """Every executed command (passthrough included) bumps the user counter."""
        row = self.db.execute("SELECT v FROM meta WHERE k='tick'").fetchone()
        t = (row[0] if row else 0) + 1
        self.db.execute("INSERT INTO meta(k,v) VALUES('tick',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (t,))
        self.db.commit()
        return t

    def tick_now(self) -> int:
        row = self.db.execute("SELECT v FROM meta WHERE k='tick'").fetchone()
        return row[0] if row else 0

    def hash_input(self, s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def record_accept(self, inp: str, correction: str) -> None:
        t = self.tick()
        ih = self.hash_input(inp)
        row = self.db.execute("SELECT id, n_eff, accepted, first_tick, last_tick FROM pairs WHERE input_hash=? AND correction=?", (ih, correction)).fetchone()
        if row:
            pid, n_eff, accepted, first_tick, last_tick = row
            n_eff = _d(n_eff, last_tick, t) + 1.0
            self.db.execute("UPDATE pairs SET n_eff=?, accepted=accepted+1, last_tick=? WHERE id=?", (n_eff, t, pid))
        else:
            self.db.execute("INSERT INTO pairs(input_hash, correction, n_eff, accepted, first_tick, last_tick) VALUES (?,?,?,?,?,?)",
                            (ih, correction, 1.0, 1, t, t))
        self.db.commit()

    def record_reject(self, inp: str, correction: str) -> None:
        t = self.tick()
        ih = self.hash_input(inp)
        row = self.db.execute("SELECT id, n_eff, rejected, last_tick FROM pairs WHERE input_hash=? AND correction=?", (ih, correction)).fetchone()
        if row:
            pid, n_eff, rejected, last_tick = row
            # Spec §5.1: 1st reject halves decayed n_eff; 2nd+ reject resets n_eff to 0 universally.
            new_neff = 0.0 if rejected >= 1 else (_d(n_eff, last_tick, t) * REJECT_HALVE)
            self.db.execute("UPDATE pairs SET n_eff=?, rejected=rejected+1, last_tick=? WHERE id=?", (new_neff, t, pid))
        else:
            self.db.execute("INSERT INTO pairs(input_hash, correction, n_eff, rejected, first_tick, last_tick, quarantined_until_tick) VALUES (?,?,?,?,?,?,?)",
                            (ih, correction, 0.0, 1, t, t, t + 30))
        self.db.commit()

    def auto_eligible(self, inp: str, correction: str) -> bool:
        ih = self.hash_input(inp)
        row = self.db.execute("SELECT n_eff, rejected, first_tick, last_tick, quarantined_until_tick FROM pairs WHERE input_hash=? AND correction=?", (ih, correction)).fetchone()
        if not row:
            return False
        n_eff, rejected, first_tick, last_tick, quar = row
        t = self.tick_now()
        n_eff = _d(n_eff, last_tick, t)
        span = t - first_tick
        return (n_eff >= AUTO_NEFF and rejected == 0 and t >= quar and span >= 100)

    def quarantined(self, inp: str, correction: str) -> bool:
        ih = self.hash_input(inp)
        row = self.db.execute("SELECT quarantined_until_tick FROM pairs WHERE input_hash=? AND correction=?", (ih, correction)).fetchone()
        return bool(row and self.tick_now() < row[0])

    def reset_pair(self, input_hash: str, correction: str) -> None:
        """Wipe a pair's evidence (2 strikes -> UNKNOWN, spec §5.2)."""
        self.db.execute("DELETE FROM pairs WHERE input_hash=? AND correction=?",
                        (input_hash, correction))
        self.db.commit()

    def top_corrections(self, inp: str, limit: int = 3) -> list[tuple[str, float]]:
        """Spec §3: user-history frequencies for an exact input, decayed to current tick."""
        ih = self.hash_input(inp)
        t = self.tick_now()
        rows = self.db.execute(
            "SELECT correction, n_eff, last_tick, quarantined_until_tick FROM pairs WHERE input_hash=?",
            (ih,),
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for corr, n_eff, last_tick, quar in rows:
            if t < quar:
                continue
            decayed = _d(n_eff, last_tick, t)
            if decayed > 0.0:
                scored.append((corr, decayed))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:limit]

    def popular_corrections(self, limit: int = 7) -> list[tuple[str, float]]:
        """Spec §3: user-history global frequency prior across inputs."""
        t = self.tick_now()
        rows = self.db.execute(
            "SELECT correction, n_eff, last_tick, quarantined_until_tick FROM pairs"
        ).fetchall()
        totals: dict[str, float] = {}
        for corr, n_eff, last_tick, quar in rows:
            if t < quar:
                continue
            decayed = _d(n_eff, last_tick, t)
            if decayed > 0.0:
                totals[corr] = totals.get(corr, 0.0) + decayed
        ranked = sorted(totals.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:limit]

