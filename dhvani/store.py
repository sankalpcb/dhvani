"""SQLite-backed content-addressed cache, job state, and spend ledger.

Idempotency is enforced by the schema — PRIMARY KEY (segment_id, tier) —
rather than by application logic.

FIX ROUND 2 (C1): the MAX_SPEND_USD ceiling is now enforced by a single
atomic statement, reserve_spend(), not by check_budget() followed by
record_spend(). Those are two separate autocommitted transactions, so two
Store handles on the same DB file could both read the same stale total,
both conclude there was room, and both then record — leaving the ledger
above the ceiling with neither call having raised. dhvani.cli's --db
defaults to a fixed shared path, so two concurrent runs share exactly that
file. See tests/test_store.py::test_two_step_check_then_record_is_not_atomic,
which pins the broken sequence, and the concurrency test beside it.

FIX ROUND 2 (I2/I3): the hypothesis cache key is no longer the bare tier.
The `tier` column now carries a composite built by
ids.hypothesis_key(tier, variant_key) — tier + POLICY_ID + a backend
variant string. Byte-identical PCM decoded with a different lang or
model_id is a different hypothesis and must not collide, and bumping
POLICY_ID must actually invalidate what is cached. The composite still
lives in the same column, so PRIMARY KEY (segment_id, tier) keeps
enforcing idempotency unchanged.
"""

import json
import sqlite3
import time

from dhvani.config import MAX_SPEND_USD
from dhvani.ids import hypothesis_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
  segment_id   TEXT PRIMARY KEY,
  source_id    TEXT NOT NULL,
  t_start_ms   INTEGER NOT NULL,
  t_end_ms     INTEGER NOT NULL,
  duration_ms  INTEGER NOT NULL,
  lang_hint    TEXT,
  created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS hypotheses (
  segment_id   TEXT NOT NULL,
  tier         TEXT NOT NULL,
  text         TEXT NOT NULL,
  signals_json TEXT NOT NULL,
  cost_usd     REAL NOT NULL,
  created_at   INTEGER NOT NULL,
  PRIMARY KEY (segment_id, tier)
);

CREATE TABLE IF NOT EXISTS spend (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  tier       TEXT NOT NULL,
  cost_usd   REAL NOT NULL,
  created_at INTEGER NOT NULL
);
"""


class BudgetExceeded(RuntimeError):
    """Raised before any paid call that would breach MAX_SPEND_USD."""


class Store:
    def __init__(self, path: str, timeout: float = 30.0):
        # timeout is SQLite's busy timeout: when another handle holds the
        # write lock (see reserve_spend), block and retry for this long
        # rather than failing immediately with "database is locked". The
        # ceiling is enforced by serializing writers, so writers must be
        # willing to wait for each other.
        self.conn = sqlite3.connect(path, timeout=timeout)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.conn.close()
        return False

    def put_segment(self, segment_id, source_id, t_start_ms, t_end_ms, lang_hint=None):
        self.conn.execute(
            "INSERT OR IGNORE INTO segments "
            "(segment_id, source_id, t_start_ms, t_end_ms, duration_ms, lang_hint, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (segment_id, source_id, t_start_ms, t_end_ms,
             t_end_ms - t_start_ms, lang_hint, int(time.time())),
        )
        self.conn.commit()

    def put_hypothesis(self, segment_id, tier, text, signals, cost_usd,
                       variant_key: str = "") -> bool:
        """Returns True if newly inserted, False if already present (no-op).

        variant_key identifies the backend configuration that produced this
        text (lang, model_id, recognizer). The store — not the caller —
        folds it and POLICY_ID into the stored key, so no caller can
        forget to and silently reintroduce the collision.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO hypotheses "
            "(segment_id, tier, text, signals_json, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (segment_id, hypothesis_key(tier, variant_key), text,
             json.dumps(signals, sort_keys=True), cost_usd, int(time.time())),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_hypothesis(self, segment_id, tier, variant_key: str = ""):
        row = self.conn.execute(
            "SELECT text, signals_json, cost_usd FROM hypotheses "
            "WHERE segment_id = ? AND tier = ?",
            (segment_id, hypothesis_key(tier, variant_key)),
        ).fetchone()
        if row is None:
            return None
        return {
            "text": row["text"],
            "signals": json.loads(row["signals_json"]),
            "cost_usd": row["cost_usd"],
        }

    def record_spend(self, tier: str, cost_usd: float) -> None:
        self.conn.execute(
            "INSERT INTO spend (tier, cost_usd, created_at) VALUES (?, ?, ?)",
            (tier, cost_usd, int(time.time())),
        )
        self.conn.commit()

    def total_spend(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(cost_usd), 0.0) AS t FROM spend").fetchone()
        return float(row["t"])

    def check_budget(self, cost_usd: float) -> None:
        """Fail closed before a paid call.

        NOT ATOMIC with record_spend(): this reads the ledger in its own
        transaction, so a second handle can read the same total before
        this one records anything. Kept as a public read-only predicate
        (callers that only want to ask "would this fit?" without
        committing to it), but anything that is about to actually spend
        money must call reserve_spend() instead.
        """
        projected = self.total_spend() + cost_usd
        if projected > MAX_SPEND_USD:
            raise BudgetExceeded(
                f"call costing ${cost_usd:.4f} would exceed ceiling: "
                f"${projected:.4f} > ${MAX_SPEND_USD:.2f}"
            )

    def reserve_spend(self, tier: str, cost_usd: float) -> None:
        """Atomically check the ceiling and record the spend, or raise.

        This is the only safe way to authorize a paid call. The check and
        the insert are ONE SQL statement, so SQLite evaluates the running
        total and appends the row under a single write lock: no other
        connection can commit a spend row between the two halves. A
        concurrent writer either waits for this statement (up to the
        connection's busy timeout) or is serialized after it, and then
        sees this row in its own SUM.

        rowcount == 0 means the WHERE clause rejected the reservation —
        the projected total would have exceeded MAX_SPEND_USD — and
        nothing was written.

        Boundary: projected == MAX_SPEND_USD is allowed (`<=`), matching
        check_budget's strict `>`. Spending the budget down to exactly the
        ceiling is legal; a hair over is not.
        """
        cur = self.conn.execute(
            "INSERT INTO spend (tier, cost_usd, created_at) "
            "SELECT ?, ?, ? "
            "WHERE (SELECT COALESCE(SUM(cost_usd), 0.0) FROM spend) + ? <= ?",
            (tier, cost_usd, int(time.time()), cost_usd, MAX_SPEND_USD),
        )
        self.conn.commit()
        if cur.rowcount != 1:
            # Nothing was inserted, so total_spend() below is the true
            # pre-reservation total and the message is accurate.
            projected = self.total_spend() + cost_usd
            raise BudgetExceeded(
                f"call costing ${cost_usd:.4f} would exceed ceiling: "
                f"${projected:.4f} > ${MAX_SPEND_USD:.2f}"
            )
