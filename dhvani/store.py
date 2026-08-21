"""SQLite-backed content-addressed cache, job state, and spend ledger.

Idempotency is enforced by the schema — PRIMARY KEY (segment_id, tier) —
rather than by application logic.
"""

import json
import sqlite3
import time

from dhvani.config import MAX_SPEND_USD

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
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
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

    def put_hypothesis(self, segment_id, tier, text, signals, cost_usd) -> bool:
        """Returns True if newly inserted, False if already present (no-op)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO hypotheses "
            "(segment_id, tier, text, signals_json, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (segment_id, tier, text, json.dumps(signals, sort_keys=True),
             cost_usd, int(time.time())),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_hypothesis(self, segment_id, tier):
        row = self.conn.execute(
            "SELECT text, signals_json, cost_usd FROM hypotheses "
            "WHERE segment_id = ? AND tier = ?",
            (segment_id, tier),
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
        """Fail closed before a paid call."""
        projected = self.total_spend() + cost_usd
        if projected > MAX_SPEND_USD:
            raise BudgetExceeded(
                f"call costing ${cost_usd:.4f} would exceed ceiling: "
                f"${projected:.4f} > ${MAX_SPEND_USD:.2f}"
            )
