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

CREATE TABLE IF NOT EXISTS jobs (
  job_id       TEXT PRIMARY KEY,
  source_id    TEXT NOT NULL,
  tier         TEXT NOT NULL,
  variant_key  TEXT NOT NULL,
  state        TEXT NOT NULL,
  segment_ids  TEXT NOT NULL,
  submitted_at INTEGER NOT NULL,
  settled_at   INTEGER,
  attempts     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tracks (
  source_id    TEXT NOT NULL,
  version      INTEGER NOT NULL,
  policy_id    TEXT NOT NULL,
  content_json TEXT NOT NULL,
  cost_usd     REAL NOT NULL,
  created_at   INTEGER NOT NULL,
  PRIMARY KEY (source_id, version)
);

CREATE TABLE IF NOT EXISTS references_ (
  segment_id  TEXT PRIMARY KEY,
  reference   TEXT NOT NULL,
  lang        TEXT NOT NULL,
  speaker_id  TEXT,
  district    TEXT,
  created_at  INTEGER NOT NULL
);
"""

JOB_STATES = ("pending", "running", "done", "failed")


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
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns the shipped schema gained after a DB was created.

        CREATE TABLE IF NOT EXISTS does nothing to an existing table, and
        dhvani.cli's --db defaults to a fixed shared path -- so a DB
        written before jobs.source_id existed is a real, reachable case,
        and every put_job() against it would fail with "table jobs has no
        column named source_id". Backfilled rows get '' , which matches no
        real source: such a job stops being visible to any source-filtered
        reconcile() rather than being settled by the wrong one, which is
        the safe direction for the bug this column exists to fix.
        """
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(jobs)")}
        if "source_id" not in columns:
            self.conn.execute(
                "ALTER TABLE jobs ADD COLUMN source_id TEXT NOT NULL DEFAULT ''"
            )

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

    def put_job(self, job_id, tier, variant_key, segment_ids, source_id) -> bool:
        """Register a submitted batch. Idempotent: re-registering is a no-op.

        source_id is required, not optional (FIX ROUND 3, C1): a job that
        does not know which source it belongs to cannot be filtered by
        one, and open_jobs() then hands every source's jobs to every
        reconcile() pass. A default here would let a caller silently
        reintroduce exactly that.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO jobs "
            "(job_id, source_id, tier, variant_key, state, segment_ids, "
            "submitted_at, attempts) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?, 0)",
            (job_id, source_id, tier, variant_key,
             json.dumps(list(segment_ids)), int(time.time())),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_job(self, job_id):
        row = self.conn.execute(
            "SELECT job_id, source_id, tier, variant_key, state, segment_ids, "
            "attempts FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"], "source_id": row["source_id"],
            "tier": row["tier"],
            "variant_key": row["variant_key"], "state": row["state"],
            "segment_ids": json.loads(row["segment_ids"]),
            "attempts": row["attempts"],
        }

    def set_job_state(self, job_id: str, state: str) -> None:
        if state not in JOB_STATES:
            raise ValueError(f"unknown state: {state!r}; expected one of {JOB_STATES}")
        settled = int(time.time()) if state in ("done", "failed") else None
        self.conn.execute(
            "UPDATE jobs SET state = ?, settled_at = ? WHERE job_id = ?",
            (state, settled, job_id),
        )
        self.conn.commit()

    def bump_job_attempts(self, job_id: str) -> int:
        self.conn.execute(
            "UPDATE jobs SET attempts = attempts + 1 WHERE job_id = ?", (job_id,)
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT attempts FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return int(row["attempts"]) if row else 0

    def open_jobs(self, source_id=None):
        """Jobs still awaiting results, ordered by job_id for determinism.

        FIX ROUND 3 (C1): pass source_id to see only that source's jobs.
        reconcile() polls and settles everything this returns, so an
        unfiltered listing let a reconcile() pass for one video mark
        another video's job done -- merge_entries() ignores the foreign
        segment_ids, so the wrong track was not corrupted, but the foreign
        job was settled, never retried, and its paid results were lost.
        Callers that genuinely want every source (diagnostics, tests) may
        still omit it.
        """
        if source_id is None:
            rows = self.conn.execute(
                "SELECT job_id FROM jobs WHERE state IN ('pending','running') "
                "ORDER BY job_id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT job_id FROM jobs WHERE state IN ('pending','running') "
                "AND source_id = ? ORDER BY job_id", (source_id,)
            ).fetchall()
        return [self.get_job(r["job_id"]) for r in rows]

    def put_track(self, source_id, version, policy_id, content_json, cost_usd) -> bool:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO tracks "
            "(source_id, version, policy_id, content_json, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, version, policy_id, content_json, cost_usd, int(time.time())),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_track(self, source_id, version):
        row = self.conn.execute(
            "SELECT content_json, policy_id, cost_usd FROM tracks "
            "WHERE source_id = ? AND version = ?", (source_id, version)
        ).fetchone()
        if row is None:
            return None
        return {"content_json": row["content_json"], "policy_id": row["policy_id"],
                "cost_usd": row["cost_usd"]}

    def latest_track_version(self, source_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM tracks WHERE source_id = ?",
            (source_id,)
        ).fetchone()
        return int(row["v"])

    def put_reference(self, segment_id, reference, lang,
                      speaker_id=None, district=None) -> bool:
        """Ground-truth transcript for a calibration segment.

        Empty in production; populated only by the calibration harness.
        Idempotent by primary key, matching every other write in this store.
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO references_ "
            "(segment_id, reference, lang, speaker_id, district, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (segment_id, reference, lang, speaker_id, district, int(time.time())),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_reference(self, segment_id):
        row = self.conn.execute(
            "SELECT reference, lang, speaker_id, district FROM references_ "
            "WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        if row is None:
            return None
        return {"reference": row["reference"], "lang": row["lang"],
                "speaker_id": row["speaker_id"], "district": row["district"]}
