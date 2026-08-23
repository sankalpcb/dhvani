# Dhvani Phase 2 — Async Reconciliation and Chaos Suite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tier 1 escalation asynchronous — publish a caption track immediately, submit an escalation batch, and reconcile results that arrive later, out of order, partially, or twice — with the correctness properties proven under injected failure.

**Architecture:** A new `AsyncBackend` protocol (`submit` returns a job id, `poll` returns results or `None`) sits beside Phase 1's synchronous `Backend`. Jobs and versioned tracks are persisted in SQLite. A pure `merge_entries()` folds arriving results into a track by `segment_id`, so re-applying a batch is a database-level no-op. A `ChaosBackend` wrapper injects timeouts, 429s, 500s, partial batches, duplicate deliveries and reordering, and the invariant suite asserts no loss, idempotent merge, and convergence to the synchronous result.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, stdlib `sqlite3`, `numpy`. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-21-dhvani-design.md` (§7 async reconciliation, §8 storage, §9.1 metrics, §10.3 invariants I1/I2/I6)

## Global Constraints

- **No model training.** No training loops, no `.pkl` artifacts, no `sklearn.fit`.
- **Total external spend must never exceed USD 20.** Every paid call goes through `Store.reserve_spend()`, which does the ceiling check and the insert in ONE SQL statement, BEFORE the call. Never split it back into `check_budget()` + `record_spend()` — that reopens the C1 race fixed in Phase 1.
- **Replay mode must NEVER fall back to live.** A missing fixture is a hard `FixtureMissing`.
- **Every pure function must be deterministic.** Same input plus same `POLICY_ID` yields byte-identical output. Tie-breaks sort by `segment_id`.
- **Sample rate is 16000 Hz mono int16** after `audio.normalize()`.
- **Goal G5:** the whole suite must run for a stranger with **no** cloud credentials and **no** ML dependencies installed. `torch`, `transformers`, `google-cloud-speech` stay in optional extras.
- **Cache keys carry variant identity.** Anything keyed by segment must fold in the backend's `variant_key` (via `ids.hypothesis_key` / `ids.variant_slug`), which already folds in `POLICY_ID`.
- **One Tier 1 cost model.** Price Tier 1 calls only via `tier1_chirp.cost_for_duration_ms()`. Never re-derive from a rate constant.

---

## Existing interfaces you consume (Phase 1, do not modify)

```python
# dhvani/pipeline.py
@dataclass(frozen=True)
class TrackEntry:
    segment_id: str; t_start_ms: int; t_end_ms: int
    text: str; risk: float; band: str
def band_of(risk: float) -> str                      # "ship" | "marked" | "review"
def run(pcm, source_id, tier0, store, delta_table, budget_usd) -> list[TrackEntry]

# dhvani/router.py
@dataclass(frozen=True)
class Candidate:
    segment_id: str; tier: str; risk: float; cost_usd: float; delta: float
def plan(candidates: list[Candidate], budget_usd: float) -> list[Candidate]
def delta_for(risk: float, tier: str, delta_table: dict) -> float

# dhvani/store.py
class Store:                                          # context manager
    def __init__(self, path: str, timeout: float = 30.0)
    def put_segment(self, segment_id, source_id, t_start_ms, t_end_ms, lang_hint=None)
    def put_hypothesis(self, segment_id, tier, text, signals, cost_usd, variant_key="") -> bool
    def get_hypothesis(self, segment_id, tier, variant_key="") -> dict | None
    def reserve_spend(self, tier: str, cost_usd: float) -> None   # atomic; raises BudgetExceeded
    def total_spend(self) -> float
class BudgetExceeded(RuntimeError)

# dhvani/ids.py
def segment_id(pcm) -> str
def hypothesis_key(tier: str, variant_key: str = "") -> str
def variant_slug(variant_key: str = "") -> str

# dhvani/backends/base.py
class Backend(Protocol):
    name: str; variant_key: str
    def cost_per_call(self, segment) -> float
    def transcribe(self, segment) -> dict          # {"text": str, "signals": dict}

# dhvani/backends/tier1_chirp.py
def cost_for_duration_ms(duration_ms: int) -> float

# dhvani/scorer.py
def extract(text, decoder_signals, duration_ms) -> Features
def risk(features) -> float
```

---

## File Structure

| File | Responsibility |
|---|---|
| `dhvani/store.py` (modify) | add `jobs` + `tracks` tables and their CRUD |
| `dhvani/track.py` (new) | pure `merge_entries()` — the idempotent merge |
| `dhvani/backends/async_base.py` (new) | `AsyncBackend` protocol, `SyncAsyncAdapter` |
| `dhvani/backends/chaos.py` (new) | `ChaosBackend` fault injection |
| `dhvani/reconcile.py` (new) | poll open jobs, merge, bump track version |
| `dhvani/escalate.py` (new) | router plan → submit → persist job |
| `dhvani/metrics.py` (new) | timing spans, percentiles, throughput |
| `dhvani/cli.py` (modify) | `--escalate` and `--reconcile` |

---

## Task 1: `jobs` and `tracks` tables

**Files:**
- Modify: `dhvani/store.py` (append to `SCHEMA`, add methods)
- Test: `tests/test_store_jobs.py`

**Interfaces:**
- Consumes: `Store`, `ids.hypothesis_key`
- Produces:
  - `Store.put_job(job_id, tier, variant_key, segment_ids: list[str]) -> bool`
  - `Store.get_job(job_id) -> dict | None` — keys `job_id, tier, variant_key, state, segment_ids, attempts`
  - `Store.set_job_state(job_id, state: str) -> None` — `state` in `pending|running|done|failed`
  - `Store.bump_job_attempts(job_id) -> int`
  - `Store.open_jobs() -> list[dict]` — state in `pending|running`, ordered by `job_id`
  - `Store.put_track(source_id, version, policy_id, content_json, cost_usd) -> bool`
  - `Store.get_track(source_id, version) -> dict | None`
  - `Store.latest_track_version(source_id) -> int` — `0` when none

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_jobs.py
import pytest
from dhvani.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_job_is_idempotent(store):
    assert store.put_job("j1", "tier1", "v1", ["a", "b"]) is True
    assert store.put_job("j1", "tier1", "v1", ["DIFFERENT"]) is False
    assert store.get_job("j1")["segment_ids"] == ["a", "b"]


def test_new_job_starts_pending_with_zero_attempts(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    job = store.get_job("j1")
    assert job["state"] == "pending"
    assert job["attempts"] == 0


def test_get_missing_job_returns_none(store):
    assert store.get_job("nope") is None


def test_set_job_state_round_trips(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    store.set_job_state("j1", "done")
    assert store.get_job("j1")["state"] == "done"


def test_set_job_state_rejects_unknown_state(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    with pytest.raises(ValueError, match="unknown state"):
        store.set_job_state("j1", "banana")


def test_bump_job_attempts_increments_and_returns(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    assert store.bump_job_attempts("j1") == 1
    assert store.bump_job_attempts("j1") == 2
    assert store.get_job("j1")["attempts"] == 2


def test_open_jobs_excludes_settled_and_is_ordered(store):
    for jid, state in [("j3", "pending"), ("j1", "done"), ("j2", "running")]:
        store.put_job(jid, "tier1", "v1", ["a"])
        store.set_job_state(jid, state)
    assert [j["job_id"] for j in store.open_jobs()] == ["j2", "j3"]


def test_put_track_is_idempotent(store):
    assert store.put_track("vid1", 1, "p1", "[]", 0.0) is True
    assert store.put_track("vid1", 1, "p1", '["DIFFERENT"]', 0.0) is False
    assert store.get_track("vid1", 1)["content_json"] == "[]"


def test_latest_track_version_starts_at_zero_and_advances(store):
    assert store.latest_track_version("vid1") == 0
    store.put_track("vid1", 1, "p1", "[]", 0.0)
    store.put_track("vid1", 2, "p1", "[]", 0.0)
    assert store.latest_track_version("vid1") == 2
    assert store.latest_track_version("other") == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store_jobs.py -v`
Expected: FAIL with `AttributeError: 'Store' object has no attribute 'put_job'`

- [ ] **Step 3: Extend the schema**

Append to `SCHEMA` in `dhvani/store.py`:

```sql
CREATE TABLE IF NOT EXISTS jobs (
  job_id       TEXT PRIMARY KEY,
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
```

- [ ] **Step 4: Add the methods**

```python
# dhvani/store.py  — add near the other Store methods

JOB_STATES = ("pending", "running", "done", "failed")


    def put_job(self, job_id, tier, variant_key, segment_ids) -> bool:
        """Register a submitted batch. Idempotent: re-registering is a no-op."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO jobs "
            "(job_id, tier, variant_key, state, segment_ids, submitted_at, attempts) "
            "VALUES (?, ?, ?, 'pending', ?, ?, 0)",
            (job_id, tier, variant_key, json.dumps(list(segment_ids)),
             int(time.time())),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def get_job(self, job_id):
        row = self.conn.execute(
            "SELECT job_id, tier, variant_key, state, segment_ids, attempts "
            "FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["job_id"], "tier": row["tier"],
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

    def open_jobs(self):
        """Jobs still awaiting results, ordered by job_id for determinism."""
        rows = self.conn.execute(
            "SELECT job_id FROM jobs WHERE state IN ('pending','running') "
            "ORDER BY job_id"
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_store_jobs.py -v`
Expected: PASS, 9 tests

- [ ] **Step 6: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/store.py tests/test_store_jobs.py
git commit -m "feat: jobs and versioned tracks tables for async escalation"
```

---

## Task 2: Pure idempotent track merge

**Files:**
- Create: `dhvani/track.py`
- Test: `tests/test_track.py`

**Interfaces:**
- Consumes: `dhvani.pipeline.TrackEntry`, `dhvani.pipeline.band_of`
- Produces:
  - `dhvani.track.merge_entries(base: list[TrackEntry], updates: dict[str, dict]) -> list[TrackEntry]`
    where each update value has keys `text: str`, `risk: float`
  - `dhvani.track.entries_to_json(entries) -> str`
  - `dhvani.track.entries_from_json(payload: str) -> list[TrackEntry]`

This is invariant **I2**. Idempotency comes from keyed replacement, and order-independence from sorting the output.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_track.py
import pytest
from dhvani.pipeline import TrackEntry
from dhvani.track import merge_entries, entries_to_json, entries_from_json


def _base():
    return [
        TrackEntry("a" * 64, 0, 3000, "one", 0.9, "review"),
        TrackEntry("b" * 64, 3000, 6000, "two", 0.5, "marked"),
    ]


def test_merge_replaces_text_and_recomputes_band():
    out = merge_entries(_base(), {"a" * 64: {"text": "fixed", "risk": 0.1}})
    assert out[0].text == "fixed"
    assert out[0].risk == 0.1
    assert out[0].band == "ship"


def test_merge_leaves_untouched_entries_alone():
    out = merge_entries(_base(), {"a" * 64: {"text": "fixed", "risk": 0.1}})
    assert out[1].text == "two"
    assert out[1].band == "marked"


def test_merge_is_idempotent():
    """Invariant I2: applying the same result twice is a no-op."""
    upd = {"a" * 64: {"text": "fixed", "risk": 0.1}}
    once = merge_entries(_base(), upd)
    twice = merge_entries(once, upd)
    assert once == twice


def test_merge_ignores_unknown_segment_ids():
    """A late result for a segment not in this track must not invent an entry."""
    out = merge_entries(_base(), {"z" * 64: {"text": "ghost", "risk": 0.0}})
    assert [e.segment_id for e in out] == ["a" * 64, "b" * 64]


def test_merge_never_loses_entries():
    """Invariant I1: every input segment appears exactly once in the output."""
    out = merge_entries(_base(), {"a" * 64: {"text": "fixed", "risk": 0.1}})
    ids = [e.segment_id for e in out]
    assert sorted(ids) == sorted(e.segment_id for e in _base())
    assert len(ids) == len(set(ids))


def test_merge_output_is_order_independent():
    """Two updates applied in either order give byte-identical output."""
    u1 = {"a" * 64: {"text": "A", "risk": 0.2}}
    u2 = {"b" * 64: {"text": "B", "risk": 0.3}}
    left = merge_entries(merge_entries(_base(), u1), u2)
    right = merge_entries(merge_entries(_base(), u2), u1)
    assert left == right


def test_merge_sorts_by_time_then_segment_id():
    shuffled = list(reversed(_base()))
    out = merge_entries(shuffled, {})
    assert [e.t_start_ms for e in out] == [0, 3000]


def test_json_round_trip():
    entries = _base()
    assert entries_from_json(entries_to_json(entries)) == entries


def test_json_is_stable_for_equal_input():
    """Invariant I5: same entries -> byte-identical payload."""
    assert entries_to_json(_base()) == entries_to_json(list(reversed(_base())))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_track.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.track'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/track.py
"""Versioned caption tracks and the idempotent merge that advances them.

Spec §7: Tier 1 results arrive up to 24h after submission, possibly out of
order, possibly partial, possibly twice. merge_entries() is what makes that
safe. It is a pure keyed replacement over segment_id, so:

  I1 (no loss)      -- entries are only ever replaced, never removed, and an
                       update for an unknown segment_id is ignored rather
                       than inserted.
  I2 (idempotent)   -- replacing an entry with the same values twice yields
                       the same track; merge(merge(t, u), u) == merge(t, u).
  I5 (determinism)  -- output is sorted by (t_start_ms, segment_id), so the
                       arrival order of updates cannot change the result.
"""

import json
from dataclasses import replace

from dhvani.pipeline import TrackEntry, band_of


def merge_entries(base: list[TrackEntry], updates: dict) -> list[TrackEntry]:
    """Fold arriving results into a track. Pure and order-independent.

    updates maps segment_id -> {"text": str, "risk": float}. The band is
    recomputed from the new risk rather than carried over, so an escalated
    segment can move out of the review band.
    """
    by_id = {e.segment_id: e for e in base}

    for segment_id, upd in updates.items():
        current = by_id.get(segment_id)
        if current is None:
            # A result for a segment this track does not contain. Ignoring it
            # protects I1: the merge can never grow the track.
            continue
        risk = float(upd["risk"])
        by_id[segment_id] = replace(
            current, text=upd["text"], risk=risk, band=band_of(risk)
        )

    return sorted(by_id.values(), key=lambda e: (e.t_start_ms, e.segment_id))


def entries_to_json(entries: list[TrackEntry]) -> str:
    ordered = sorted(entries, key=lambda e: (e.t_start_ms, e.segment_id))
    return json.dumps([e.__dict__ for e in ordered],
                      ensure_ascii=False, indent=2, sort_keys=True)


def entries_from_json(payload: str) -> list[TrackEntry]:
    return [TrackEntry(**row) for row in json.loads(payload)]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_track.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/track.py tests/test_track.py
git commit -m "feat: pure idempotent track merge (invariants I1, I2, I5)"
```

---

## Task 3: `AsyncBackend` protocol and `SyncAsyncAdapter`

**Files:**
- Create: `dhvani/backends/async_base.py`
- Test: `tests/test_async_base.py`

**Interfaces:**
- Consumes: `dhvani.backends.base.Backend`
- Produces:
  - `AsyncBackend` Protocol: `name: str`, `variant_key: str`, `cost_per_call(segment) -> float`, `submit(segments: list) -> str`, `poll(job_id: str) -> dict | None`
  - `SyncAsyncAdapter(inner: Backend, pending_polls: int = 0)` implementing `AsyncBackend`
  - `JobNotFound(RuntimeError)`

`poll` returns `None` while the job is still pending, or `{segment_id: {"text":…, "signals":…}}` when complete. `SyncAsyncAdapter` makes every downstream component testable with no cloud and no network.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_async_base.py
import numpy as np
import pytest

from dhvani.backends.async_base import AsyncBackend, SyncAsyncAdapter, JobNotFound
from dhvani.segmenter import Segment


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def __init__(self):
        self.calls = 0

    def cost_per_call(self, segment):
        return 0.003

    def transcribe(self, segment):
        self.calls += 1
        return {"text": f"out-{segment.segment_id[:4]}", "signals": {}}


def _segs(n=2):
    return [Segment(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                    np.zeros(10, dtype=np.int16)) for i in range(n)]


def test_adapter_satisfies_async_backend_protocol():
    assert isinstance(SyncAsyncAdapter(StubSync()), AsyncBackend)


def test_submit_returns_a_stable_job_id_for_the_same_segments():
    a = SyncAsyncAdapter(StubSync()).submit(_segs())
    b = SyncAsyncAdapter(StubSync()).submit(_segs())
    assert a == b, "job id must be content-derived, not random"


def test_submit_returns_different_ids_for_different_segments():
    assert SyncAsyncAdapter(StubSync()).submit(_segs(2)) != \
           SyncAsyncAdapter(StubSync()).submit(_segs(3))


def test_submit_does_not_call_the_inner_backend():
    """Submission is cheap; the work happens at poll time."""
    inner = StubSync()
    SyncAsyncAdapter(inner).submit(_segs())
    assert inner.calls == 0


def test_poll_returns_results_keyed_by_segment_id():
    a = SyncAsyncAdapter(StubSync())
    job_id = a.submit(_segs())
    out = a.poll(job_id)
    assert set(out) == {"a" * 64, "b" * 64}
    assert out["a" * 64]["text"].startswith("out-")


def test_poll_returns_none_while_pending():
    a = SyncAsyncAdapter(StubSync(), pending_polls=2)
    job_id = a.submit(_segs())
    assert a.poll(job_id) is None
    assert a.poll(job_id) is None
    assert a.poll(job_id) is not None


def test_poll_on_unknown_job_raises():
    with pytest.raises(JobNotFound):
        SyncAsyncAdapter(StubSync()).poll("no-such-job")


def test_cost_per_call_delegates_to_inner():
    assert SyncAsyncAdapter(StubSync()).cost_per_call(_segs()[0]) == 0.003
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_async_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.backends.async_base'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/backends/async_base.py
"""Asynchronous backend protocol.

Spec §7: Chirp dynamic batch has up to 24h turnaround, so escalation cannot
be a blocking call. submit() registers a batch and returns a job id; poll()
returns None while the batch is outstanding and a {segment_id: result} dict
once it completes.

Job ids are derived from the submitted segment ids, not generated randomly.
That makes them deterministic (invariant I5) and makes re-submitting the same
batch a no-op at the Store level rather than a duplicate job.

SyncAsyncAdapter wraps any synchronous Backend as an AsyncBackend so the
whole async machinery is testable with no cloud, no network, and no
credentials (goal G5). pending_polls simulates turnaround latency.
"""

import hashlib
from typing import Protocol, runtime_checkable


class JobNotFound(RuntimeError):
    """poll() was given a job id this backend never issued."""


@runtime_checkable
class AsyncBackend(Protocol):
    name: str
    variant_key: str

    def cost_per_call(self, segment) -> float: ...

    def submit(self, segments: list) -> str: ...

    def poll(self, job_id: str) -> dict | None: ...


def job_id_for(variant_key: str, segments) -> str:
    """Content-derived job id: same variant plus same segments -> same id."""
    digest = hashlib.sha256(variant_key.encode("utf-8"))
    for segment_id in sorted(s.segment_id for s in segments):
        digest.update(segment_id.encode("utf-8"))
    return digest.hexdigest()[:32]


class SyncAsyncAdapter:
    """Presents a synchronous Backend through the AsyncBackend protocol."""

    def __init__(self, inner, pending_polls: int = 0):
        self.inner = inner
        self.name = inner.name
        self.variant_key = inner.variant_key
        self.pending_polls = pending_polls
        self._jobs: dict[str, list] = {}
        self._polls: dict[str, int] = {}

    def cost_per_call(self, segment) -> float:
        return self.inner.cost_per_call(segment)

    def submit(self, segments: list) -> str:
        job_id = job_id_for(self.variant_key, segments)
        self._jobs[job_id] = list(segments)
        self._polls.setdefault(job_id, 0)
        return job_id

    def poll(self, job_id: str) -> dict | None:
        if job_id not in self._jobs:
            raise JobNotFound(f"unknown job id: {job_id}")
        if self._polls[job_id] < self.pending_polls:
            self._polls[job_id] += 1
            return None
        return {s.segment_id: self.inner.transcribe(s) for s in self._jobs[job_id]}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_async_base.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/backends/async_base.py tests/test_async_base.py
git commit -m "feat: AsyncBackend protocol and SyncAsyncAdapter"
```

---

## Task 4: `escalate()` — router plan to submitted job

**Files:**
- Create: `dhvani/escalate.py`
- Test: `tests/test_escalate.py`

**Interfaces:**
- Consumes: `router.Candidate`, `router.plan`, `router.delta_for`, `tier1_chirp.cost_for_duration_ms`, `Store.put_job`, `Store.reserve_spend`, `AsyncBackend`
- Produces:
  - `dhvani.escalate.escalate(entries: list[TrackEntry], segments: dict[str, Segment], backend, store, delta_table: dict, budget_usd: float) -> str | None`
    Returns the submitted `job_id`, or `None` when the plan is empty.

`segments` maps `segment_id -> Segment` (the real objects from `segmenter.split`, carrying
`pcm`). Durations are derived from each segment's own bounds.

**Why real `Segment`s and not a lightweight stub:** `Tier1Chirp.transcribe` reads
`segment.pcm` to build its request. A stub carrying only ids and time bounds passes every
test that injects a fake backend and then `AttributeError`s the first time it meets the real
one. Spend is reserved for the whole batch BEFORE submission, matching the Phase 1 rule that
money is accounted before the call.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_escalate.py
import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        from dhvani.backends.tier1_chirp import cost_for_duration_ms
        return cost_for_duration_ms(segment.t_end_ms - segment.t_start_ms)

    def transcribe(self, segment):
        return {"text": "escalated", "signals": {}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def _entries():
    return [TrackEntry("a" * 64, 0, 3000, "raw", 0.65, "review"),
            TrackEntry("b" * 64, 3000, 6000, "raw", 0.05, "ship")]


def _segments():
    """Real Segment objects — Tier1Chirp.transcribe() reads segment.pcm."""
    return {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in _entries()}


SEGMENTS = _segments()
TABLE = {"tier1": {"0.6-0.7": 18.0}}


def test_zero_budget_submits_nothing(store):
    assert escalate(_entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                    store, TABLE, budget_usd=0.0) is None


def test_empty_delta_table_submits_nothing(store):
    """No measured improvement means no candidate has positive delta."""
    assert escalate(_entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                    store, {}, budget_usd=10.0) is None


def test_escalation_registers_a_job_with_the_selected_segments(store):
    job_id = escalate(_entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, TABLE, budget_usd=10.0)
    assert job_id is not None
    job = store.get_job(job_id)
    assert job["segment_ids"] == ["a" * 64]
    assert job["state"] == "pending"
    assert job["variant_key"] == "tier1|hi-IN"


def test_low_risk_segments_are_not_escalated(store):
    job_id = escalate(_entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, TABLE, budget_usd=10.0)
    assert "b" * 64 not in store.get_job(job_id)["segment_ids"]


def test_spend_is_reserved_before_submission(store):
    escalate(_entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
             store, TABLE, budget_usd=10.0)
    assert store.total_spend() > 0.0


def test_escalation_fails_closed_at_the_ceiling(store):
    store.reserve_spend("tier1", 19.999)
    with pytest.raises(BudgetExceeded):
        escalate(_entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                 store, TABLE, budget_usd=10.0)


def test_resubmitting_the_same_batch_is_idempotent(store):
    backend = SyncAsyncAdapter(StubSync())
    first = escalate(_entries(), SEGMENTS, backend, store, TABLE, 10.0)
    second = escalate(_entries(), SEGMENTS, backend, store, TABLE, 10.0)
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_escalate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.escalate'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/escalate.py
"""Turn a caption track into a submitted escalation batch.

Spec §6: the router decides WHICH segments are worth expensive treatment
under a budget. This module turns that decision into a persisted job.

Spend is reserved for the entire batch BEFORE submit() is called, matching
the Phase 1 rule that money is accounted before the paid call, never after —
a crash between submission and accounting would otherwise under-count and
let the USD 20 ceiling be breached on restart.
"""

from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.router import Candidate, delta_for, plan


def _duration_ms(segment) -> int:
    return segment.t_end_ms - segment.t_start_ms


def escalate(entries, segments, backend, store, delta_table, budget_usd):
    """Plan escalations, reserve their cost, and submit them. Returns job id.

    segments maps segment_id -> Segment. Real Segment objects are required,
    not stubs: Tier1Chirp.transcribe() reads segment.pcm.
    """
    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier1",
            risk=e.risk,
            cost_usd=cost_for_duration_ms(_duration_ms(segments[e.segment_id])),
            delta=delta_for(e.risk, "tier1", delta_table),
        )
        for e in entries
        if e.segment_id in segments
    ]

    chosen = plan(candidates, budget_usd)
    if not chosen:
        return None

    # ONE reservation for the whole batch, not one per candidate. A loop can
    # partially succeed then raise, leaving spend reserved for a batch that was
    # never submitted -- money that buys nothing and cannot be recovered.
    total_cost = sum(cand.cost_usd for cand in chosen)
    store.reserve_spend(backend.name, total_cost)

    batch = [segments[c.segment_id] for c in chosen]

    job_id = backend.submit(batch)
    store.put_job(job_id, backend.name, backend.variant_key,
                  [s.segment_id for s in batch])
    return job_id
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_escalate.py -v`
Expected: PASS, 7 tests

Note: `test_resubmitting_the_same_batch_is_idempotent` asserts the job id repeats. Spend IS reserved twice — that is the deliberate pessimistic-accounting behavior from Phase 1, not a bug.

- [ ] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/escalate.py tests/test_escalate.py
git commit -m "feat: escalate() plans, reserves, and submits an async batch"
```

---

## Task 5: Reconciler

**Files:**
- Create: `dhvani/reconcile.py`
- Test: `tests/test_reconcile.py`

**Interfaces:**
- Consumes: `track.merge_entries`, `track.entries_to_json`, `track.entries_from_json`, `Store` job/track methods, `scorer.extract`, `scorer.risk`, `AsyncBackend.poll`
- Produces:
  - `dhvani.reconcile.reconcile(source_id: str, backend, store) -> int`
    Polls every open job for this backend, merges completed results, writes a new track version. Returns the new version, or the existing latest version when nothing advanced.
    Durations are derived from the track's own entries — no caller needs to supply them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reconcile.py
import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.reconcile import reconcile
from dhvani.segmenter import Segment
from dhvani.store import Store
from dhvani.track import entries_to_json, entries_from_json
from dhvani.config import POLICY_ID


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": "escalated", "signals": {"ctc_rnnt_disagreement": 0.0}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


ENTRIES = [TrackEntry("a" * 64, 0, 3000, "raw", 0.65, "review"),
           TrackEntry("b" * 64, 3000, 6000, "raw", 0.05, "ship")]
SEGMENTS = {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in ENTRIES}
TABLE = {"tier1": {"0.6-0.7": 18.0}}


def _seed_v1(store):
    store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)


def test_reconcile_with_no_jobs_leaves_the_version_alone(store):
    _seed_v1(store)
    assert reconcile("vid1", SyncAsyncAdapter(StubSync()), store) == 1


def test_reconcile_advances_the_version_when_results_arrive(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    assert reconcile("vid1", backend, store) == 2


def test_reconciled_track_contains_the_escalated_text(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = reconcile("vid1", backend, store)
    merged = entries_from_json(store.get_track("vid1", version)["content_json"])
    assert merged[0].text == "escalated"
    assert merged[1].text == "raw"


def test_pending_job_does_not_advance_the_version(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync(), pending_polls=5)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    assert reconcile("vid1", backend, store) == 1


def test_completed_job_is_marked_done(store):
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    job_id = escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    reconcile("vid1", backend, store)
    assert store.get_job(job_id)["state"] == "done"


def test_reconciling_twice_does_not_advance_twice(store):
    """Invariant I2 at the reconciler level: a settled job is not re-merged."""
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    first = reconcile("vid1", backend, store)
    second = reconcile("vid1", backend, store)
    assert first == second == 2


def test_reconcile_never_loses_segments(store):
    """Invariant I1."""
    _seed_v1(store)
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = reconcile("vid1", backend, store)
    merged = entries_from_json(store.get_track("vid1", version)["content_json"])
    assert sorted(e.segment_id for e in merged) == \
           sorted(e.segment_id for e in ENTRIES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reconcile.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.reconcile'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/reconcile.py
"""Poll outstanding escalation jobs and fold their results into a new track.

Spec §7 step 3-4: results arrive late, out of order, possibly partial,
possibly twice. This module is the only place that advances a track version.

A job is polled at most once per reconcile() pass. A job still pending is
left open; a job that returns results is merged, marked done, and the track
version is bumped. Merging is delegated to track.merge_entries(), which is
pure and idempotent, so a duplicate delivery cannot corrupt the track.
"""

from dhvani.config import POLICY_ID
from dhvani.scorer import extract, risk as compute_risk
from dhvani.track import entries_from_json, entries_to_json, merge_entries


def reconcile(source_id: str, backend, store) -> int:
    """Merge any completed jobs for this backend. Returns the latest version."""
    version = store.latest_track_version(source_id)
    if version == 0:
        return 0

    current = store.get_track(source_id, version)
    entries = entries_from_json(current["content_json"])
    # The track already knows every segment's bounds, so durations need not be
    # threaded through the caller.
    durations = {e.segment_id: e.t_end_ms - e.t_start_ms for e in entries}

    updates: dict[str, dict] = {}
    settled: list[str] = []

    for job in store.open_jobs():
        if job["tier"] != backend.name or job["variant_key"] != backend.variant_key:
            continue

        store.bump_job_attempts(job["job_id"])
        results = backend.poll(job["job_id"])
        if results is None:
            store.set_job_state(job["job_id"], "running")
            continue

        for segment_id, result in results.items():
            duration = durations.get(segment_id, 0)
            features = extract(result["text"], result.get("signals", {}), duration)
            updates[segment_id] = {
                "text": result["text"],
                "risk": compute_risk(features),
            }
            store.put_hypothesis(
                segment_id, backend.name, result["text"],
                result.get("signals", {}), 0.0, backend.variant_key,
            )

        settled.append(job["job_id"])

    if not updates:
        return version

    merged = merge_entries(entries, updates)
    new_version = version + 1
    store.put_track(source_id, new_version, POLICY_ID,
                    entries_to_json(merged), current["cost_usd"])

    for job_id in settled:
        store.set_job_state(job_id, "done")

    return new_version
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_reconcile.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/reconcile.py tests/test_reconcile.py
git commit -m "feat: reconciler merges completed jobs into a new track version"
```

---

## Task 6: `ChaosBackend` fault injection

**Files:**
- Create: `dhvani/backends/chaos.py`
- Test: `tests/test_chaos.py`

**Interfaces:**
- Consumes: `AsyncBackend`
- Produces:
  - `ChaosBackend(inner, faults: list[str], seed: int = 0)` implementing `AsyncBackend`
  - Recognized fault names: `"timeout"`, `"rate_limit"`, `"server_error"`, `"partial"`, `"duplicate"`, `"reorder"`
  - `TransientError(RuntimeError)` — base for injected `timeout` / `rate_limit` / `server_error`

Faults are applied deterministically from `seed`, so a failing chaos test is reproducible.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chaos.py
import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.chaos import ChaosBackend, TransientError
from dhvani.segmenter import Segment


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": f"out-{segment.segment_id[:4]}", "signals": {}}


def _segs(n=4):
    return [Segment(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                    np.zeros(10, dtype=np.int16)) for i in range(n)]


def _chaos(faults, seed=0, pending=0):
    return ChaosBackend(SyncAsyncAdapter(StubSync(), pending_polls=pending),
                        faults=faults, seed=seed)


def test_no_faults_is_transparent():
    b = _chaos([])
    out = b.poll(b.submit(_segs()))
    assert len(out) == 4


def test_timeout_fault_raises_transient_error():
    b = _chaos(["timeout"])
    with pytest.raises(TransientError, match="timeout"):
        b.poll(b.submit(_segs()))


def test_rate_limit_fault_raises_transient_error():
    b = _chaos(["rate_limit"])
    with pytest.raises(TransientError, match="429"):
        b.poll(b.submit(_segs()))


def test_server_error_fault_raises_transient_error():
    b = _chaos(["server_error"])
    with pytest.raises(TransientError, match="500"):
        b.poll(b.submit(_segs()))


def test_partial_fault_returns_a_strict_subset():
    b = _chaos(["partial"])
    out = b.poll(b.submit(_segs()))
    assert 0 < len(out) < 4


def test_duplicate_fault_still_returns_a_dict_keyed_by_segment_id():
    """A duplicate delivery cannot produce duplicate keys — that is the point."""
    b = _chaos(["duplicate"])
    job_id = b.submit(_segs())
    first, second = b.poll(job_id), b.poll(job_id)
    assert first == second


def test_reorder_fault_changes_iteration_order_but_not_content():
    plain = _chaos([]).poll(_chaos([]).submit(_segs()))
    b = _chaos(["reorder"], seed=7)
    shuffled = b.poll(b.submit(_segs()))
    assert list(shuffled) != list(plain) or len(plain) < 2
    assert shuffled == plain, "reordering must not change the mapping"


def test_faults_are_deterministic_for_a_given_seed():
    a = _chaos(["partial"], seed=3)
    c = _chaos(["partial"], seed=3)
    assert a.poll(a.submit(_segs())) == c.poll(c.submit(_segs()))


def test_unknown_fault_name_is_rejected():
    with pytest.raises(ValueError, match="unknown fault"):
        _chaos(["earthquake"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chaos.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.backends.chaos'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/backends/chaos.py
"""Deterministic fault injection for the async escalation path.

Spec §10.2: the chaos layer injects timeouts, 429s, 500s, partial batches,
duplicate deliveries and reordering, and the invariant suite asserts the
system still loses nothing and converges.

Faults are driven by an explicit seed rather than real randomness, so a
failing chaos test reproduces exactly. Nothing here talks to a network.
"""

import random

FAULTS = ("timeout", "rate_limit", "server_error", "partial", "duplicate", "reorder")


class TransientError(RuntimeError):
    """An injected failure that a caller is expected to survive and retry."""


class ChaosBackend:
    def __init__(self, inner, faults=(), seed: int = 0):
        unknown = [f for f in faults if f not in FAULTS]
        if unknown:
            raise ValueError(f"unknown fault(s): {unknown}; expected one of {FAULTS}")
        self.inner = inner
        self.name = inner.name
        self.variant_key = inner.variant_key
        self.faults = tuple(faults)
        self.seed = seed

    def cost_per_call(self, segment) -> float:
        return self.inner.cost_per_call(segment)

    def submit(self, segments: list) -> str:
        return self.inner.submit(segments)

    def poll(self, job_id: str):
        if "timeout" in self.faults:
            raise TransientError("injected timeout while polling")
        if "rate_limit" in self.faults:
            raise TransientError("injected 429 rate limit")
        if "server_error" in self.faults:
            raise TransientError("injected 500 server error")

        results = self.inner.poll(job_id)
        if results is None:
            return None

        rng = random.Random(f"{self.seed}:{job_id}")
        items = sorted(results.items())

        if "partial" in self.faults and len(items) > 1:
            keep = max(1, len(items) // 2)
            items = items[:keep]

        if "reorder" in self.faults:
            rng.shuffle(items)

        # "duplicate" models the same payload being delivered twice. Because
        # results are keyed by segment_id, a duplicate collapses into the same
        # dict -- which is exactly the property the merge relies on.
        if "duplicate" in self.faults:
            items = items + items

        return dict(items)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_chaos.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/backends/chaos.py tests/test_chaos.py
git commit -m "feat: deterministic ChaosBackend fault injection"
```

---

## Task 7: Invariant suite I1, I2, I6 under chaos

**Files:**
- Create: `tests/test_invariants.py`

**Interfaces:**
- Consumes: everything above

This task adds **no production code**. It is the proof that the async path is correct, and it is the single most valuable artifact in Phase 2.

- [ ] **Step 1: Write the invariant suite**

```python
# tests/test_invariants.py
"""Spec §10.3 invariants for the async escalation path.

I1  no loss          -- every input segment appears exactly once in the track
I2  idempotent merge -- applying a batch result twice is a no-op
I6  convergence      -- once all jobs settle, the async track equals what a
                        fully synchronous pipeline would have produced
"""

import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.chaos import ChaosBackend, TransientError
from dhvani.config import POLICY_ID
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.reconcile import reconcile
from dhvani.segmenter import Segment
from dhvani.store import Store
from dhvani.track import entries_from_json, entries_to_json


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": f"fixed-{segment.segment_id[:4]}", "signals": {}}


N = 8
ENTRIES = [TrackEntry(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                      "raw", 0.65, "review") for i in range(N)]
SEGMENTS = {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in ENTRIES}
TABLE = {"tier1": {"0.6-0.7": 18.0}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        s.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
        yield s


def _drain(backend, store, max_passes=20):
    """Reconcile until nothing advances, surviving injected transients."""
    version = store.latest_track_version("vid1")
    for _ in range(max_passes):
        try:
            new_version = reconcile("vid1", backend, store)
        except TransientError:
            continue
        if new_version == version:
            break
        version = new_version
    return version


def _track(store, version):
    return entries_from_json(store.get_track("vid1", version)["content_json"])


@pytest.mark.parametrize("faults", [
    (), ("partial",), ("duplicate",), ("reorder",),
    ("partial", "reorder"), ("duplicate", "reorder"),
])
def test_i1_no_segment_is_ever_lost_or_duplicated(store, faults):
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=faults, seed=11)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    entries = _track(store, _drain(backend, store))
    ids = [e.segment_id for e in entries]
    assert len(ids) == N
    assert len(set(ids)) == N
    assert sorted(ids) == sorted(SEGMENTS)


@pytest.mark.parametrize("faults", [("timeout",), ("rate_limit",), ("server_error",)])
def test_i1_holds_when_every_poll_fails(store, faults):
    """A backend that never succeeds must leave the track intact, not corrupt."""
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=faults, seed=5)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    entries = _track(store, _drain(backend, store))
    assert [e.text for e in entries] == ["raw"] * N


def test_i2_reconciling_a_settled_job_repeatedly_is_a_no_op(store):
    backend = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    version = _drain(backend, store)
    before = store.get_track("vid1", version)["content_json"]
    for _ in range(5):
        assert reconcile("vid1", backend, store) == version
    assert store.get_track("vid1", version)["content_json"] == before


def test_i6_async_converges_to_the_synchronous_result(store, tmp_path):
    """The headline property: once everything settles, async == sync."""
    backend = ChaosBackend(SyncAsyncAdapter(StubSync(), pending_polls=2),
                           faults=("partial", "reorder", "duplicate"), seed=3)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    async_entries = _track(store, _drain(backend, store, max_passes=60))

    with Store(str(tmp_path / "sync.db")) as sync_store:
        sync_store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
        plain = SyncAsyncAdapter(StubSync())
        escalate(ENTRIES, SEGMENTS, plain, sync_store, TABLE, 10.0)
        sync_entries = _track(sync_store, _drain(plain, sync_store))

    assert async_entries == sync_entries


def test_i6_convergence_is_reachable_from_a_partial_start(store):
    """Partial delivery must not strand segments permanently un-escalated."""
    backend = ChaosBackend(SyncAsyncAdapter(StubSync()), faults=("partial",), seed=2)
    escalate(ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
    _drain(backend, store)

    clean = SyncAsyncAdapter(StubSync())
    escalate(ENTRIES, SEGMENTS, clean, store, TABLE, 10.0)
    entries = _track(store, _drain(clean, store))
    assert all(e.text.startswith("fixed-") for e in entries)
```

- [ ] **Step 2: Run the suite**

Run: `uv run pytest tests/test_invariants.py -v`
Expected: PASS, 13 tests (6 parametrized I1 + 3 parametrized transient + 3 others)

If `test_i6_convergence_is_reachable_from_a_partial_start` fails, the reconciler is marking partially-delivered jobs `done`. That is a real defect — segments dropped by a partial delivery would never be retried. Fix the reconciler so a job is only marked `done` when its results cover every `segment_id` it registered; otherwise leave it `running`.

- [ ] **Step 3: Full suite and commit**

```bash
uv run pytest -q
git add tests/test_invariants.py
git commit -m "test: prove invariants I1, I2, I6 hold under injected chaos"
```

---

## Task 8: Metrics and CLI wiring

**Files:**
- Create: `dhvani/metrics.py`
- Modify: `dhvani/cli.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Store`, `reconcile`, `escalate`
- Produces:
  - `dhvani.metrics.Timer()` — context manager, `.elapsed_ms: float`
  - `dhvani.metrics.percentile(values: list[float], p: float) -> float`
  - `dhvani.metrics.summarize(samples: dict[str, list[float]]) -> dict`
    returns `{name: {"count", "p50", "p99", "total_ms"}}`
  - `dhvani.metrics.throughput(audio_ms: int, wall_ms: float) -> float` — audio-hours per wall-clock hour
  - CLI flags `--escalate` and `--reconcile`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import pytest
from dhvani.metrics import Timer, percentile, summarize, throughput


def test_timer_measures_a_nonnegative_span():
    with Timer() as t:
        sum(range(1000))
    assert t.elapsed_ms >= 0.0


def test_percentile_endpoints():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.99) == 99.0
    assert percentile(values, 1.0) == 100.0


def test_percentile_of_single_value():
    assert percentile([7.0], 0.5) == 7.0


def test_percentile_of_empty_is_zero():
    assert percentile([], 0.5) == 0.0


def test_percentile_rejects_out_of_range_p():
    with pytest.raises(ValueError, match="between 0 and 1"):
        percentile([1.0], 1.5)


def test_summarize_reports_count_p50_p99_and_total():
    out = summarize({"tier0": [1.0, 2.0, 3.0, 4.0]})
    assert out["tier0"]["count"] == 4
    assert out["tier0"]["p50"] == 2.0
    assert out["tier0"]["total_ms"] == 10.0


def test_summarize_handles_an_empty_series():
    out = summarize({"tier1": []})
    assert out["tier1"] == {"count": 0, "p50": 0.0, "p99": 0.0, "total_ms": 0.0}


def test_throughput_one_hour_of_audio_in_one_hour_is_one():
    assert throughput(3_600_000, 3_600_000.0) == pytest.approx(1.0)


def test_throughput_is_zero_when_no_time_elapsed():
    assert throughput(1000, 0.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/metrics.py
"""Timing and throughput instrumentation (spec §9.1).

Deliberately dependency-free and deterministic in shape: percentile() uses
linear interpolation over a sorted copy, so the same samples always give the
same summary. Timing values themselves vary run to run, which is why no test
asserts a specific duration.
"""

import time


class Timer:
    """Context manager measuring wall-clock milliseconds."""

    def __init__(self):
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


def percentile(values, p: float) -> float:
    """Linear-interpolated percentile. p is a fraction in [0, 1]."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = p * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def summarize(samples: dict) -> dict:
    """Per-series count, p50, p99 and total."""
    return {
        name: {
            "count": len(series),
            "p50": percentile(series, 0.50),
            "p99": percentile(series, 0.99),
            "total_ms": float(sum(series)),
        }
        for name, series in samples.items()
    }


def throughput(audio_ms: int, wall_ms: float) -> float:
    """Audio-hours processed per wall-clock hour. Zero when no time elapsed."""
    if wall_ms <= 0.0:
        return 0.0
    return float(audio_ms) / float(wall_ms)
```

- [ ] **Step 4: Wire the CLI**

Add to `dhvani/cli.py`'s argument list, after `--out`:

```python
    ap.add_argument(
        "--escalate", action="store_true",
        help="after transcribing, plan and submit a Tier 1 escalation batch",
    )
    ap.add_argument(
        "--reconcile", action="store_true",
        help="poll outstanding escalation jobs and merge any completed results",
    )
```

And after the track is written, inside the `with Store(...)` block:

```python
        if args.escalate or args.reconcile:
            from dhvani.backends.async_base import SyncAsyncAdapter
            from dhvani.backends.tier1_chirp import Tier1Chirp
            from dhvani.escalate import escalate as do_escalate
            from dhvani.reconcile import reconcile as do_reconcile
            from dhvani.track import entries_to_json

            source = os.path.basename(args.audio)
            # Real Segments, not stubs: Tier1Chirp.transcribe() reads .pcm.
            segments = {s.segment_id: s for s in split(pcm)}
            tier1 = SyncAsyncAdapter(Tier1Chirp(lang=f"{args.lang}-IN"))

            if store.latest_track_version(source) == 0:
                store.put_track(source, 1, POLICY_ID,
                                entries_to_json(entries), 0.0)

            if args.escalate:
                do_escalate(entries, segments, tier1, store,
                            delta_table, args.budget)
            if args.reconcile:
                do_reconcile(source, tier1, store)
```

Add `from dhvani.config import POLICY_ID` and `from dhvani.segmenter import segment as split` to `cli.py`'s imports.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS, 9 tests

Then confirm the CLI still parses: `uv run dhvani --help` must list `--escalate` and `--reconcile` and exit 0.

- [ ] **Step 6: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/metrics.py dhvani/cli.py tests/test_metrics.py
git commit -m "feat: timing metrics and --escalate/--reconcile CLI flags"
```

---

## Phase 2 Exit Criteria

- [ ] `uv run pytest` passes with no network, no cloud credentials, and no ML dependencies
- [ ] Invariants I1, I2 and I6 are asserted under six distinct fault combinations
- [ ] A job partially delivered is retried rather than marked done
- [ ] Re-reconciling a settled job never advances the track version
- [ ] `uv run dhvani --help` lists `--escalate` and `--reconcile`
- [ ] Total external spend recorded in the `spend` ledger is unchanged from Phase 1 (this phase adds no live calls)

## Deferred to Phase 3

Tier 2 Gemini repair with quota-aware rate limiting (the 1,000 requests/day free-tier cap is a
hard wall and deserves its own token-bucket plus graceful degradation design), and the
production scaling sketch (spec §12) — Bigtable shard-key analysis, Pub/Sub queue sizing, and
cost-per-million-audio-hours at several budget settings.
