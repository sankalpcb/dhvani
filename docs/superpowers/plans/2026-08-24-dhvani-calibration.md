# Dhvani Calibration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Kept honest by `tests/test_plan_docs.py`.** Every `def` and `class` this
> plan declares under a `# path/to/file.py` heading is checked against that
> file's real signature on each test run, so the plan cannot quietly describe
> an interface the code has moved on from. What is checked is names and
> parameters. The code inside each block is still an ABBREVIATED proposal,
> not a copy of the file, and the prose around it still records what was
> planned rather than what shipped — read the source for the full story.

**Goal:** Produce a measured `delta_table.json` — the lookup table the router needs to decide which segments are worth escalating — by transcribing real Indic speech with both tiers and measuring the toWER improvement per risk bucket.

**Architecture:** Two decoupled phases. Phase 1 streams IndicVoices utterances, transcribes each locally with Tier 0, scores its risk, and persists everything content-addressed — slow and free. Phase 2 takes a stratified sample across all ten risk buckets, sends only those to Chirp in dynamic batch, and computes deltas — fast and paid. Every external dependency is injected, so the whole harness tests without a model, a cloud SDK, credentials, or a network.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, stdlib `sqlite3`, `numpy`, `datasets` (new optional extra).

**Spec:** `docs/superpowers/specs/2026-08-24-dhvani-calibration-design.md`

## Global Constraints

- **No model training.** Measurement only. No training loops, no `.pkl`, no `sklearn.fit`.
- **Total external spend must never exceed USD 20.** Every paid call goes through `Store.reserve_spend()`, which does the ceiling check and insert in ONE SQL statement, BEFORE the call. Never split it.
- **Expected spend for a full run is ~$0.75**, against USD 10 authorized. Phase 2 prints an estimate and requires `--confirm` before its first paid call.
- **Replay mode must NEVER fall back to live.** A missing fixture is a hard `FixtureMissing`.
- **Every pure function must be deterministic.** Same input plus same `POLICY_ID` yields byte-identical output. Tie-breaks sort by `segment_id`.
- **Goal G5:** the full suite must run with NO ML dependencies, NO cloud SDK, NO credentials, NO network. `torch`, `transformers`, `google-cloud-speech`, `datasets` all live in optional extras.
- **Calibration bypasses the segmenter.** One dataset utterance = one segment. See spec §1.2.
- **`MIN_BUCKET_SAMPLES = 20`.** A bucket with fewer samples is OMITTED from the table, never included with a noisy average.
- **Partial failure writes nothing.** If `BudgetExceeded` raises mid-run, `delta_table.json` is not written.

---

## Existing interfaces you consume (do not modify)

```python
# dhvani/store.py
class Store:  # context manager; has a _migrate() pattern
    def __init__(self, path: str, timeout: float=30.0): ...
    def put_segment(self, segment_id, source_id, t_start_ms, t_end_ms, lang_hint=None): ...
    def put_hypothesis(self, segment_id, tier, text, signals, cost_usd, variant_key: str='') -> bool: ...
    def get_hypothesis(self, segment_id, tier, variant_key: str=''): ...
    def reserve_spend(self, tier: str, cost_usd: float) -> None: ...  # atomic; raises BudgetExceeded
    def total_spend(self) -> float: ...

class BudgetExceeded(RuntimeError): ...

# dhvani/ids.py
def segment_id(pcm: np.ndarray) -> str: ...  # SHA256 of mono int16 PCM

# dhvani/audio.py
def normalize(samples: np.ndarray, src_rate: int) -> np.ndarray: ...  # mono int16 @ 16kHz

# dhvani/segmenter.py
@dataclass(frozen=True)
class Segment:
    segment_id: str
    t_start_ms: int
    t_end_ms: int
    pcm: np.ndarray = field(compare=False, repr=False)

# dhvani/scorer.py
def extract(text: str, decoder_signals: dict, duration_ms: int) -> Features: ...
def risk(f: Features) -> float: ...

# dhvani/router.py
def bucket_of(risk: float) -> str: ...  # "0.6-0.7"; clamps to [0,1]
N_BUCKETS: int

# dhvani/delta_table.py
def build(rows: list[dict]) -> dict: ...  # rows: {risk, reference, tier0_text, tier1_text}

# dhvani/backends/base.py
class Recorded:  # mode in record|replay|live
    def __init__(self, inner: Backend, mode: Mode, fixture_dir: str, store=None): ...

# dhvani/backends/tier0_conformer.py
class Tier0Conformer:  # .name="tier0", .variant_key
    def __init__(self, model=None, lang: str='hi', model_id: str=MODEL_ID): ...

# dhvani/backends/tier1_chirp.py
class Tier1Chirp:  # .name="tier1", .variant_key
    def __init__(self, client=None, lang: str='hi-IN', recognizer: str=''): ...
    def cost_per_call(self, segment) -> float: ...

def cost_for_duration_ms(duration_ms: int) -> float: ...  # THE single Tier 1 cost model

# dhvani/config.py
POLICY_ID: str
RISK_WEIGHTS: dict
```

---

## File Structure

| File | Responsibility |
|---|---|
| `dhvani/store.py` (modify) | + `references_` table, `put_reference`, `get_reference` |
| `dhvani/corpus.py` (new) | `CorpusItem`, `FakeCorpus`, `IndicVoicesCorpus` |
| `dhvani/calibrate.py` (new) | `stratify`, `collect`, `escalate_selected`, `write_table` (row assembly is inlined in `escalate_selected`) |
| `dhvani/cli_calibrate.py` (new) | `collect` / `escalate` subcommands, `--dry-run`, `--confirm` |
| `pyproject.toml` (modify) | + `data` extra, + `dhvani-calibrate` script |

---

## Task 1: `references_` table

**Files:**
- Modify: `dhvani/store.py`
- Test: `tests/test_store_references.py`

**Interfaces:**
- Consumes: `Store`
- Produces:
  - `Store.put_reference(segment_id, reference, lang, speaker_id=None, district=None) -> bool`
  - `Store.get_reference(segment_id) -> dict | None` — keys `reference, lang, speaker_id, district`

- [x] **Step 1: Write the failing test**

```python
# tests/test_store_references.py
import pytest
from dhvani.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_reference_round_trips(store):
    store.put_reference("a" * 64, "नमस्ते", "hi-IN", "spk1", "Pune")
    got = store.get_reference("a" * 64)
    assert got == {"reference": "नमस्ते", "lang": "hi-IN",
                   "speaker_id": "spk1", "district": "Pune"}


def test_put_reference_is_idempotent(store):
    assert store.put_reference("a" * 64, "first", "hi-IN") is True
    assert store.put_reference("a" * 64, "SECOND", "hi-IN") is False
    assert store.get_reference("a" * 64)["reference"] == "first"


def test_get_missing_reference_returns_none(store):
    assert store.get_reference("nope") is None


def test_speaker_and_district_are_optional(store):
    store.put_reference("b" * 64, "text", "kn-IN")
    got = store.get_reference("b" * 64)
    assert got["speaker_id"] is None and got["district"] is None


def test_references_do_not_disturb_hypotheses(store):
    """The new table must not interfere with the existing content-addressed cache."""
    store.put_segment("c" * 64, "vid1", 0, 3000)
    store.put_hypothesis("c" * 64, "tier0", "hyp", {}, 0.0)
    store.put_reference("c" * 64, "ref", "hi-IN")
    assert store.get_hypothesis("c" * 64, "tier0")["text"] == "hyp"
    assert store.get_reference("c" * 64)["reference"] == "ref"
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store_references.py -v`
Expected: FAIL with `AttributeError: 'Store' object has no attribute 'put_reference'`

- [x] **Step 3: Extend the schema**

Append to `SCHEMA` in `dhvani/store.py`. Note the trailing underscore — `references` is a SQL reserved word.

```sql
CREATE TABLE IF NOT EXISTS references_ (
  segment_id  TEXT PRIMARY KEY,
  reference   TEXT NOT NULL,
  lang        TEXT NOT NULL,
  speaker_id  TEXT,
  district    TEXT,
  created_at  INTEGER NOT NULL
);
```

- [x] **Step 4: Add the methods inside the `Store` class**

```python
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
```

- [x] **Step 5: Run tests**

Run: `uv run pytest tests/test_store_references.py -v`
Expected: PASS, 5 tests

- [x] **Step 6: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/store.py tests/test_store_references.py
git commit -m "feat: references_ table for calibration ground truth"
```

---

## Task 2: Corpus source

**Files:**
- Create: `dhvani/corpus.py`
- Modify: `pyproject.toml` (add `data` extra)
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `dhvani.audio.normalize`, `dhvani.ids.segment_id`
- Produces:
  - `CorpusItem` — frozen dataclass: `segment_id: str`, `pcm: np.ndarray`, `reference: str`, `lang: str`, `speaker_id: str`, `district: str`, `duration_ms: int`
  - `FakeCorpus(items: list[tuple])` — `.stream(lang, limit) -> Iterator[CorpusItem]`
  - `IndicVoicesCorpus(dataset_id="ai4bharat/IndicVoices")` — same `.stream` contract
  - `SCHEMA_PROBE_ROWS = 50` and `_extract_item_from_row(row, lang)` — the fail-loud guard
    added in review: examining this many rows without producing one item raises, naming the
    fields expected and the keys actually present, so a wrong schema guess cannot masquerade
    as an empty corpus
  - `disjoint_by(items, key: str) -> bool` — True when no value of `key` repeats

- [x] **Step 1: Write the failing test**

```python
# tests/test_corpus.py
import numpy as np
import pytest
from dhvani.corpus import CorpusItem, FakeCorpus, disjoint_by


def _raw(seed, seconds=2.0, sr=16000):
    rng = np.random.default_rng(seed)
    return 0.3 * rng.standard_normal(int(sr * seconds)), sr


def _fake(n=4):
    return FakeCorpus([
        (*_raw(i), f"reference-{i}", "hi-IN", f"spk{i}", f"district{i}")
        for i in range(n)
    ])


def test_stream_yields_corpus_items():
    items = list(_fake().stream("hi-IN", limit=4))
    assert len(items) == 4
    assert all(isinstance(i, CorpusItem) for i in items)


def test_limit_is_respected():
    assert len(list(_fake(10).stream("hi-IN", limit=3))) == 3


def test_pcm_is_normalized_int16():
    item = next(_fake().stream("hi-IN", limit=1))
    assert item.pcm.dtype == np.int16
    assert item.pcm.ndim == 1


def test_segment_id_is_content_addressed():
    """Same audio must yield the same id across separate streams."""
    a = next(_fake().stream("hi-IN", limit=1))
    b = next(_fake().stream("hi-IN", limit=1))
    assert a.segment_id == b.segment_id
    assert len(a.segment_id) == 64


def test_duration_is_derived_from_pcm():
    item = next(_fake().stream("hi-IN", limit=1))
    assert 1900 <= item.duration_ms <= 2100


def test_language_filter_excludes_other_languages():
    corpus = FakeCorpus([
        (*_raw(1), "hindi", "hi-IN", "s1", "d1"),
        (*_raw(2), "kannada", "kn-IN", "s2", "d2"),
    ])
    assert [i.lang for i in corpus.stream("kn-IN", limit=10)] == ["kn-IN"]


def test_disjoint_by_detects_repeats():
    items = list(_fake(3).stream("hi-IN", limit=3))
    assert disjoint_by(items, "speaker_id") is True
    assert disjoint_by(items + items[:1], "speaker_id") is False


def test_disjoint_by_ignores_none_values():
    """Missing metadata must not be treated as a collision."""
    items = [CorpusItem("a" * 64, np.zeros(10, np.int16), "r", "hi-IN", None, None, 10),
             CorpusItem("b" * 64, np.zeros(10, np.int16), "r", "hi-IN", None, None, 10)]
    assert disjoint_by(items, "speaker_id") is True
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.corpus'`

- [x] **Step 3: Add the `data` extra**

In `pyproject.toml`, alongside the existing `models` and `cloud` extras:

```toml
data = ["datasets>=2.19", "soundfile>=0.12"]
```

- [x] **Step 4: Write the implementation**

```python
# dhvani/corpus.py
"""Calibration corpus sources.

Calibration bypasses the segmenter: one dataset utterance is one segment, so
each segment keeps its own reference transcript (spec §1.2). segment_id is
still SHA256 of normalized PCM, so the store, cache and fixture layers work
unchanged.

IndicVoicesCorpus imports `datasets` lazily, inside stream(), so importing
this module costs nothing and the test suite runs with the `data` extra
absent (goal G5).
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from dhvani.audio import normalize
from dhvani.config import SAMPLE_RATE
from dhvani.ids import segment_id as compute_id


@dataclass(frozen=True)
class CorpusItem:
    segment_id: str
    pcm: np.ndarray
    reference: str
    lang: str
    speaker_id: str | None
    district: str | None
    duration_ms: int


def _make_item(raw, src_rate, reference, lang, speaker_id, district) -> CorpusItem:
    pcm = normalize(np.asarray(raw), src_rate)
    return CorpusItem(
        segment_id=compute_id(pcm),
        pcm=pcm,
        reference=reference,
        lang=lang,
        speaker_id=speaker_id,
        district=district,
        duration_ms=int(len(pcm) * 1000 / SAMPLE_RATE),
    )


def disjoint_by(items, key: str) -> bool:
    """True when no non-None value of `key` repeats across items.

    Used to assert speaker- and district-disjointness on the SELECTED set,
    not merely on what was requested. None values are ignored: absent
    metadata is not a collision.
    """
    seen = set()
    for item in items:
        value = getattr(item, key)
        if value is None:
            continue
        if value in seen:
            return False
        seen.add(value)
    return True


class FakeCorpus:
    """In-memory corpus for tests. No download, no `datasets` dependency."""

    def __init__(self, items, dataset_id: str='fake'):
        # items: list of (raw_audio, src_rate, reference, lang, speaker_id, district)
        self._items = list(items)

    def stream(self, lang: str, limit: int) -> Iterator[CorpusItem]:
        count = 0
        for raw, rate, reference, item_lang, speaker, district in self._items:
            if item_lang != lang:
                continue
            if count >= limit:
                return
            yield _make_item(raw, rate, reference, item_lang, speaker, district)
            count += 1


class IndicVoicesCorpus:
    """Streams AI4Bharat IndicVoices from HuggingFace.

    Streaming rather than downloading: the full corpus is thousands of hours
    and calibration needs a few thousand utterances.
    """

    def __init__(self, dataset_id: str = "ai4bharat/IndicVoices"):
        self.dataset_id = dataset_id

    def stream(self, lang: str, limit: int) -> Iterator[CorpusItem]:
        from datasets import load_dataset  # lazy: keeps the `data` extra optional

        config = lang.split("-")[0]  # "hi-IN" -> "hi"
        ds = load_dataset(self.dataset_id, config, split="train", streaming=True)

        count = 0
        for row in ds:
            if count >= limit:
                return
            audio = row.get("audio")
            reference = row.get("text") or row.get("transcript") or ""
            if not audio or not reference.strip():
                continue
            yield _make_item(
                audio["array"], audio["sampling_rate"], reference, lang,
                row.get("speaker_id"), row.get("district"),
            )
            count += 1
```

- [x] **Step 5: Run tests**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: PASS, 8 tests

- [x] **Step 6: Confirm G5 holds**

Run: `uv run python -c "import datasets"` — expect `ModuleNotFoundError`. Then `uv run pytest -q` and confirm everything still passes. Importing `dhvani.corpus` must not require `datasets`.

- [x] **Step 7: Commit**

```bash
git add dhvani/corpus.py pyproject.toml tests/test_corpus.py
git commit -m "feat: calibration corpus sources with lazy datasets import"
```

---

## Task 3: Stratified sampling (pure)

**Files:**
- Create: `dhvani/calibrate.py`
- Test: `tests/test_stratify.py`

**Interfaces:**
- Consumes: `dhvani.router.bucket_of`
- Produces:
  - `dhvani.calibrate.MIN_BUCKET_SAMPLES = 20`
  - `dhvani.calibrate.N_PER_BUCKET = 100`
  - `stratify(scored: list[dict], n_per_bucket=N_PER_BUCKET) -> list[dict]`
    where each `scored` item has at least `segment_id` and `risk`
  - `histogram(scored: list[dict]) -> dict[str, int]` — bucket label -> count

- [x] **Step 1: Write the failing test**

```python
# tests/test_stratify.py
import pytest
from dhvani.calibrate import stratify, histogram, MIN_BUCKET_SAMPLES


def _scored(n, risk, prefix="s"):
    return [{"segment_id": f"{prefix}{i:04d}" + "0" * 58, "risk": risk} for i in range(n)]


def test_histogram_counts_by_bucket():
    scored = _scored(3, 0.65) + _scored(2, 0.05, "t")
    assert histogram(scored) == {"0.0-0.1": 2, "0.6-0.7": 3}


def test_histogram_of_empty_is_empty():
    assert histogram([]) == {}


def test_stratify_caps_at_n_per_bucket():
    assert len(stratify(_scored(500, 0.65), n_per_bucket=100)) == 100


def test_stratify_takes_all_when_under_the_cap():
    assert len(stratify(_scored(40, 0.65), n_per_bucket=100)) == 40


def test_stratify_omits_thin_buckets():
    """A bucket under MIN_BUCKET_SAMPLES is dropped, not sampled noisily."""
    thin = _scored(MIN_BUCKET_SAMPLES - 1, 0.35, "thin")
    fat = _scored(50, 0.65, "fat")
    out = stratify(thin + fat)
    assert {s["segment_id"][:3] for s in out} == {"fat"}


def test_stratify_keeps_a_bucket_exactly_at_the_floor():
    at_floor = _scored(MIN_BUCKET_SAMPLES, 0.35)
    assert len(stratify(at_floor)) == MIN_BUCKET_SAMPLES


def test_stratify_spans_multiple_buckets():
    out = stratify(_scored(30, 0.15, "a") + _scored(30, 0.85, "b"))
    prefixes = {s["segment_id"][0] for s in out}
    assert prefixes == {"a", "b"}


def test_stratify_is_deterministic_and_order_independent():
    """Invariant I5: a re-run must select the same segments, so the cache hits."""
    scored = _scored(200, 0.65)
    assert stratify(scored) == stratify(list(reversed(scored)))


def test_stratify_of_empty_is_empty():
    assert stratify([]) == []
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stratify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.calibrate'`

- [x] **Step 3: Write the implementation**

```python
# dhvani/calibrate.py
"""Calibration harness: measures the delta table the router needs.

Spec: docs/superpowers/specs/2026-08-24-dhvani-calibration-design.md

The router cannot pick its own calibration set — it selects by delta, and
delta is what is being measured (spec §1.1). So calibration escalates a
STRATIFIED sample across every risk bucket, deliberately including low-risk
ones, because that is the only way to discover the negative deltas that
invariant I3 exists to filter.
"""

from collections import defaultdict

from dhvani.router import bucket_of

# A bucket with fewer samples than this is OMITTED from the table rather
# than included with a noisy average. Omission degrades to "do not
# escalate"; a noisy average degrades to "escalate wrongly, and pay".
MIN_BUCKET_SAMPLES = 20

N_PER_BUCKET = 100


def histogram(scored: list[dict]) -> dict:
    """Bucket label -> count. Printed before any paid call so the risk
    distribution is visible while it is still free to act on."""
    counts: dict[str, int] = defaultdict(int)
    for item in scored:
        counts[bucket_of(item["risk"])] += 1
    return dict(sorted(counts.items()))


def stratify(scored: list[dict], n_per_bucket: int = N_PER_BUCKET) -> list[dict]:
    """Sample up to n_per_bucket from each risk bucket. Pure and deterministic.

    Selection sorts by segment_id, so a re-run picks the same segments and
    hits the content-addressed cache instead of paying again.
    """
    by_bucket: dict[str, list] = defaultdict(list)
    for item in scored:
        by_bucket[bucket_of(item["risk"])].append(item)

    chosen: list[dict] = []
    for bucket in sorted(by_bucket):
        members = by_bucket[bucket]
        if len(members) < MIN_BUCKET_SAMPLES:
            continue
        members.sort(key=lambda i: i["segment_id"])
        chosen.extend(members[:n_per_bucket])
    return chosen
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/test_stratify.py -v`
Expected: PASS, 9 tests

- [x] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/calibrate.py tests/test_stratify.py
git commit -m "feat: deterministic stratified sampling across risk buckets"
```

---

## Task 4: Phase 1 — collect

**Files:**
- Modify: `dhvani/calibrate.py`
- Test: `tests/test_collect.py`

**Interfaces:**
- Consumes: `Store.put_segment/put_hypothesis/get_hypothesis/put_reference`, `scorer.extract/risk`, `Segment`, `CorpusItem`
- Produces:
  - `collect(corpus, tier0, store, langs: list[str], per_lang: int) -> list[dict]`
    returns scored items: `{segment_id, risk, lang, duration_ms}`

- [x] **Step 1: Write the failing test**

```python
# tests/test_collect.py
import numpy as np
import pytest

from dhvani.calibrate import collect
from dhvani.corpus import FakeCorpus
from dhvani.store import Store


class StubTier0:
    name = "tier0"
    variant_key = "tier0|hi|m"

    def __init__(self, lang='hi'):
        self.calls = 0

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        self.calls += 1
        return {"text": "नमस्ते world", "signals": {"ctc_rnnt_disagreement": 0.4}}


def _corpus(n=3, lang='hi-IN', dataset_id='fake', seed=0):
    rng = np.random.default_rng(0)
    return FakeCorpus([
        (0.3 * rng.standard_normal(32000), 16000, f"ref-{i}", lang, f"spk{i}", f"d{i}")
        for i in range(n)
    ])


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_collect_returns_one_scored_item_per_utterance(store, pcm_dir):
    out = collect(_corpus(3), StubTier0(), store, ["hi-IN"], per_lang=3)
    assert len(out) == 3
    assert all(0.0 <= s["risk"] <= 1.0 for s in out)


def test_collect_persists_reference_and_hypothesis(store, pcm_dir):
    out = collect(_corpus(1), StubTier0(), store, ["hi-IN"], per_lang=1)
    sid = out[0]["segment_id"]
    assert store.get_reference(sid)["reference"] == "ref-0"
    assert store.get_hypothesis(sid, "tier0", "tier0|hi|m")["text"] == "नमस्ते world"


def test_collect_is_resumable_and_does_not_retranscribe(store, pcm_dir):
    """The property that makes a multi-hour run survivable."""
    tier0 = StubTier0()
    collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    first = tier0.calls
    collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    assert tier0.calls == first, "cached segments must not be re-transcribed"


def test_collect_scores_identically_on_a_cached_rerun(store, pcm_dir):
    tier0 = StubTier0()
    a = collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    b = collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    assert a == b


def test_collect_spans_requested_languages(store, pcm_dir):
    rng = np.random.default_rng(1)
    corpus = FakeCorpus([
        (0.3 * rng.standard_normal(32000), 16000, "h", "hi-IN", "s1", "d1"),
        (0.3 * rng.standard_normal(32000), 16000, "k", "kn-IN", "s2", "d2"),
    ])
    out = collect(corpus, StubTier0(), store, ["hi-IN", "kn-IN"], per_lang=1)
    assert {s["lang"] for s in out} == {"hi-IN", "kn-IN"}


def test_collect_carries_duration_for_later_pricing(store, pcm_dir):
    out = collect(_corpus(1), StubTier0(), store, ["hi-IN"], per_lang=1)
    assert out[0]["duration_ms"] == 2000
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_collect.py -v`
Expected: FAIL with `ImportError: cannot import name 'collect'`

- [x] **Step 3: Write the implementation**

Append to `dhvani/calibrate.py`:

```python
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import Segment


def collect(corpus, tier0, store, langs, per_lang: int) -> list[dict]:
    """Phase 1: transcribe and score a corpus locally. Slow, free, resumable.

    One utterance is one segment (spec §1.2), so the segmenter is bypassed
    and every segment keeps its own reference. Already-transcribed segments
    are read from the store rather than re-run — that is what lets a
    multi-hour run be killed and restarted without losing work.
    """
    scored: list[dict] = []

    for lang in langs:
        for item in corpus.stream(lang, limit=per_lang):
            store.put_segment(item.segment_id, f"calib:{lang}", 0,
                              item.duration_ms, lang)
            store.put_reference(item.segment_id, item.reference, lang,
                                item.speaker_id, item.district)

            cached = store.get_hypothesis(item.segment_id, "tier0", tier0.variant_key)
            if cached is None:
                segment = Segment(item.segment_id, 0, item.duration_ms, item.pcm)
                result = tier0.transcribe(segment)
                store.put_hypothesis(item.segment_id, "tier0", result["text"],
                                     result["signals"], 0.0, tier0.variant_key)
            else:
                result = {"text": cached["text"], "signals": cached["signals"]}

            features = extract(result["text"], result["signals"], item.duration_ms)
            scored.append({
                "segment_id": item.segment_id,
                "risk": compute_risk(features),
                "lang": lang,
                "duration_ms": item.duration_ms,
            })

    return scored
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/test_collect.py -v`
Expected: PASS, 6 tests

- [x] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/calibrate.py tests/test_collect.py
git commit -m "feat: phase 1 collect — resumable local transcription and scoring"
```

---

## Task 5: Phase 2 — escalate, assemble, write

**Files:**
- Modify: `dhvani/calibrate.py`
- Test: `tests/test_escalate_phase.py`

**Interfaces:**
- Consumes: `Store.reserve_spend/get_reference/get_hypothesis/put_hypothesis`, `tier1_chirp.cost_for_duration_ms`, `delta_table.build`, `config.POLICY_ID/RISK_WEIGHTS`
- Produces:
  - `estimate_cost(selected: list[dict]) -> float`
  - `escalate_selected(selected, tier1, store, segments_by_id: dict, tier0_variant: str = "") -> list[dict]` — returns rows
  - `write_table(rows, selected, path, spend_usd, langs) -> dict`

- [x] **Step 1: Write the failing test**

```python
# tests/test_escalate_phase.py
import json
import numpy as np
import pytest

from dhvani.calibrate import estimate_cost, escalate_selected, write_table
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded

# NOTE: test_budget_failure_leaves_no_table_behind is intentionally omitted
# here. It drives the CLI rather than escalate_selected directly, because
# the "writes nothing" guarantee is a property of the CLI's ordering
# (escalate before write_table), not of escalate_selected alone. It belongs
# to Task 6, which is where dhvani.cli_calibrate is created.


class StubTier1:
    name = "tier1"
    variant_key = "tier1|hi-IN|"

    def cost_per_call(self, segment):
        from dhvani.backends.tier1_chirp import cost_for_duration_ms
        return cost_for_duration_ms(segment.t_end_ms - segment.t_start_ms)

    def transcribe(self, segment):
        return {"text": "chirp output", "signals": {}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def _selected(n=3):
    return [{"segment_id": f"s{i:04d}" + "0" * 59, "risk": 0.65,
             "lang": "hi-IN", "duration_ms": 3000} for i in range(n)]


def _segments(selected):
    return {s["segment_id"]: Segment(s["segment_id"], 0, s["duration_ms"],
                                     np.zeros(10, dtype=np.int16))
            for s in selected}


def _seed_refs(store, selected):
    for s in selected:
        store.put_reference(s["segment_id"], "alpha beta gamma delta", "hi-IN")
        store.put_hypothesis(s["segment_id"], "tier0", "alpha beta gamma WRONG",
                             {}, 0.0, "tier0|hi|m")


def test_estimate_uses_the_single_cost_model():
    from dhvani.backends.tier1_chirp import cost_for_duration_ms
    assert estimate_cost(_selected(4)) == pytest.approx(4 * cost_for_duration_ms(3000))


def test_estimate_of_empty_is_zero():
    assert estimate_cost([]) == 0.0


def test_escalate_produces_one_row_per_selected_segment(store):
    sel = _selected(3)
    _seed_refs(store, sel)
    rows = escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert len(rows) == 3
    assert set(rows[0]) == {"risk", "reference", "tier0_text", "tier1_text"}


def test_escalate_reserves_spend_before_calling(store, tmp_path):
    sel = _selected(2)
    _seed_refs(store, sel)
    escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert store.total_spend() > 0.0


def test_rerunning_escalation_reserves_nothing_further(store, tmp_path):
    """Idempotent spend: cached tier1 hypotheses must not be re-paid."""
    sel = _selected(3)
    _seed_refs(store, sel)
    escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    after_first = store.total_spend()
    escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert store.total_spend() == pytest.approx(after_first)


def test_escalation_fails_closed_at_the_ceiling(store, tmp_path):
    sel = _selected(3)
    _seed_refs(store, sel)
    store.reserve_spend("tier1", 20.0 - 0.0001)
    with pytest.raises(BudgetExceeded):
        escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")


def test_segments_missing_a_reference_are_skipped(store):
    """A segment with no ground truth cannot produce a meaningful delta."""
    sel = _selected(2)
    _seed_refs(store, sel[:1])
    rows = escalate_selected(sel, StubTier1(), store, _segments(sel), "tier0|hi|m")
    assert len(rows) == 1


def test_write_table_records_provenance(tmp_path):
    rows = [{"risk": 0.65, "reference": "a b c d",
             "tier0_text": "a b c X", "tier1_text": "a b c d"}] * 25
    sel = _selected(25)
    path = tmp_path / "delta_table.json"
    table = write_table(rows, sel, str(path), spend_usd=0.019, langs=["hi-IN"])

    written = json.loads(path.read_text())
    assert "tier1" in written
    meta = written["meta"]
    assert meta["policy_id"] and meta["risk_weights"]
    assert meta["segments_escalated"] == 25
    assert meta["spend_usd"] == pytest.approx(0.019)
    assert meta["languages"] == ["hi-IN"]
    assert table["tier1"] == written["tier1"]


def test_write_table_records_per_bucket_counts(tmp_path):
    rows = [{"risk": 0.65, "reference": "a b", "tier0_text": "a b",
             "tier1_text": "a b"}] * 22
    sel = _selected(22)
    path = tmp_path / "t.json"
    write_table(rows, sel, str(path), spend_usd=0.0, langs=["hi-IN"])
    assert json.loads(path.read_text())["meta"]["bucket_n"]["0.6-0.7"] == 22
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_escalate_phase.py -v`
Expected: FAIL with `ImportError: cannot import name 'estimate_cost'`

- [x] **Step 3: Write the implementation**

Append to `dhvani/calibrate.py`:

```python
import json
from datetime import date

from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.config import POLICY_ID, RISK_WEIGHTS
from dhvani.delta_table import build as build_delta_table


def estimate_cost(selected: list[dict]) -> float:
    """Pre-flight estimate, printed before the first paid call.

    Prices through cost_for_duration_ms — the single Tier 1 cost model — so
    the estimate cannot drift from what is actually reserved.
    """
    return sum(cost_for_duration_ms(s["duration_ms"]) for s in selected)


def escalate_selected(selected, tier1, store, segments_by_id,
                      tier0_variant: str = "") -> list[dict]:
    """Phase 2: run Tier 1 over the stratified sample and assemble rows.

    Spend is reserved BEFORE each paid call, and a cached Tier 1 hypothesis
    is reused without reserving again — re-running a calibration pass must
    not re-charge for work already done.
    """
    rows: list[dict] = []

    for item in selected:
        segment_id = item["segment_id"]
        reference = store.get_reference(segment_id)
        tier0 = store.get_hypothesis(segment_id, "tier0", tier0_variant)
        if reference is None or tier0 is None:
            # No ground truth or no Tier 0 output means no meaningful delta.
            continue

        cached = store.get_hypothesis(segment_id, "tier1", tier1.variant_key)
        if cached is None:
            segment = segments_by_id[segment_id]
            cost = tier1.cost_per_call(segment)
            store.reserve_spend(tier1.name, cost)
            result = tier1.transcribe(segment)
            store.put_hypothesis(segment_id, "tier1", result["text"],
                                 result.get("signals", {}), cost, tier1.variant_key)
            tier1_text = result["text"]
        else:
            tier1_text = cached["text"]

        rows.append({
            "risk": item["risk"],
            "reference": reference["reference"],
            "tier0_text": tier0["text"],
            "tier1_text": tier1_text,
        })

    return rows


def write_table(rows, selected, path: str, spend_usd: float, langs) -> dict:
    """Build the delta table and write it with provenance.

    build()'s contract is untouched; meta is additive. Nothing enforces meta
    (spec non-goal N3), but a stale table becomes visible rather than silent.
    """
    table = build_delta_table(rows)

    bucket_n: dict[str, int] = defaultdict(int)
    for row in rows:
        bucket_n[bucket_of(row["risk"])] += 1

    payload = dict(table)
    payload["meta"] = {
        "policy_id": POLICY_ID,
        "risk_weights": dict(RISK_WEIGHTS),
        "bucket_n": dict(sorted(bucket_n.items())),
        "languages": list(langs),
        "segments_escalated": len(selected),
        "spend_usd": round(spend_usd, 6),
        "measured_at": date.today().isoformat(),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return table
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/test_escalate_phase.py -v`
Expected: PASS, 11 tests

- [x] **Step 5: Full suite and commit**

```bash
uv run pytest -q
git add dhvani/calibrate.py tests/test_escalate_phase.py
git commit -m "feat: phase 2 escalation, row assembly, and provenance-carrying table"
```

---

## Task 6: CLI

**Files:**
- Create: `dhvani/cli_calibrate.py`
- Modify: `pyproject.toml` (add `dhvani-calibrate` script)
- Test: `tests/test_cli_calibrate.py`

**Interfaces:**
- Consumes: everything above
- Produces: `main(argv=None) -> int` with `collect` and `escalate` subcommands

- [x] **Step 1: Write the failing test**

```python
# tests/test_cli_calibrate.py
import json

import pytest

from dhvani.cli_calibrate import main


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_collect_subcommand_exists():
    with pytest.raises(SystemExit) as exc:
        main(["collect", "--help"])
    assert exc.value.code == 0


def test_escalate_subcommand_exists():
    with pytest.raises(SystemExit) as exc:
        main(["escalate", "--help"])
    assert exc.value.code == 0


def _seed_scored(tmp_path, n=25, risk=0.65, prefix='s'):
    """A scored.json with one populated bucket, so escalate reaches its cost
    gate instead of exiting early on a missing input file."""
    scored = [{"segment_id": f"s{i:04d}" + "0" * 59, "risk": 0.65,
               "lang": "hi-IN", "duration_ms": 3000} for i in range(n)]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))
    return str(path)


def test_escalate_without_confirm_refuses_to_spend(tmp_path, capsys):
    """The cost gate: a run that would spend must not do so silently."""
    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", _seed_scored(tmp_path),
               "--out", str(tmp_path / "d.json")])
    out = capsys.readouterr()
    assert rc == 2, "must exit non-zero specifically on the confirm gate"
    assert "--confirm" in (out.out + out.err)


def test_missing_scored_input_is_a_different_failure(tmp_path, capsys):
    """Distinguishes 'no input' from 'refused to spend' — otherwise the
    confirm-gate test would pass without ever reaching the gate."""
    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", str(tmp_path / "absent.json"),
               "--out", str(tmp_path / "d.json")])
    assert rc == 1
    assert "collect" in capsys.readouterr().err


def test_dry_run_writes_no_table(tmp_path):
    path = tmp_path / "d.json"
    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", _seed_scored(tmp_path),
               "--out", str(path), "--dry-run"])
    assert rc == 0
    assert not path.exists()


def test_pyproject_declares_the_calibrate_script():
    import pathlib, tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["scripts"]["dhvani-calibrate"] == "dhvani.cli_calibrate:main"
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.cli_calibrate'`

- [x] **Step 3: Add the script entry**

In `pyproject.toml` under `[project.scripts]`:

```toml
dhvani-calibrate = "dhvani.cli_calibrate:main"
```

- [x] **Step 4: Write the implementation**

```python
# dhvani/cli_calibrate.py
"""Calibration CLI: dhvani-calibrate collect | escalate

Phase 1 (collect) is slow, free and local. Phase 2 (escalate) is fast, paid
and remote, and refuses to spend without --confirm.
"""

import argparse
import json
import sys

from dhvani.calibrate import (
    collect, escalate_selected, estimate_cost, histogram, stratify, write_table,
)
from dhvani.store import Store

DEFAULT_LANGS = ["kn-IN", "ml-IN", "hi-IN"]


def _print_histogram(scored, stream=sys.stderr):
    hist = histogram(scored)
    print(f"collected {len(scored)} segments", file=stream)
    for bucket, count in hist.items():
        print(f"  {bucket:10} {count:6d}  {'#' * min(count // 10, 50)}", file=stream)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dhvani-calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="phase 1: transcribe and score locally (free)")
    c.add_argument("--db", default="calibration.db")
    c.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    c.add_argument("--per-lang", type=int, default=1000)
    c.add_argument("--scored-out", default="scored.json")

    e = sub.add_parser("escalate", help="phase 2: stratify and run Tier 1 (paid)")
    e.add_argument("--db", default="calibration.db")
    e.add_argument("--scored-in", default="scored.json")
    e.add_argument("--out", default="delta_table.json")
    e.add_argument("--dry-run", action="store_true",
                   help="stratify, print the estimate, and exit without spending")
    e.add_argument("--confirm", action="store_true",
                   help="required before any paid call")

    args = ap.parse_args(argv)

    if args.cmd == "collect":
        from dhvani.backends.tier0_conformer import Tier0Conformer
        from dhvani.corpus import IndicVoicesCorpus

        corpus = IndicVoicesCorpus()
        with Store(args.db) as store:
            scored = collect(corpus, Tier0Conformer(), store, args.langs, args.per_lang)
        _print_histogram(scored)
        with open(args.scored_out, "w", encoding="utf-8") as fh:
            json.dump(scored, fh, indent=2)
        print(f"wrote {args.scored_out}", file=sys.stderr)
        return 0

    # escalate
    try:
        with open(args.scored_in, encoding="utf-8") as fh:
            scored = json.load(fh)
    except FileNotFoundError:
        print(f"missing {args.scored_in}; run `dhvani-calibrate collect` first",
              file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"{args.scored_in} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    selected = stratify(scored)
    estimate = estimate_cost(selected)
    _print_histogram(scored)
    print(f"stratified {len(selected)} segments; estimated cost ${estimate:.4f}",
          file=sys.stderr)

    if args.dry_run:
        print("dry run: nothing spent, no table written", file=sys.stderr)
        return 0

    if not args.confirm:
        print(f"refusing to spend ${estimate:.4f} without --confirm", file=sys.stderr)
        return 2

    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp
    from dhvani.segmenter import Segment
    import numpy as np

    with Store(args.db) as store:
        before = store.total_spend()
        tier1 = Recorded(Tier1Chirp(), "live", "fixtures", store)
        # Audio is not resent: Tier 1 reads it from GCS, and only the time
        # bounds are needed to price the call.
        segments = {s["segment_id"]: Segment(s["segment_id"], 0, s["duration_ms"],
                                             np.zeros(1, dtype=np.int16))
                    for s in selected}
        rows = escalate_selected(selected, tier1, store, segments)
        spent = store.total_spend() - before

    langs = sorted({s["lang"] for s in selected})
    write_table(rows, selected, args.out, spent, langs)
    print(f"wrote {args.out} from {len(rows)} rows; spent ${spent:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 5: Run tests**

Run: `uv run pytest tests/test_cli_calibrate.py -v`
Expected: PASS, 7 tests

- [x] **Step 6: Confirm G5 and commit**

Run `uv run python -c "import torch"`, `import datasets`, `import google.cloud` — all three must raise `ModuleNotFoundError`. Then `uv run pytest -q` must still pass, and `uv run dhvani-calibrate --help` must exit 0.

```bash
git add dhvani/cli_calibrate.py pyproject.toml tests/test_cli_calibrate.py
git commit -m "feat: dhvani-calibrate CLI with dry-run and spend confirmation"
```

---

## Exit Criteria

- [x] `uv run pytest` passes with no ML deps, no cloud SDK, no `datasets`, no credentials, no network
- [x] `uv run dhvani-calibrate --help` exits 0 and lists both subcommands
- [x] `escalate` without `--confirm` refuses and exits non-zero
- [x] `--dry-run` prints the histogram and estimate and writes nothing
- [x] Re-running `collect` over the same corpus makes zero Tier 0 calls
- [x] Re-running `escalate` over the same sample reserves zero additional spend
- [x] A thin bucket (n < 20) is absent from the produced table

## Deferred

Running the harness for real — reinstalling the `models` and `data` extras, streaming
IndicVoices, and spending the ~$0.75 — is an operational step after this plan lands, not part
of it. Goal G5 is verified before and after that run, not during.

**Status 2026-09-02: run, and cheaper than estimated.** Executed 2026-08-25 — 150 IndicVoices
utterances, 124 escalated to Chirp 2, **$0.1013**, against the ~$0.75 estimated here. The
result was that both measured deltas are negative, so the shipped table escalates nothing;
that is the finding, not a failed run. G5 was re-verified on 2026-09-02 with the extras
uninstalled. Note the calibration DB and PCM cache from that run are gone (both gitignored),
so the table can no longer be re-derived for $0 — a repeat means re-running both phases.
