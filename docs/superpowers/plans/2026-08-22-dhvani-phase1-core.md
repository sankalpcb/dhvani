# Dhvani Phase 1 — Synchronous Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the synchronous half of the Dhvani caption cascade — content-addressed segmentation, Tier 0 ASR, a deterministic risk scorer, a budget-constrained router, and a toWER evaluator — producing a reproducible cost/quality frontier.

**Architecture:** A pipeline of small, mostly-pure Python modules over a SQLite store. Audio is split by VAD into content-addressed segments (`SHA256` of normalized PCM), transcribed by a local IndicConformer model, scored by a deterministic weighted risk function, and routed for escalation by a pure greedy-knapsack policy under a budget. All external calls go through a record/replay `Backend` layer so the entire test suite runs offline at zero cost.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, stdlib `sqlite3`, `numpy`, `soundfile`, `scipy`, `silero-vad`, `jiwer`, `indic-transliteration`.

**Spec:** `docs/superpowers/specs/2026-08-21-dhvani-design.md`

## Global Constraints

- **No model training.** Spec non-goal N1 is binding. No training loops, no `.pkl` model artifacts, no `sklearn.fit`. Grid search over a fixed weight config is permitted; it is configuration tuning, not training.
- **Total external spend must not exceed USD 20.** Enforced in code by `MAX_SPEND_USD`, checked against the `spend` ledger before every paid call. Fail closed.
- **Replay mode must never fall back to live.** A missing fixture is a hard error.
- **Every pure function must be deterministic.** Same input plus same `POLICY_ID` yields byte-identical output. Tie-breaks sort by `segment_id`.
- **Sample rate is 16000 Hz mono int16** everywhere after `audio.normalize()`.
- **Splits must be speaker-disjoint and district-disjoint.** Never split IndicVoices randomly.
- `POLICY_ID` must be bumped whenever audio normalization, risk weights, or the delta table change.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | deps, pytest config |
| `dhvani/config.py` | `POLICY_ID`, risk weights, thresholds, `MAX_SPEND_USD` |
| `dhvani/audio.py` | deterministic PCM normalization |
| `dhvani/ids.py` | content-addressed `segment_id` |
| `dhvani/store.py` | SQLite schema, idempotent writes, spend ledger |
| `dhvani/segmenter.py` | VAD → segments |
| `dhvani/signals.py` | script entropy, romanization smell, pure feature extraction |
| `dhvani/scorer.py` | pure risk function |
| `dhvani/router.py` | pure greedy knapsack |
| `dhvani/backends/base.py` | `Backend` protocol, record/replay wrapper, spend guard |
| `dhvani/backends/tier0_conformer.py` | IndicConformer adapter |
| `dhvani/evaluator.py` | `to_wer`, WER, metrics |
| `dhvani/pipeline.py` | orchestration |
| `dhvani/cli.py` | entrypoint |
| `dhvani/delta_table.py` | builds `delta_table.json` from calibration data |
| `dhvani/report.py` | cost/quality frontier report |

---

## Task 1: Project scaffolding, audio normalization, content-addressed IDs

**Files:**
- Create: `pyproject.toml`, `dhvani/__init__.py`, `dhvani/config.py`, `dhvani/audio.py`, `dhvani/ids.py`
- Test: `tests/test_ids.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `dhvani.config.SAMPLE_RATE: int = 16000`
  - `dhvani.config.POLICY_ID: str`
  - `dhvani.config.MAX_SPEND_USD: float`
  - `dhvani.audio.normalize(samples: np.ndarray, src_rate: int) -> np.ndarray` (mono int16 @ 16kHz)
  - `dhvani.ids.segment_id(pcm: np.ndarray) -> str` (64-char hex)

- [ ] **Step 1: Create the project skeleton**

```bash
mkdir -p dhvani tests fixtures
cat > pyproject.toml <<'EOF'
[project]
name = "dhvani"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "soundfile>=0.12",
  "jiwer>=3.0",
  "indic-transliteration>=2.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
EOF
touch dhvani/__init__.py
uv venv && uv pip install -e ".[dev]"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ids.py
import numpy as np
import pytest
from dhvani.audio import normalize
from dhvani.ids import segment_id
from dhvani.config import SAMPLE_RATE


def _tone(seconds=1.0, rate=SAMPLE_RATE, freq=440.0):
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def test_same_audio_yields_same_id():
    pcm = normalize(_tone(), SAMPLE_RATE)
    assert segment_id(pcm) == segment_id(pcm.copy())


def test_id_is_64_char_hex():
    pcm = normalize(_tone(), SAMPLE_RATE)
    sid = segment_id(pcm)
    assert len(sid) == 64
    assert all(c in "0123456789abcdef" for c in sid)


def test_resampling_converges_to_same_id():
    """Same signal captured at 44.1kHz and 16kHz normalizes to near-identical PCM."""
    a = normalize(_tone(rate=44100), 44100)
    b = normalize(_tone(rate=SAMPLE_RATE), SAMPLE_RATE)
    assert len(a) == len(b)
    # Resampling is lossy; assert close, not identical.
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) < 200


def test_different_audio_yields_different_id():
    a = normalize(_tone(freq=440.0), SAMPLE_RATE)
    b = normalize(_tone(freq=880.0), SAMPLE_RATE)
    assert segment_id(a) != segment_id(b)


def test_stereo_is_downmixed_to_mono():
    stereo = np.stack([_tone(), _tone()], axis=1)
    pcm = normalize(stereo, SAMPLE_RATE)
    assert pcm.ndim == 1


def test_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="int16"):
        segment_id(np.zeros(10, dtype=np.float32))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_ids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.audio'`

- [ ] **Step 4: Write the implementation**

```python
# dhvani/config.py
"""Central configuration. Bump POLICY_ID when normalization, weights, or
the delta table change — it invalidates the content-addressed cache."""

SAMPLE_RATE = 16000
POLICY_ID = "p1-2026-08-22"
MAX_SPEND_USD = 20.0

# Risk function weights. Fixed by offline grid search (Task 11), not trained.
RISK_WEIGHTS = {
    "ctc_rnnt_disagreement": 0.35,
    "mean_neg_logprob": 0.25,
    "script_mix_entropy": 0.20,
    "romanization_smell": 0.15,
    "short_segment": 0.05,
}

TAU_SHIP = 0.30
TAU_FLAG = 0.65
```

```python
# dhvani/audio.py
"""Deterministic audio normalization. Any change here is cache-invalidating."""

import numpy as np
import scipy.signal

from dhvani.config import SAMPLE_RATE


def normalize(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample float audio in [-1, 1] to mono int16 at SAMPLE_RATE.

    Deterministic: same input always produces byte-identical output.
    """
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D array, got shape {samples.shape}")

    samples = samples.astype(np.float64)

    if src_rate != SAMPLE_RATE:
        n_out = int(round(len(samples) * SAMPLE_RATE / src_rate))
        samples = scipy.signal.resample(samples, n_out)

    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).round().astype(np.int16)
```

```python
# dhvani/ids.py
"""Content-addressed segment identity."""

import hashlib

import numpy as np


def segment_id(pcm: np.ndarray) -> str:
    """SHA256 of normalized PCM bytes.

    pcm must be mono int16 at config.SAMPLE_RATE — see audio.normalize().
    """
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm.dtype}")
    if pcm.ndim != 1:
        raise ValueError(f"expected mono 1-D array, got shape {pcm.shape}")
    return hashlib.sha256(pcm.tobytes()).hexdigest()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_ids.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml dhvani/ tests/test_ids.py
git commit -m "feat: audio normalization and content-addressed segment IDs"
```

---

## Task 2: SQLite store with idempotent writes and a spend ceiling

**Files:**
- Create: `dhvani/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `dhvani.config.MAX_SPEND_USD`
- Produces:
  - `Store(path: str)` — context manager
  - `Store.put_segment(segment_id, source_id, t_start_ms, t_end_ms, lang_hint) -> None`
  - `Store.put_hypothesis(segment_id, tier, text, signals: dict, cost_usd) -> bool` (False if already present)
  - `Store.get_hypothesis(segment_id, tier) -> dict | None`
  - `Store.total_spend() -> float`
  - `Store.record_spend(tier, cost_usd) -> None`
  - `Store.check_budget(cost_usd) -> None` (raises `BudgetExceeded`)
  - `dhvani.store.BudgetExceeded` (Exception)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
import pytest
from dhvani.store import Store, BudgetExceeded


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_hypothesis_is_idempotent(store):
    """Invariant I2: applying the same result twice is a no-op."""
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    first = store.put_hypothesis("abc", "tier0", "hello", {"x": 1}, 0.0)
    second = store.put_hypothesis("abc", "tier0", "DIFFERENT", {"x": 2}, 0.0)
    assert first is True
    assert second is False
    assert store.get_hypothesis("abc", "tier0")["text"] == "hello"


def test_get_missing_hypothesis_returns_none(store):
    assert store.get_hypothesis("nope", "tier0") is None


def test_signals_round_trip(store):
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    store.put_hypothesis("abc", "tier0", "hello", {"entropy": 0.5}, 0.0)
    assert store.get_hypothesis("abc", "tier0")["signals"] == {"entropy": 0.5}


def test_spend_accumulates(store):
    store.record_spend("tier1", 1.5)
    store.record_spend("tier1", 2.25)
    assert store.total_spend() == pytest.approx(3.75)


def test_check_budget_allows_under_ceiling(store):
    store.record_spend("tier1", 1.0)
    store.check_budget(0.5)  # must not raise


def test_check_budget_fails_closed_at_ceiling(store):
    """Invariant I4: total spend never exceeds the configured budget."""
    store.record_spend("tier1", 19.9)
    with pytest.raises(BudgetExceeded, match="would exceed"):
        store.check_budget(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.store'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/store.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add dhvani/store.py tests/test_store.py
git commit -m "feat: SQLite store with schema-enforced idempotency and spend ceiling"
```

---

## Task 3: VAD segmenter

**Files:**
- Create: `dhvani/segmenter.py`
- Test: `tests/test_segmenter.py`

**Interfaces:**
- Consumes: `dhvani.audio.normalize`, `dhvani.ids.segment_id`, `dhvani.config.SAMPLE_RATE`
- Produces:
  - `dhvani.segmenter.Segment` — frozen dataclass with fields `segment_id: str`, `t_start_ms: int`, `t_end_ms: int`, `pcm: np.ndarray`
  - `dhvani.segmenter.segment(pcm: np.ndarray, min_ms=2000, max_ms=8000) -> list[Segment]`

Energy-based VAD is used rather than Silero to keep the test suite dependency-free and
deterministic. Swapping in Silero later is a one-function change behind this interface.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segmenter.py
import numpy as np
from dhvani.config import SAMPLE_RATE
from dhvani.segmenter import segment


def _speech(seconds):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float64)


def _silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds))


def _pcm(x):
    return (np.clip(x, -1, 1) * 32767).round().astype(np.int16)


def test_silence_only_yields_no_segments():
    assert segment(_pcm(_silence(5.0))) == []


def test_single_speech_burst_yields_one_segment():
    audio = np.concatenate([_silence(0.5), _speech(3.0), _silence(0.5)])
    segs = segment(_pcm(audio))
    assert len(segs) == 1


def test_two_bursts_separated_by_silence_yield_two_segments():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(3.0)])
    segs = segment(_pcm(audio))
    assert len(segs) == 2


def test_long_burst_is_split_at_max_duration():
    segs = segment(_pcm(_speech(20.0)), max_ms=8000)
    assert len(segs) >= 3
    assert all(s.t_end_ms - s.t_start_ms <= 8000 for s in segs)


def test_segments_are_time_ordered_and_non_overlapping():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(3.0)])
    segs = segment(_pcm(audio))
    for a, b in zip(segs, segs[1:]):
        assert a.t_end_ms <= b.t_start_ms


def test_segment_ids_are_populated_and_unique():
    audio = np.concatenate([_speech(3.0), _silence(1.5), _speech(4.0)])
    segs = segment(_pcm(audio))
    ids = [s.segment_id for s in segs]
    assert all(len(i) == 64 for i in ids)
    assert len(set(ids)) == len(ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segmenter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.segmenter'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/segmenter.py
"""Energy-based voice activity detection producing caption-sized segments."""

from dataclasses import dataclass, field

import numpy as np

from dhvani.config import SAMPLE_RATE
from dhvani.ids import segment_id as compute_id

FRAME_MS = 30
SILENCE_RATIO = 0.02   # frame RMS below this fraction of peak counts as silence
MIN_SILENCE_MS = 400   # silence shorter than this does not split a segment


@dataclass(frozen=True)
class Segment:
    segment_id: str
    t_start_ms: int
    t_end_ms: int
    pcm: np.ndarray = field(compare=False, repr=False)


def _voiced_frames(pcm: np.ndarray, frame_len: int) -> np.ndarray:
    n = len(pcm) // frame_len
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = pcm[: n * frame_len].reshape(n, frame_len).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    peak = rms.max()
    if peak == 0:
        return np.zeros(n, dtype=bool)
    return rms > (SILENCE_RATIO * peak)


def _runs(voiced: np.ndarray, gap_frames: int) -> list[tuple[int, int]]:
    """Contiguous voiced runs, bridging silences shorter than gap_frames."""
    runs, start, silence = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_frames:
                runs.append((start, i - silence + 1))
                start = None
    if start is not None:
        runs.append((start, len(voiced)))
    return runs


def segment(pcm: np.ndarray, min_ms: int = 2000, max_ms: int = 8000) -> list[Segment]:
    """Split int16 PCM into caption-sized segments. Deterministic."""
    frame_len = SAMPLE_RATE * FRAME_MS // 1000
    voiced = _voiced_frames(pcm, frame_len)
    gap_frames = MIN_SILENCE_MS // FRAME_MS

    max_frames = max_ms // FRAME_MS
    out: list[Segment] = []

    for run_start, run_end in _runs(voiced, gap_frames):
        cursor = run_start
        while cursor < run_end:
            chunk_end = min(cursor + max_frames, run_end)
            s0, s1 = cursor * frame_len, chunk_end * frame_len
            piece = pcm[s0:s1]
            if len(piece) == 0:
                break
            out.append(Segment(
                segment_id=compute_id(piece),
                t_start_ms=int(s0 * 1000 / SAMPLE_RATE),
                t_end_ms=int(s1 * 1000 / SAMPLE_RATE),
                pcm=piece,
            ))
            cursor = chunk_end

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_segmenter.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add dhvani/segmenter.py tests/test_segmenter.py
git commit -m "feat: energy-based VAD segmenter with content-addressed segments"
```

---

## Task 4: Signal extraction — script entropy and romanization smell

**Files:**
- Create: `dhvani/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `dhvani.signals.script_of(ch: str) -> str | None`
  - `dhvani.signals.script_mix_entropy(text: str) -> float` in `[0.0, 1.0]`
  - `dhvani.signals.romanization_smell(text: str) -> float` in `[0.0, 1.0]`
  - `dhvani.signals.code_mixing_index(text: str) -> float` in `[0.0, 100.0]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signals.py
import pytest
from dhvani.signals import (
    script_of, script_mix_entropy, romanization_smell, code_mixing_index,
)


def test_script_of_identifies_blocks():
    assert script_of("अ") == "DEVANAGARI"
    assert script_of("ം") == "MALAYALAM"
    assert script_of("ಅ") == "KANNADA"
    assert script_of("a") == "LATIN"
    assert script_of(" ") is None
    assert script_of("1") is None


def test_monoscript_text_has_zero_entropy():
    assert script_mix_entropy("मैंने उस को कर दिया") == 0.0
    assert script_mix_entropy("i fixed the bug") == 0.0


def test_empty_text_has_zero_entropy():
    assert script_mix_entropy("") == 0.0
    assert script_mix_entropy("123 !!!") == 0.0


def test_balanced_two_script_mix_has_max_entropy():
    # Four Devanagari letters, four Latin letters.
    assert script_mix_entropy("अआइई abcd") == pytest.approx(1.0, abs=0.01)


def test_skewed_mix_has_intermediate_entropy():
    h = script_mix_entropy("मैंने उस को कर दिया bug")
    assert 0.0 < h < 1.0


def test_romanization_smell_flags_non_words():
    assert romanization_smell("the deployment is pending") == 0.0
    assert romanization_smell("thh dplymnt zz xqk") > 0.5


def test_code_mixing_index_is_zero_for_monolingual():
    assert code_mixing_index("i fixed the bug today") == 0.0


def test_code_mixing_index_rises_with_mixing():
    low = code_mixing_index("मैंने उस को कर दिया था वहाँ bug")
    high = code_mixing_index("मैंने bug को fix किया")
    assert 0.0 < low < high <= 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.signals'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/signals.py
"""Pure, cheap text signals used by the risk scorer.

script_mix_entropy targets the Chirp 3 rendering-ambiguity defect: a hypothesis
that flips script repeatedly inside one utterance is the direct fingerprint.
"""

import math
import re
import unicodedata
from collections import Counter

_INDIC_SCRIPTS = (
    "DEVANAGARI", "MALAYALAM", "KANNADA",
    "TAMIL", "TELUGU", "BENGALI", "GUJARATI", "ORIYA", "GURMUKHI",
)

_VOWELS = set("aeiou")


def script_of(ch: str) -> str | None:
    """Unicode script block of a letter, or None for non-letters."""
    if ch.isascii():
        return "LATIN" if ch.isalpha() else None
    name = unicodedata.name(ch, "")
    for script in _INDIC_SCRIPTS:
        if name.startswith(script):
            return script
    return None


def script_mix_entropy(text: str) -> float:
    """Normalized Shannon entropy over script blocks. 0.0 = single script."""
    counts = Counter(s for ch in text if (s := script_of(ch)) is not None)
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return h / math.log2(len(counts))


def romanization_smell(text: str) -> float:
    """Fraction of Latin tokens that look like neither English nor romanized Indic.

    Heuristic: a plausible word has at least one vowel and no run of four
    or more consonants.
    """
    tokens = [t for t in re.findall(r"[A-Za-z]+", text)]
    if not tokens:
        return 0.0
    bad = 0
    for tok in tokens:
        low = tok.lower()
        if not (_VOWELS & set(low)):
            bad += 1
            continue
        if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", low):
            bad += 1
    return bad / len(tokens)


def code_mixing_index(text: str) -> float:
    """Das & Gambaeck CMI: 100 * (1 - max_lang_tokens / non_neutral_tokens).

    Language is approximated by the dominant script of each token.
    """
    langs = []
    for tok in text.split():
        scripts = Counter(s for ch in tok if (s := script_of(ch)) is not None)
        if scripts:
            langs.append(scripts.most_common(1)[0][0])
    if not langs:
        return 0.0
    counts = Counter(langs)
    return 100.0 * (1.0 - counts.most_common(1)[0][1] / len(langs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_signals.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add dhvani/signals.py tests/test_signals.py
git commit -m "feat: script entropy, romanization smell, and CMI signals"
```

---

## Task 5: Deterministic risk scorer

**Files:**
- Create: `dhvani/scorer.py`
- Test: `tests/test_scorer.py`

**Interfaces:**
- Consumes: `dhvani.config.RISK_WEIGHTS`, `dhvani.signals.*`
- Produces:
  - `dhvani.scorer.Features` — frozen dataclass: `ctc_rnnt_disagreement: float`, `mean_neg_logprob: float`, `script_mix_entropy: float`, `romanization_smell: float`, `short_segment: float`
  - `dhvani.scorer.extract(text: str, decoder_signals: dict, duration_ms: int) -> Features`
  - `dhvani.scorer.risk(f: Features) -> float` in `[0.0, 1.0]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scorer.py
import pytest
from dhvani.scorer import Features, extract, risk

ZERO = Features(0.0, 0.0, 0.0, 0.0, 0.0)


def test_all_zero_features_give_zero_risk():
    assert risk(ZERO) == 0.0


def test_all_max_features_give_risk_one():
    assert risk(Features(1.0, 1.0, 1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_risk_is_bounded():
    assert 0.0 <= risk(Features(2.0, 2.0, 2.0, 2.0, 2.0)) <= 1.0
    assert 0.0 <= risk(Features(-1.0, -1.0, 0.0, 0.0, 0.0)) <= 1.0


def test_risk_is_monotonic_in_each_feature():
    base = risk(ZERO)
    for i in range(5):
        vals = [0.0] * 5
        vals[i] = 1.0
        assert risk(Features(*vals)) > base, f"feature {i} is not monotonic"


def test_risk_is_deterministic():
    f = Features(0.3, 0.4, 0.5, 0.2, 0.0)
    assert risk(f) == risk(f)


def test_extract_flags_short_segments():
    assert extract("hello", {}, duration_ms=800).short_segment == 1.0
    assert extract("hello", {}, duration_ms=3000).short_segment == 0.0


def test_extract_reads_script_entropy_from_text():
    f = extract("अआइई abcd", {}, duration_ms=3000)
    assert f.script_mix_entropy == pytest.approx(1.0, abs=0.01)


def test_extract_normalizes_missing_decoder_signals_to_zero():
    f = extract("hello world", {}, duration_ms=3000)
    assert f.ctc_rnnt_disagreement == 0.0
    assert f.mean_neg_logprob == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.scorer'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/scorer.py
"""Deterministic risk scoring. No model artifact — weights come from config."""

from dataclasses import asdict, dataclass

from dhvani.config import RISK_WEIGHTS
from dhvani.signals import romanization_smell, script_mix_entropy

SHORT_SEGMENT_MS = 1500


@dataclass(frozen=True)
class Features:
    ctc_rnnt_disagreement: float
    mean_neg_logprob: float
    script_mix_entropy: float
    romanization_smell: float
    short_segment: float


def extract(text: str, decoder_signals: dict, duration_ms: int) -> Features:
    """Build a feature vector. Used identically at fit time and at inference
    time — importing this one function everywhere is what prevents skew."""
    return Features(
        ctc_rnnt_disagreement=float(decoder_signals.get("ctc_rnnt_disagreement", 0.0)),
        mean_neg_logprob=float(decoder_signals.get("mean_neg_logprob", 0.0)),
        script_mix_entropy=script_mix_entropy(text),
        romanization_smell=romanization_smell(text),
        short_segment=1.0 if duration_ms < SHORT_SEGMENT_MS else 0.0,
    )


def risk(f: Features) -> float:
    """Weighted sum of clamped features, in [0, 1]."""
    total = sum(
        RISK_WEIGHTS[name] * min(max(value, 0.0), 1.0)
        for name, value in asdict(f).items()
    )
    return min(max(total, 0.0), 1.0)
```

Note: `RISK_WEIGHTS` sums to 1.0, so all-max features yield exactly 1.0.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_scorer.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add dhvani/scorer.py tests/test_scorer.py
git commit -m "feat: deterministic weighted risk scorer"
```

---

## Task 6: Pure greedy-knapsack router

**Files:**
- Create: `dhvani/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: nothing (pure)
- Produces:
  - `dhvani.router.Candidate` — frozen dataclass: `segment_id: str`, `tier: str`, `risk: float`, `cost_usd: float`, `delta: float`
  - `dhvani.router.plan(candidates: list[Candidate], budget_usd: float) -> list[Candidate]`
  - `dhvani.router.bucket_of(risk: float) -> str` (e.g. `"0.6-0.7"`)
  - `dhvani.router.delta_for(risk: float, tier: str, delta_table: dict) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
import pytest
from dhvani.router import Candidate, plan, bucket_of, delta_for


def c(sid, delta, cost=0.01, risk=0.5, tier="tier1"):
    return Candidate(segment_id=sid, tier=tier, risk=risk, cost_usd=cost, delta=delta)


def test_zero_budget_escalates_nothing():
    """Graceful degradation: B=0 still produces valid output."""
    assert plan([c("a", 10.0), c("b", 5.0)], budget_usd=0.0) == []


def test_large_budget_escalates_everything_positive():
    chosen = plan([c("a", 10.0), c("b", 5.0)], budget_usd=1000.0)
    assert {x.segment_id for x in chosen} == {"a", "b"}


def test_never_escalates_non_positive_delta():
    """Invariant I3: no negative-value escalation."""
    chosen = plan([c("a", 0.0), c("b", -3.0), c("c", 1.0)], budget_usd=1000.0)
    assert [x.segment_id for x in chosen] == ["c"]


def test_respects_budget():
    """Invariant I4."""
    cands = [c(str(i), delta=1.0, cost=0.01) for i in range(100)]
    chosen = plan(cands, budget_usd=0.05)
    assert sum(x.cost_usd for x in chosen) <= 0.05 + 1e-9
    assert len(chosen) == 5


def test_prefers_higher_delta_per_cost():
    cheap_good = c("cheap", delta=10.0, cost=0.01)   # ratio 1000
    dear_good = c("dear", delta=20.0, cost=1.00)     # ratio 20
    chosen = plan([dear_good, cheap_good], budget_usd=0.01)
    assert [x.segment_id for x in chosen] == ["cheap"]


def test_is_deterministic_under_ties():
    """Invariant I5: ties break by segment_id, never by input order."""
    a = [c("b", 1.0), c("a", 1.0)]
    b = [c("a", 1.0), c("b", 1.0)]
    assert plan(a, 0.01) == plan(b, 0.01)


def test_bucket_of_partitions_unit_interval():
    assert bucket_of(0.0) == "0.0-0.1"
    assert bucket_of(0.65) == "0.6-0.7"
    assert bucket_of(1.0) == "0.9-1.0"


def test_delta_for_reads_table_and_defaults_to_zero():
    table = {"tier1": {"0.6-0.7": 18.2}}
    assert delta_for(0.65, "tier1", table) == 18.2
    assert delta_for(0.05, "tier1", table) == 0.0
    assert delta_for(0.65, "tier2", table) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.router'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/router.py
"""Budget-constrained escalation policy.

Pure function, no I/O. This is the intellectual core of the system: given a
fixed spend per audio-hour, decide which segments deserve expensive treatment.
Greedy by delta/cost is the standard approximation to fractional knapsack.
"""

from dataclasses import dataclass

N_BUCKETS = 10


@dataclass(frozen=True)
class Candidate:
    segment_id: str
    tier: str
    risk: float
    cost_usd: float
    delta: float  # expected toWER points reduced, from the measured delta table


def bucket_of(risk: float) -> str:
    """Map a risk score to its decile bucket label."""
    idx = min(int(risk * N_BUCKETS), N_BUCKETS - 1)
    return f"{idx / N_BUCKETS:.1f}-{(idx + 1) / N_BUCKETS:.1f}"


def delta_for(risk: float, tier: str, delta_table: dict) -> float:
    """Measured expected improvement for this risk bucket and tier."""
    return float(delta_table.get(tier, {}).get(bucket_of(risk), 0.0))


def plan(candidates: list[Candidate], budget_usd: float) -> list[Candidate]:
    """Select escalations maximizing expected improvement within budget.

    Invariant I3: candidates with delta <= 0 are never selected.
    Invariant I4: total selected cost never exceeds budget_usd.
    Invariant I5: ties break on segment_id, so output is order-independent.
    """
    eligible = [c for c in candidates if c.delta > 0.0 and c.cost_usd > 0.0]
    eligible.sort(key=lambda c: (-(c.delta / c.cost_usd), c.segment_id, c.tier))

    chosen: list[Candidate] = []
    spent = 0.0
    for cand in eligible:
        if spent + cand.cost_usd <= budget_usd + 1e-9:
            chosen.append(cand)
            spent += cand.cost_usd
    return chosen
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_router.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add dhvani/router.py tests/test_router.py
git commit -m "feat: pure greedy-knapsack escalation router"
```

---

## Task 7: Backend protocol with record/replay and spend guard

**Files:**
- Create: `dhvani/backends/__init__.py`, `dhvani/backends/base.py`
- Test: `tests/test_backends.py`

**Interfaces:**
- Consumes: `dhvani.store.Store`, `dhvani.store.BudgetExceeded`
- Produces:
  - `dhvani.backends.base.Backend` — Protocol with `name: str`, `cost_per_call(segment) -> float`, `transcribe(segment) -> dict`
  - `dhvani.backends.base.FixtureMissing` (Exception)
  - `dhvani.backends.base.Mode` — `"record" | "replay" | "live"`
  - `dhvani.backends.base.Recorded(inner: Backend, mode: Mode, fixture_dir: str, store: Store | None)` — wrapper implementing `Backend`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends.py
import json
import numpy as np
import pytest

from dhvani.backends.base import Recorded, FixtureMissing
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def cost_per_call(self, segment):
        return 0.5

    def transcribe(self, segment):
        self.calls += 1
        return {"text": f"hello-{self.calls}", "signals": {}}


def _seg(sid="a" * 64):
    return Segment(segment_id=sid, t_start_ms=0, t_end_ms=3000,
                   pcm=np.zeros(10, dtype=np.int16))


def test_record_mode_writes_a_fixture(tmp_path):
    inner = FakeBackend()
    b = Recorded(inner, mode="record", fixture_dir=str(tmp_path), store=None)
    out = b.transcribe(_seg())
    assert out["text"] == "hello-1"
    written = tmp_path / "fake" / f"{'a' * 64}.json"
    assert json.loads(written.read_text())["text"] == "hello-1"


def test_replay_mode_reads_fixture_without_calling_inner(tmp_path):
    inner = FakeBackend()
    Recorded(inner, "record", str(tmp_path), None).transcribe(_seg())
    assert inner.calls == 1

    replayer = Recorded(inner, "replay", str(tmp_path), None)
    assert replayer.transcribe(_seg())["text"] == "hello-1"
    assert inner.calls == 1, "replay must not invoke the live backend"


def test_replay_hard_fails_on_missing_fixture(tmp_path):
    """Silent fallback to live is how a test run quietly spends money."""
    b = Recorded(FakeBackend(), "replay", str(tmp_path), None)
    with pytest.raises(FixtureMissing, match="no fixture"):
        b.transcribe(_seg())


def test_replay_costs_nothing(tmp_path):
    inner = FakeBackend()
    Recorded(inner, "record", str(tmp_path), None).transcribe(_seg())
    b = Recorded(inner, "replay", str(tmp_path), None)
    assert b.cost_per_call(_seg()) == 0.0


def test_live_mode_records_spend(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        b = Recorded(FakeBackend(), "live", str(tmp_path), store)
        b.transcribe(_seg())
        assert store.total_spend() == pytest.approx(0.5)


def test_live_mode_fails_closed_at_budget_ceiling(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        store.record_spend("fake", 19.8)
        b = Recorded(FakeBackend(), "live", str(tmp_path), store)
        with pytest.raises(BudgetExceeded):
            b.transcribe(_seg())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.backends'`

- [ ] **Step 3: Write the implementation**

```bash
mkdir -p dhvani/backends && touch dhvani/backends/__init__.py
```

```python
# dhvani/backends/base.py
"""Backend protocol plus the record/replay wrapper.

Two rules, both fail-closed:
  1. Replay never falls back to live on a cache miss.
  2. Live calls check the spend ledger before spending anything.
"""

import json
import os
from typing import Literal, Protocol, runtime_checkable

Mode = Literal["record", "replay", "live"]


class FixtureMissing(RuntimeError):
    """Replay mode was asked for a segment with no recorded fixture."""


@runtime_checkable
class Backend(Protocol):
    name: str

    def cost_per_call(self, segment) -> float: ...

    def transcribe(self, segment) -> dict: ...


class Recorded:
    """Wraps a Backend with record/replay and budget enforcement."""

    def __init__(self, inner: Backend, mode: Mode, fixture_dir: str, store=None):
        if mode not in ("record", "replay", "live"):
            raise ValueError(f"unknown mode: {mode}")
        self.inner = inner
        self.mode = mode
        self.fixture_dir = fixture_dir
        self.store = store
        self.name = inner.name

    def _path(self, segment) -> str:
        return os.path.join(self.fixture_dir, self.inner.name, f"{segment.segment_id}.json")

    def cost_per_call(self, segment) -> float:
        if self.mode == "replay":
            return 0.0
        return self.inner.cost_per_call(segment)

    def transcribe(self, segment) -> dict:
        if self.mode == "replay":
            path = self._path(segment)
            if not os.path.exists(path):
                raise FixtureMissing(
                    f"no fixture for {segment.segment_id} at {path}. "
                    f"Re-run in record mode; replay never falls back to live."
                )
            with open(path) as fh:
                return json.load(fh)

        cost = self.inner.cost_per_call(segment)
        if self.store is not None:
            self.store.check_budget(cost)

        result = self.inner.transcribe(segment)

        if self.store is not None:
            self.store.record_spend(self.inner.name, cost)

        if self.mode == "record":
            path = self._path(segment)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(result, fh, sort_keys=True, indent=2)

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backends.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add dhvani/backends/ tests/test_backends.py
git commit -m "feat: record/replay backend wrapper with fail-closed spend guard"
```

---

## Task 8: toWER evaluator

**Files:**
- Create: `dhvani/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `dhvani.signals.script_of`
- Produces:
  - `dhvani.evaluator.to_latin(text: str) -> str`
  - `dhvani.evaluator.to_wer(reference: str, hypothesis: str) -> float`
  - `dhvani.evaluator.plain_wer(reference: str, hypothesis: str) -> float`

**Known limitation to document in the module docstring:** `to_wer` reduces but does not
eliminate the penalty for Latin-vs-Indic renderings of English loanwords, because romanized
Malayalam of "deployment" does not match English spelling. Eliminating it entirely needs a
phonetic distance or a loanword lexicon. Out of scope for Phase 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluator.py
import pytest
from dhvani.evaluator import to_latin, to_wer, plain_wer


def test_identical_text_scores_zero():
    assert to_wer("hello world", "hello world") == 0.0


def test_plain_wer_counts_substitutions():
    assert plain_wer("a b c", "a b d") == pytest.approx(1 / 3)


def test_to_latin_is_idempotent_on_ascii():
    assert to_latin("hello world") == "hello world"


def test_same_word_in_two_indic_scripts_collapses():
    """Devanagari and Malayalam renderings of one word must converge."""
    deva, mala = "कम", "കമ"
    assert plain_wer(deva, mala) == 1.0
    assert to_wer(deva, mala) == 0.0


def test_benign_script_variance_is_penalized_less_than_plain_wer():
    """Spec 1.3: toWER must not treat a script flip like a semantic error."""
    ref = "deployment अभी pending है"
    benign = "ഡിപ്ലോയ്‌മെന്റ് अभी pending है"
    assert to_wer(ref, benign) <= plain_wer(ref, benign)


def test_semantic_corruption_is_still_penalized():
    """The counterpart: destroying meaning must still cost."""
    ref = "deployment अभी pending है"
    corrupt = "डिब्बा अभी pending है"
    assert to_wer(ref, corrupt) > 0.0


def test_to_wer_is_bounded_below_by_zero():
    assert to_wer("", "") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.evaluator'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/evaluator.py
"""Transliteration-optimized WER.

Plain WER treats a benign script flip and a semantic corruption as identical
single substitutions (spec 1.3). toWER maps every script to one writing system
before scoring, so benign variance stops inflating the error rate.

Prior art: Google Research, "Transliteration based approaches to improve
code-switched speech recognition performance". We adopt the metric; we do not
claim it.

Known limitation: this reduces but does not eliminate the penalty for
Latin-vs-Indic renderings of English loanwords, since romanized Malayalam of
"deployment" does not match English spelling. Removing that entirely requires a
phonetic distance or a loanword lexicon — out of scope for Phase 1.
"""

from collections import Counter

import jiwer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from dhvani.signals import script_of

_SCHEMES = {
    "DEVANAGARI": sanscript.DEVANAGARI,
    "MALAYALAM": sanscript.MALAYALAM,
    "KANNADA": sanscript.KANNADA,
    "TAMIL": sanscript.TAMIL,
    "TELUGU": sanscript.TELUGU,
    "BENGALI": sanscript.BENGALI,
    "GUJARATI": sanscript.GUJARATI,
    "ORIYA": sanscript.ORIYA,
    "GURMUKHI": sanscript.GURMUKHI,
}


def to_latin(text: str) -> str:
    """Map every Indic-script token to ITRANS, lowercase everything."""
    out = []
    for token in text.split():
        scripts = Counter(
            s for ch in token
            if (s := script_of(ch)) is not None and s != "LATIN"
        )
        if scripts:
            dominant = scripts.most_common(1)[0][0]
            scheme = _SCHEMES.get(dominant)
            if scheme is not None:
                token = transliterate(token, scheme, sanscript.ITRANS)
        out.append(token.lower())
    return " ".join(out)


def plain_wer(reference: str, hypothesis: str) -> float:
    """Standard WER. Empty reference and hypothesis scores 0."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    return float(jiwer.wer(reference, hypothesis))


def to_wer(reference: str, hypothesis: str) -> float:
    """WER computed after transliterating both sides to a common script."""
    return plain_wer(to_latin(reference), to_latin(hypothesis))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: PASS, 7 tests

If `test_same_word_in_two_indic_scripts_collapses` fails, the two source words are not
true transliteration equivalents — verify with
`python -c "from dhvani.evaluator import to_latin; print(to_latin('कम'), to_latin('कമ'))"`
and pick a pair that genuinely matches before adjusting the implementation.

- [ ] **Step 5: Commit**

```bash
git add dhvani/evaluator.py tests/test_evaluator.py
git commit -m "feat: transliteration-optimized WER evaluator"
```

---

## Task 9: Tier 0 IndicConformer adapter

**Files:**
- Create: `dhvani/backends/tier0_conformer.py`, `scripts/spike_conformer.py`
- Test: `tests/test_tier0.py`

**Interfaces:**
- Consumes: `dhvani.backends.base.Backend`
- Produces:
  - `dhvani.backends.tier0_conformer.Tier0Conformer(model=None, lang: str = "hi", model_id: str = MODEL_ID)` implementing `Backend`
  - `dhvani.backends.tier0_conformer.disagreement(ctc_text: str, rnnt_text: str) -> float`
  - `transcribe(segment) -> {"text": str, "signals": {"ctc_rnnt_disagreement": float, "mean_neg_logprob": float}}`

- [ ] **Step 1: Run the day-one spike (spec §14 risk)**

Determine whether the model exposes both CTC and RNNT heads. If it does not,
`ctc_rnnt_disagreement` is unavailable and `config.RISK_WEIGHTS` must be re-fit over the
remaining four features in Task 11. Record the answer in the commit message.

```python
# scripts/spike_conformer.py
"""Day-one spike: does IndicConformer expose both CTC and RNNT decoder heads?"""

import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
wav = torch.from_numpy(np.zeros(16000, dtype=np.float32)).unsqueeze(0)

for decoding in ("ctc", "rnnt"):
    try:
        out = model(wav, "hi", decoding)
        print(f"{decoding}: OK -> {out!r}")
    except Exception as exc:
        print(f"{decoding}: UNAVAILABLE -> {type(exc).__name__}: {exc}")
```

Run: `uv run python scripts/spike_conformer.py`

- [ ] **Step 2: Write the failing test**

The adapter is tested through the `Backend` contract using a stub, so the suite never needs
model weights or a network.

```python
# tests/test_tier0.py
import numpy as np
import pytest

from dhvani.backends.base import Backend
from dhvani.backends.tier0_conformer import Tier0Conformer, disagreement
from dhvani.segmenter import Segment


def _seg():
    return Segment(segment_id="a" * 64, t_start_ms=0, t_end_ms=3000,
                   pcm=np.zeros(16000, dtype=np.int16))


class StubModel:
    def __init__(self, ctc_text, rnnt_text):
        self.ctc_text, self.rnnt_text = ctc_text, rnnt_text

    def __call__(self, wav, lang, decoding):
        return self.ctc_text if decoding == "ctc" else self.rnnt_text


def test_satisfies_backend_protocol():
    assert isinstance(Tier0Conformer(model=StubModel("a", "a"), lang="hi"), Backend)


def test_tier0_is_free():
    assert Tier0Conformer(model=StubModel("a", "a"), lang="hi").cost_per_call(_seg()) == 0.0


def test_returns_rnnt_text_as_primary_hypothesis():
    b = Tier0Conformer(model=StubModel("ctc out", "rnnt out"), lang="hi")
    assert b.transcribe(_seg())["text"] == "rnnt out"


def test_agreeing_heads_give_zero_disagreement():
    b = Tier0Conformer(model=StubModel("same words here", "same words here"), lang="hi")
    assert b.transcribe(_seg())["signals"]["ctc_rnnt_disagreement"] == 0.0


def test_disagreeing_heads_give_positive_disagreement():
    b = Tier0Conformer(model=StubModel("alpha beta gamma", "alpha beta delta"), lang="hi")
    assert b.transcribe(_seg())["signals"]["ctc_rnnt_disagreement"] > 0.0


def test_disagreement_is_bounded():
    assert disagreement("a b c", "x y z") <= 1.0
    assert disagreement("", "") == 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_tier0.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.backends.tier0_conformer'`

- [ ] **Step 4: Write the implementation**

```python
# dhvani/backends/tier0_conformer.py
"""Tier 0: local IndicConformer. Runs on 100% of segments at zero marginal cost.

The model is hybrid CTC-RNNT: two decoders over one shared encoder. Disagreement
between the heads is ensemble uncertainty for free, and is expected to be the
strongest single risk signal.
"""

import numpy as np

from dhvani.evaluator import plain_wer

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


def disagreement(ctc_text: str, rnnt_text: str) -> float:
    """Normalized edit distance between the two decoder heads, in [0, 1]."""
    if not ctc_text.strip() and not rnnt_text.strip():
        return 0.0
    return min(plain_wer(rnnt_text, ctc_text), 1.0)


def _load(model_id: str):
    import torch  # noqa: F401  (imported for side effects / availability check)
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, trust_remote_code=True)


class Tier0Conformer:
    name = "tier0"

    def __init__(self, model=None, lang: str = "hi", model_id: str = MODEL_ID):
        self._model = model if model is not None else _load(model_id)
        self.lang = lang

    def cost_per_call(self, segment) -> float:
        return 0.0  # local inference

    def transcribe(self, segment) -> dict:
        wav = self._to_float(segment.pcm)
        ctc_text = str(self._model(wav, self.lang, "ctc"))
        rnnt_text = str(self._model(wav, self.lang, "rnnt"))
        return {
            "text": rnnt_text,
            "signals": {
                "ctc_rnnt_disagreement": disagreement(ctc_text, rnnt_text),
                "mean_neg_logprob": 0.0,  # not exposed by this model; see spec §14
            },
        }

    @staticmethod
    def _to_float(pcm: np.ndarray):
        import torch

        return torch.from_numpy((pcm.astype(np.float32) / 32768.0)).unsqueeze(0)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_tier0.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Commit**

```bash
git add dhvani/backends/tier0_conformer.py scripts/spike_conformer.py tests/test_tier0.py
git commit -m "feat: Tier 0 IndicConformer adapter with CTC-RNNT disagreement signal

Spike result: replace this line with the actual scripts/spike_conformer.py\noutput, i.e. whether ctc and rnnt decoding both returned text."
```

---

## Task 10: Pipeline orchestration and CLI

**Files:**
- Create: `dhvani/pipeline.py`, `dhvani/cli.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `dhvani.pipeline.TrackEntry` — frozen dataclass: `segment_id`, `t_start_ms`, `t_end_ms`, `text`, `risk`, `band`
  - `dhvani.pipeline.band_of(risk: float) -> str` — `"ship" | "marked" | "review"`
  - `dhvani.pipeline.run(pcm, source_id, tier0, store, delta_table, budget_usd) -> list[TrackEntry]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import numpy as np
import pytest

from dhvani.config import SAMPLE_RATE
from dhvani.pipeline import band_of, run
from dhvani.store import Store


class StubTier0:
    name = "tier0"

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        return {"text": "नमस्ते world", "signals": {"ctc_rnnt_disagreement": 0.1}}


def _audio(seconds=6.0):
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 200 * t)
    return (x * 32767).round().astype(np.int16)


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_band_of_partitions_by_threshold():
    assert band_of(0.10) == "ship"
    assert band_of(0.45) == "marked"
    assert band_of(0.90) == "review"


def test_run_produces_one_entry_per_segment(store):
    entries = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0)
    assert len(entries) >= 1
    assert all(e.text for e in entries)


def test_every_entry_has_a_band(store):
    entries = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0)
    assert all(e.band in {"ship", "marked", "review"} for e in entries)


def test_run_is_deterministic(store, tmp_path):
    """Invariant I5."""
    a = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0)
    with Store(str(tmp_path / "u.db")) as store2:
        b = run(_audio(), "vid1", StubTier0(), store2, {}, budget_usd=0.0)
    assert [(e.segment_id, e.text, e.risk, e.band) for e in a] == \
           [(e.segment_id, e.text, e.risk, e.band) for e in b]


def test_second_run_hits_cache_and_does_not_recall_backend(store):
    class CountingTier0(StubTier0):
        calls = 0

        def transcribe(self, segment):
            CountingTier0.calls += 1
            return super().transcribe(segment)

    backend = CountingTier0()
    run(_audio(), "vid1", backend, store, {}, budget_usd=0.0)
    first = CountingTier0.calls
    run(_audio(), "vid1", backend, store, {}, budget_usd=0.0)
    assert CountingTier0.calls == first, "cached segments must not be re-transcribed"


def test_zero_budget_still_produces_a_full_track(store):
    """Graceful degradation."""
    entries = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0)
    assert len(entries) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/pipeline.py
"""Synchronous orchestration: segment, transcribe, score, route, band."""

from dataclasses import dataclass

import numpy as np

from dhvani.config import TAU_FLAG, TAU_SHIP
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import segment as split


@dataclass(frozen=True)
class TrackEntry:
    segment_id: str
    t_start_ms: int
    t_end_ms: int
    text: str
    risk: float
    band: str


def band_of(risk: float) -> str:
    """Spec §6.2 output bands. Nothing below tau_flag ships silently."""
    if risk < TAU_SHIP:
        return "ship"
    if risk < TAU_FLAG:
        return "marked"
    return "review"


def run(pcm: np.ndarray, source_id: str, tier0, store,
        delta_table: dict, budget_usd: float) -> list[TrackEntry]:
    """Produce a caption track. Cached segments are never re-transcribed.

    delta_table and budget_usd are accepted for interface stability but are
    unused in Phase 1: escalation is computed offline by report.frontier().
    Phase 2 wires them into asynchronous Tier 1 submission.
    """
    segments = split(pcm)
    entries: list[TrackEntry] = []

    for seg in segments:
        store.put_segment(seg.segment_id, source_id, seg.t_start_ms, seg.t_end_ms)

        cached = store.get_hypothesis(seg.segment_id, "tier0")
        if cached is None:
            result = tier0.transcribe(seg)
            store.put_hypothesis(
                seg.segment_id, "tier0", result["text"],
                result["signals"], tier0.cost_per_call(seg),
            )
        else:
            result = {"text": cached["text"], "signals": cached["signals"]}

        duration = seg.t_end_ms - seg.t_start_ms
        features = extract(result["text"], result["signals"], duration)
        r = compute_risk(features)

        entries.append(TrackEntry(
            segment_id=seg.segment_id,
            t_start_ms=seg.t_start_ms,
            t_end_ms=seg.t_end_ms,
            text=result["text"],
            risk=r,
            band=band_of(r),
        ))

    return entries
```

```python
# dhvani/cli.py
"""Entrypoint: dhvani transcribe <audio.wav>"""

import argparse
import json
import os

import soundfile as sf

from dhvani.audio import normalize
from dhvani.backends.base import Recorded
from dhvani.backends.tier0_conformer import Tier0Conformer
from dhvani.pipeline import run
from dhvani.store import Store


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dhvani")
    ap.add_argument("audio")
    ap.add_argument("--db", default="dhvani.db")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--mode", default=os.environ.get("DHVANI_MODE", "replay"))
    ap.add_argument("--fixtures", default="fixtures")
    ap.add_argument("--delta-table", default="delta_table.json")
    args = ap.parse_args(argv)

    samples, rate = sf.read(args.audio)
    pcm = normalize(samples, rate)

    delta_table = {}
    if os.path.exists(args.delta_table):
        with open(args.delta_table) as fh:
            delta_table = json.load(fh)

    with Store(args.db) as store:
        tier0 = Recorded(
            Tier0Conformer(lang=args.lang), args.mode, args.fixtures, store
        )
        entries = run(pcm, os.path.basename(args.audio), tier0, store,
                      delta_table, args.budget)

    print(json.dumps([e.__dict__ for e in entries], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS, 67 tests

- [ ] **Step 6: Commit**

```bash
git add dhvani/pipeline.py dhvani/cli.py tests/test_pipeline.py
git commit -m "feat: synchronous pipeline orchestration and CLI"
```

---

## Task 11: Delta table builder and cost/quality frontier report

**Files:**
- Create: `dhvani/delta_table.py`, `dhvani/report.py`, `Makefile`
- Test: `tests/test_delta_table.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `dhvani.router.bucket_of`, `dhvani.evaluator.to_wer`
- Note: `build()` takes rows containing `tier1_text`, which come from Task 12.
  Task 11's tests use synthetic rows, so it is independently testable and may be
  built first.
- Produces:
  - `dhvani.delta_table.build(rows: list[dict]) -> dict` where each row has keys
    `risk`, `reference`, `tier0_text`, `tier1_text`
  - `dhvani.report.frontier(entries, delta_table, budgets) -> list[dict]` with keys
    `budget_usd`, `escalated`, `cost_usd`, `mean_risk`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delta_table.py
import pytest
from dhvani.delta_table import build


def test_empty_input_gives_empty_table():
    assert build([]) == {"tier1": {}}


def test_measures_improvement_per_bucket():
    rows = [{
        "risk": 0.65,
        "reference": "alpha beta gamma delta",
        "tier0_text": "alpha beta gamma WRONG",
        "tier1_text": "alpha beta gamma delta",
    }]
    table = build(rows)
    # tier0 toWER = 0.25 (25 points), tier1 = 0.0 -> delta = 25.0
    assert table["tier1"]["0.6-0.7"] == pytest.approx(25.0)


def test_negative_delta_is_preserved_not_clamped():
    """Tier 1 genuinely loses on some segments; the router filters, not the table."""
    rows = [{
        "risk": 0.15,
        "reference": "alpha beta",
        "tier0_text": "alpha beta",
        "tier1_text": "alpha WRONG",
    }]
    assert build(rows)["tier1"]["0.1-0.2"] < 0.0


def test_averages_within_a_bucket():
    rows = [
        {"risk": 0.65, "reference": "a b c d", "tier0_text": "a b c X", "tier1_text": "a b c d"},
        {"risk": 0.66, "reference": "a b c d", "tier0_text": "a b c d", "tier1_text": "a b c d"},
    ]
    assert build(rows)["tier1"]["0.6-0.7"] == pytest.approx(12.5)
```

```python
# tests/test_report.py
import pytest
from dhvani.pipeline import TrackEntry
from dhvani.report import frontier


def _entry(sid, risk):
    return TrackEntry(sid, 0, 3000, "text", risk, "marked")


def test_frontier_is_monotonic_in_budget():
    entries = [_entry(f"s{i}", 0.65) for i in range(10)]
    table = {"tier1": {"0.6-0.7": 18.0}}
    rows = frontier(entries, table, budgets=[0.0, 0.001, 0.01, 1.0])
    counts = [r["escalated"] for r in rows]
    assert counts == sorted(counts)


def test_zero_budget_escalates_nothing():
    entries = [_entry("s0", 0.65)]
    rows = frontier(entries, {"tier1": {"0.6-0.7": 18.0}}, budgets=[0.0])
    assert rows[0]["escalated"] == 0
    assert rows[0]["cost_usd"] == 0.0


def test_cost_never_exceeds_budget():
    entries = [_entry(f"s{i}", 0.65) for i in range(50)]
    table = {"tier1": {"0.6-0.7": 18.0}}
    for row in frontier(entries, table, budgets=[0.0, 0.002, 0.02]):
        assert row["cost_usd"] <= row["budget_usd"] + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_delta_table.py tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.delta_table'`

- [ ] **Step 3: Write the implementation**

```python
# dhvani/delta_table.py
"""Measures expected toWER improvement per risk bucket, once, offline.

This is measurement, not training: no parameters are fitted, no model is
produced. The output is a lookup table committed to the repo.
"""

from collections import defaultdict

from dhvani.evaluator import to_wer
from dhvani.router import bucket_of


def build(rows: list[dict]) -> dict:
    """rows: {risk, reference, tier0_text, tier1_text} -> {tier: {bucket: delta}}

    Delta is in toWER *points* (percentage points), so a 0.25 -> 0.0 toWER
    improvement is 25.0. Negative deltas are preserved: Tier 1 genuinely loses
    on some segments, and the router (invariant I3) is what filters them out.
    """
    buckets = defaultdict(list)
    for row in rows:
        before = to_wer(row["reference"], row["tier0_text"])
        after = to_wer(row["reference"], row["tier1_text"])
        buckets[bucket_of(row["risk"])].append((before - after) * 100.0)

    return {"tier1": {b: sum(v) / len(v) for b, v in sorted(buckets.items())}}
```

```python
# dhvani/report.py
"""Cost/quality frontier: the headline artifact."""

from dhvani.router import Candidate, delta_for, plan

TIER1_USD_PER_MIN = 0.003


def frontier(entries, delta_table: dict, budgets: list[float]) -> list[dict]:
    """Escalation behaviour across a sweep of budgets."""
    candidates = [
        Candidate(
            segment_id=e.segment_id,
            tier="tier1",
            risk=e.risk,
            cost_usd=TIER1_USD_PER_MIN * (e.t_end_ms - e.t_start_ms) / 60000.0,
            delta=delta_for(e.risk, "tier1", delta_table),
        )
        for e in entries
    ]

    rows = []
    for budget in sorted(budgets):
        chosen = plan(candidates, budget)
        rows.append({
            "budget_usd": budget,
            "escalated": len(chosen),
            "cost_usd": sum(c.cost_usd for c in chosen),
            "mean_risk": (
                sum(c.risk for c in chosen) / len(chosen) if chosen else 0.0
            ),
        })
    return rows


def render_markdown(rows: list[dict]) -> str:
    out = ["| budget ($) | escalated | spent ($) | mean risk |",
           "|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['budget_usd']:.4f} | {r['escalated']} | "
            f"{r['cost_usd']:.4f} | {r['mean_risk']:.3f} |"
        )
    return "\n".join(out)
```

```python
# dhvani/report_cli.py
"""make bench entrypoint: renders the cost/quality frontier to stdout."""

import json
import os
import sys

from dhvani.report import frontier, render_markdown

BUDGETS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 1.0]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    track_path = argv[0] if argv else "results/track.json"
    table_path = argv[1] if len(argv) > 1 else "delta_table.json"

    if not os.path.exists(track_path):
        print(f"missing {track_path}; run `dhvani transcribe` first", file=sys.stderr)
        return 1

    from dhvani.pipeline import TrackEntry

    with open(track_path, encoding="utf-8") as fh:
        entries = [TrackEntry(**row) for row in json.load(fh)]

    delta_table = {}
    if os.path.exists(table_path):
        with open(table_path, encoding="utf-8") as fh:
            delta_table = json.load(fh)

    print("# Dhvani cost/quality frontier\n")
    print(render_markdown(frontier(entries, delta_table, BUDGETS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```makefile
# Makefile
.PHONY: test bench

test:
	uv run pytest -v

bench:
	mkdir -p results
	DHVANI_MODE=replay uv run python -m dhvani.report_cli \
	  results/track.json delta_table.json > results/report.md
	@echo "wrote results/report.md"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_delta_table.py tests/test_report.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS, 74 tests

- [ ] **Step 6: Commit**

```bash
mkdir -p results && touch results/.gitkeep
git add dhvani/delta_table.py dhvani/report.py Makefile results/.gitkeep \
        tests/test_delta_table.py tests/test_report.py
git commit -m "feat: delta table builder and cost/quality frontier report"
```

---

## Task 12: Synchronous Tier 1 Chirp adapter

Task 11 consumes `tier1_text`, which nothing yet produces. This task closes that gap with a
**synchronous** Chirp call — enough to populate `delta_table.json`. Phase 2 replaces it with
the asynchronous dynamic-batch path (spec §7).

**Files:**
- Create: `dhvani/backends/tier1_chirp.py`, `scripts/spike_chirp.py`
- Test: `tests/test_tier1.py`

**Interfaces:**
- Consumes: `dhvani.backends.base.Backend`
- Produces:
  - `dhvani.backends.tier1_chirp.Tier1Chirp(client=None, lang: str = "hi-IN", recognizer: str = "")` implementing `Backend`
  - `dhvani.backends.tier1_chirp.USD_PER_MIN_DYNAMIC_BATCH: float = 0.003`
  - `dhvani.backends.tier1_chirp.USD_PER_MIN_STANDARD: float = 0.016`

- [ ] **Step 1: Run the day-one spike (spec §14 risk)**

Confirm Chirp 3 supports the dynamic-batch processing strategy. If it does not, set
`USD_PER_MIN_DYNAMIC_BATCH = USD_PER_MIN_STANDARD` and note that the benchmark budget rises
from roughly $2.70 to roughly $14.40 — still inside the $20 ceiling, but with no margin.

```python
# scripts/spike_chirp.py
"""Day-one spike: does Chirp 3 accept the dynamic-batch processing strategy?"""

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cs

PROJECT = "REPLACE_WITH_YOUR_GCP_PROJECT_ID"
client = SpeechClient()

req = cs.BatchRecognizeRequest(
    recognizer=f"projects/{PROJECT}/locations/global/recognizers/_",
    config=cs.RecognitionConfig(
        auto_decoding_config=cs.AutoDetectDecodingConfig(),
        model="chirp_3",
        language_codes=["hi-IN"],
    ),
    processing_strategy=cs.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING,
    files=[cs.BatchRecognizeFileMetadata(uri="gs://YOUR_BUCKET/sample.wav")],
    recognition_output_config=cs.RecognitionOutputConfig(
        inline_response_config=cs.InlineOutputConfig()
    ),
)

try:
    op = client.batch_recognize(request=req)
    print("DYNAMIC_BATCHING: accepted ->", op.operation.name)
except Exception as exc:
    print(f"DYNAMIC_BATCHING: REJECTED -> {type(exc).__name__}: {exc}")
```

Run: `uv run python scripts/spike_chirp.py`

- [ ] **Step 2: Write the failing test**

Tested through the `Backend` contract with a stub client, so the suite needs no credentials.

```python
# tests/test_tier1.py
import numpy as np
import pytest

from dhvani.backends.base import Backend
from dhvani.backends.tier1_chirp import (
    Tier1Chirp, USD_PER_MIN_DYNAMIC_BATCH, USD_PER_MIN_STANDARD,
)
from dhvani.segmenter import Segment


def _seg(ms=60000):
    return Segment(segment_id="b" * 64, t_start_ms=0, t_end_ms=ms,
                   pcm=np.zeros(16000, dtype=np.int16))


class StubClient:
    def __init__(self, text="chirp output"):
        self.text = text
        self.calls = 0

    def recognize_pcm(self, pcm, lang):
        self.calls += 1
        return self.text


def test_satisfies_backend_protocol():
    assert isinstance(Tier1Chirp(client=StubClient()), Backend)


def test_one_minute_costs_the_dynamic_batch_rate():
    assert Tier1Chirp(client=StubClient()).cost_per_call(_seg(60000)) == \
        pytest.approx(USD_PER_MIN_DYNAMIC_BATCH)


def test_cost_scales_with_duration():
    b = Tier1Chirp(client=StubClient())
    assert b.cost_per_call(_seg(30000)) == pytest.approx(USD_PER_MIN_DYNAMIC_BATCH / 2)


def test_standard_rate_is_the_documented_fallback():
    assert USD_PER_MIN_STANDARD == pytest.approx(0.016)
    assert USD_PER_MIN_STANDARD > USD_PER_MIN_DYNAMIC_BATCH


def test_transcribe_returns_text_and_empty_signals():
    out = Tier1Chirp(client=StubClient("नमस्ते")).transcribe(_seg())
    assert out["text"] == "नमस्ते"
    assert out["signals"] == {}


def test_transcribe_calls_the_client_once():
    client = StubClient()
    Tier1Chirp(client=client).transcribe(_seg())
    assert client.calls == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_tier1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dhvani.backends.tier1_chirp'`

- [ ] **Step 4: Write the implementation**

```python
# dhvani/backends/tier1_chirp.py
"""Tier 1: Google Cloud Speech-to-Text v2 (Chirp 3), synchronous.

Phase 1 calls Chirp synchronously purely to populate delta_table.json. Phase 2
replaces this with asynchronous dynamic-batch submission plus reconciliation
(spec §7), which is where the interesting distributed-systems work lives.

Rates verified 2026-08-21. If the Task 12 spike shows dynamic batching is
unsupported for Chirp 3, set USD_PER_MIN_DYNAMIC_BATCH = USD_PER_MIN_STANDARD.
"""

USD_PER_MIN_DYNAMIC_BATCH = 0.003
USD_PER_MIN_STANDARD = 0.016


class Tier1Chirp:
    name = "tier1"

    def __init__(self, client=None, lang: str = "hi-IN", recognizer: str = ""):
        self._client = client if client is not None else _default_client()
        self.lang = lang
        self.recognizer = recognizer

    def cost_per_call(self, segment) -> float:
        minutes = (segment.t_end_ms - segment.t_start_ms) / 60000.0
        return USD_PER_MIN_DYNAMIC_BATCH * minutes

    def transcribe(self, segment) -> dict:
        text = self._client.recognize_pcm(segment.pcm, self.lang)
        return {"text": str(text), "signals": {}}


def _default_client():
    """Thin adapter over SpeechClient exposing recognize_pcm(pcm, lang) -> str."""
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech as cs

    class _Client:
        def __init__(self):
            self._inner = SpeechClient()

        def recognize_pcm(self, pcm, lang: str) -> str:
            req = cs.RecognizeRequest(
                recognizer=f"projects/{_project()}/locations/global/recognizers/_",
                config=cs.RecognitionConfig(
                    explicit_decoding_config=cs.ExplicitDecodingConfig(
                        encoding=cs.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=16000,
                        audio_channel_count=1,
                    ),
                    model="chirp_3",
                    language_codes=[lang],
                ),
                content=pcm.tobytes(),
            )
            resp = self._inner.recognize(request=req)
            return " ".join(
                r.alternatives[0].transcript for r in resp.results if r.alternatives
            )

    return _Client()


def _project() -> str:
    import os

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError("set GOOGLE_CLOUD_PROJECT before using the live Chirp backend")
    return project
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_tier1.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS, 80 tests

- [ ] **Step 7: Commit**

```bash
git add dhvani/backends/tier1_chirp.py scripts/spike_chirp.py tests/test_tier1.py
git commit -m "feat: synchronous Chirp Tier 1 adapter for delta table construction

Spike result: replace this line with the actual scripts/spike_chirp.py output,
i.e. whether DYNAMIC_BATCHING was accepted for chirp_3."
```

---

## Phase 1 Exit Criteria

- [ ] `uv run pytest` passes with no network access and no cloud credentials
- [ ] `dhvani transcribe sample.wav` produces a banded caption track
- [ ] Re-running the same audio makes zero backend calls (cache hit)
- [ ] `delta_table.json` is committed, built from a speaker-disjoint calibration split
- [ ] `make bench` writes a cost/quality frontier to `results/report.md`
- [ ] Total external spend recorded in the `spend` ledger is under USD 5
- [ ] Both spike results are recorded in git history: CTC/RNNT head availability
      (Task 9) and Chirp dynamic-batch support (Task 12)

## Deferred to Phase 2

Async Tier 1 submission and reconciliation (spec §7), `jobs` and `tracks` tables,
`ChaosBackend` and invariants I1/I6, Tier 2 Gemini repair with quota-aware rate
limiting, throughput and p50/p99 latency instrumentation (spec §9.1), and the
production scaling sketch (spec §12).
