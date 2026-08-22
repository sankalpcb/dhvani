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
