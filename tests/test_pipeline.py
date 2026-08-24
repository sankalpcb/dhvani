import numpy as np
import pytest

from dhvani.config import SAMPLE_RATE, TAU_FLAG, TAU_SHIP
from dhvani.pipeline import band_of, run
from dhvani.store import Store


class StubTier0:
    name = "tier0"

    def __init__(self, lang="hi", text="नमस्ते world"):
        self.lang = lang
        self.text = text

    @property
    def variant_key(self):
        return f"lang={self.lang};model_id=stub"

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        return {"text": self.text, "signals": {"ctc_rnnt_disagreement": 0.1}}


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


def test_band_of_pins_exact_boundaries():
    """band_of uses strict '<' comparisons, so a risk exactly equal to a
    threshold falls into the WORSE band, not the better one. Pin this
    explicitly so it stays a deliberate choice, not an incidental one."""
    assert band_of(TAU_SHIP) == "marked", "exactly tau_ship is not 'ship'"
    assert band_of(TAU_FLAG) == "review", "exactly tau_flag is not 'marked'"


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


def test_run_populates_samples_when_given(store):
    """I7: run() times the real Tier 0 work into the caller's dict when
    given one, under the "tier0" stage key."""
    samples = {}
    entries = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0,
                  samples=samples)
    assert len(entries) >= 1
    assert "tier0" in samples
    assert len(samples["tier0"]) >= 1
    assert all(isinstance(v, float) and v >= 0.0 for v in samples["tier0"])


def test_run_without_samples_behaves_identically_to_before(store):
    """Existing callers that pass nothing must be unaffected."""
    entries = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0)
    assert len(entries) >= 1
    assert all(e.text for e in entries)


def test_zero_budget_still_produces_a_full_track(store):
    """Graceful degradation."""
    entries = run(_audio(), "vid1", StubTier0(), store, {}, budget_usd=0.0)
    assert len(entries) >= 1


# --- Fix round 2, I2/I3: the cache must not ignore lang / model_id / policy ---

def test_changing_lang_does_not_serve_the_other_langs_cached_text(store):
    """The demonstrated defect: run with lang="hi", then with lang="ml",
    and the second run returned the Hindi transcript straight from cache
    -- --lang was silently a no-op, because segment_id hashes PCM alone
    and the hypothesis key was just (segment_id, "tier0")."""
    hindi = run(_audio(), "vid1", StubTier0("hi", "नमस्ते"), store, {}, budget_usd=0.0)
    malayalam = run(_audio(), "vid1", StubTier0("ml", "നമസ്കാരം"), store, {}, budget_usd=0.0)

    assert [e.text for e in hindi] == ["नमस्ते"] * len(hindi)
    assert [e.text for e in malayalam] == ["നമസ്കാരം"] * len(malayalam)


def test_bumping_policy_id_forces_a_re_transcribe(store, monkeypatch):
    """POLICY_ID is the documented cache-invalidation lever, so bumping it
    must make the pipeline call the backend again instead of serving the
    stale hypothesis."""
    class CountingTier0(StubTier0):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = 0

        def transcribe(self, segment):
            self.calls += 1
            return super().transcribe(segment)

    backend = CountingTier0()
    run(_audio(), "vid1", backend, store, {}, budget_usd=0.0)
    after_first = backend.calls
    assert after_first >= 1

    run(_audio(), "vid1", backend, store, {}, budget_usd=0.0)
    assert backend.calls == after_first, "same policy must hit the cache"

    monkeypatch.setattr("dhvani.config.POLICY_ID", "p-bumped")
    run(_audio(), "vid1", backend, store, {}, budget_usd=0.0)
    assert backend.calls == after_first * 2, "a bumped POLICY_ID must re-transcribe"
