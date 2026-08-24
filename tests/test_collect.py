import numpy as np
import pytest

from dhvani.calibrate import collect
from dhvani.corpus import FakeCorpus
from dhvani.store import Store


class StubTier0:
    name = "tier0"
    variant_key = "tier0|hi|m"

    def __init__(self):
        self.calls = 0

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        self.calls += 1
        return {"text": "नमस्ते world", "signals": {"ctc_rnnt_disagreement": 0.4}}


def _corpus(n=3, lang="hi-IN"):
    rng = np.random.default_rng(0)
    return FakeCorpus([
        (0.3 * rng.standard_normal(32000), 16000, f"ref-{i}", lang, f"spk{i}", f"d{i}")
        for i in range(n)
    ])


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_collect_returns_one_scored_item_per_utterance(store):
    out = collect(_corpus(3), StubTier0(), store, ["hi-IN"], per_lang=3)
    assert len(out) == 3
    assert all(0.0 <= s["risk"] <= 1.0 for s in out)


def test_collect_persists_reference_and_hypothesis(store):
    out = collect(_corpus(1), StubTier0(), store, ["hi-IN"], per_lang=1)
    sid = out[0]["segment_id"]
    assert store.get_reference(sid)["reference"] == "ref-0"
    assert store.get_hypothesis(sid, "tier0", "tier0|hi|m")["text"] == "नमस्ते world"


def test_collect_is_resumable_and_does_not_retranscribe(store):
    """The property that makes a multi-hour run survivable."""
    tier0 = StubTier0()
    collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    first = tier0.calls
    collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    assert tier0.calls == first, "cached segments must not be re-transcribed"


def test_collect_scores_identically_on_a_cached_rerun(store):
    tier0 = StubTier0()
    a = collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    b = collect(_corpus(3), tier0, store, ["hi-IN"], per_lang=3)
    assert a == b


def test_collect_spans_requested_languages(store):
    rng = np.random.default_rng(1)
    corpus = FakeCorpus([
        (0.3 * rng.standard_normal(32000), 16000, "h", "hi-IN", "s1", "d1"),
        (0.3 * rng.standard_normal(32000), 16000, "k", "kn-IN", "s2", "d2"),
    ])
    out = collect(corpus, StubTier0(), store, ["hi-IN", "kn-IN"], per_lang=1)
    assert {s["lang"] for s in out} == {"hi-IN", "kn-IN"}


def test_collect_carries_duration_for_later_pricing(store):
    out = collect(_corpus(1), StubTier0(), store, ["hi-IN"], per_lang=1)
    assert out[0]["duration_ms"] == 2000
