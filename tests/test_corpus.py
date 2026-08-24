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
