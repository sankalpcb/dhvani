import numpy as np
import pytest
from dhvani.corpus import CorpusItem, FakeCorpus, disjoint_by, _extract_item_from_row, SCHEMA_PROBE_ROWS


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


def test_extract_item_from_row_with_wrong_field_names():
    """Row with wrong field names returns None."""
    row_with_wrong_fields = {
        "sound": {"array": np.zeros(1000), "sampling_rate": 16000},
        "caption": "hello world",
    }
    result = _extract_item_from_row(row_with_wrong_fields, "hi-IN")
    assert result is None


def test_extract_item_from_row_with_correct_fields():
    """Row with correct field names produces a CorpusItem."""
    row = {
        "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
        "text": "hello world",
        "speaker_id": "spk1",
        "district": "dist1",
    }
    result = _extract_item_from_row(row, "hi-IN")
    assert result is not None
    assert isinstance(result, CorpusItem)
    assert result.reference == "hello world"
    assert result.lang == "hi-IN"
    assert result.speaker_id == "spk1"
    assert result.district == "dist1"


def test_extract_item_handles_missing_optional_fields():
    """Row with missing optional fields still produces an item."""
    row = {
        "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
        "text": "hello",
    }
    result = _extract_item_from_row(row, "hi-IN")
    assert result is not None
    assert result.speaker_id is None
    assert result.district is None


def test_indic_voices_corpus_schema_mismatch_raises():
    """IndicVoicesCorpus raises with diagnostic message after probing empty dataset."""
    from dhvani.corpus import IndicVoicesCorpus
    import sys
    import types

    # Mock a dataset with wrong field names that yields nothing
    class MockDataset:
        def __iter__(self):
            for i in range(SCHEMA_PROBE_ROWS + 10):
                yield {
                    "wrong_audio_field": {"array": np.zeros(1000), "sampling_rate": 16000},
                    "wrong_text_field": "hello",
                }

    corpus = IndicVoicesCorpus(dataset_id="test/dataset")

    # Inject mock datasets module
    mock_datasets = types.ModuleType("datasets")
    mock_datasets.load_dataset = lambda *args, **kwargs: MockDataset()
    sys.modules["datasets"] = mock_datasets

    try:
        with pytest.raises(ValueError) as exc_info:
            list(corpus.stream("hi-IN", limit=1000))

        assert "schema mismatch" in str(exc_info.value).lower()
        assert "expected fields" in str(exc_info.value).lower()
        assert "actual row keys" in str(exc_info.value).lower()
    finally:
        # Cleanup
        if "datasets" in sys.modules:
            del sys.modules["datasets"]


def test_indic_voices_corpus_small_dataset_no_error():
    """A small dataset that produces items before SCHEMA_PROBE_ROWS does not raise."""
    from dhvani.corpus import IndicVoicesCorpus

    class MockDataset:
        def __iter__(self):
            for i in range(10):  # Less than SCHEMA_PROBE_ROWS
                yield {
                    "audio": {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                    "text": f"hello {i}",
                    "speaker_id": f"spk{i}",
                    "district": f"dist{i}",
                }

    corpus = IndicVoicesCorpus(dataset_id="test/dataset")

    import sys
    import types

    mock_datasets = types.ModuleType("datasets")
    mock_datasets.load_dataset = lambda *args, **kwargs: MockDataset()
    sys.modules["datasets"] = mock_datasets

    try:
        items = list(corpus.stream("hi-IN", limit=100))
        assert len(items) == 10  # Should produce all 10 items without error
    finally:
        del sys.modules["datasets"]
