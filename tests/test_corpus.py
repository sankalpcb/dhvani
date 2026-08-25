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


def test_language_tags_map_to_the_datasets_real_config_names():
    """stream() derived its config as lang.split("-")[0], asking for "hi",
    "kn", "ml". IndicVoices names its configs by full language: "hindi",
    "kannada", "malayalam". Every collect run against the real corpus would
    have failed on the first language, for all three the project targets.

    The failure was also misdirecting: `datasets` reports a config it cannot
    find on a gated repo as "is a gated dataset ... ask for access", so the
    obvious reading was a permissions problem, not a wrong name.
    """
    from dhvani.corpus import indicvoices_config

    assert indicvoices_config("hi-IN") == "hindi"
    assert indicvoices_config("kn-IN") == "kannada"
    assert indicvoices_config("ml-IN") == "malayalam"


def test_every_default_language_is_mappable():
    """The three languages the CLI ships as defaults must all resolve, or
    `dhvani-calibrate collect` breaks with no arguments at all."""
    from dhvani.cli_calibrate import DEFAULT_LANGS
    from dhvani.corpus import indicvoices_config

    for lang in DEFAULT_LANGS:
        assert indicvoices_config(lang)


def test_a_bare_language_code_works_too():
    """Accept "hi" as well as "hi-IN": Tier0Conformer speaks the bare code
    and the corpus speaks the full tag, and this has already been a bug once
    (I6, one backend per language)."""
    from dhvani.corpus import indicvoices_config

    assert indicvoices_config("hi") == indicvoices_config("hi-IN")


def test_an_unknown_language_says_what_is_available():
    """Better than a downstream "gated dataset" error that sends the reader
    to the permissions page for a typo."""
    from dhvani.corpus import indicvoices_config

    with pytest.raises(ValueError) as exc:
        indicvoices_config("xx-XX")
    message = str(exc.value)
    assert "xx-XX" in message
    assert "hindi" in message, "must list what IS available"


def _fake_audio_bytes(seconds=1.0, rate=16000):
    import io
    import soundfile as sf
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
             rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_a_row_in_indicvoices_real_schema_is_extracted():
    """The published schema is not the one this module was written against.

    IndicVoices stores audio under `audio_filepath` as {bytes, path} -- the
    ENCODED file -- not under `audio` as {array, sampling_rate}. collect()
    would have failed its schema probe on every row even once streaming
    worked and the config name was right.
    """
    from dhvani.corpus import _extract_item_from_row

    item = _extract_item_from_row({
        "audio_filepath": {"bytes": _fake_audio_bytes(), "path": "x.wav"},
        "text": "नमस्ते दुनिया",
        "speaker_id": "S1", "district": "Narsinghpur",
    }, "hi-IN")

    assert item is not None, "real IndicVoices rows must extract"
    assert item.reference == "नमस्ते दुनिया"
    assert item.speaker_id == "S1"
    assert item.district == "Narsinghpur"
    assert item.duration_ms > 0
    assert item.pcm.dtype == np.int16


def test_the_previously_assumed_schema_still_works():
    """Decoded {array, sampling_rate} rows must keep extracting -- other
    corpora and the existing tests supply that shape."""
    from dhvani.corpus import _extract_item_from_row

    item = _extract_item_from_row({
        "audio": {"array": np.zeros(16000, dtype=np.float32),
                  "sampling_rate": 16000},
        "text": "हैलो",
    }, "hi-IN")
    assert item is not None and item.duration_ms == 1000


def test_a_row_with_no_usable_audio_is_skipped_not_crashed():
    from dhvani.corpus import _extract_item_from_row

    assert _extract_item_from_row({"text": "x"}, "hi-IN") is None
    assert _extract_item_from_row(
        {"audio_filepath": {"bytes": None, "path": "p"}, "text": "x"},
        "hi-IN") is None
    assert _extract_item_from_row(
        {"audio_filepath": {"bytes": _fake_audio_bytes()}, "text": "   "},
        "hi-IN") is None


def _write_shard(path, rows):
    """A parquet file shaped like a real IndicVoices shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def _corpus_over(monkeypatch, shard_path, dataset_id="test/dataset"):
    from dhvani.corpus import IndicVoicesCorpus
    corpus = IndicVoicesCorpus(dataset_id=dataset_id)
    seen = {}

    def shard_names(config):
        seen["config"] = config
        return ["shard-0.parquet"]

    monkeypatch.setattr(corpus, "shard_names", shard_names)
    monkeypatch.setattr(corpus, "local_shard", lambda name: shard_path)
    return corpus, seen


def test_stream_asks_for_the_mapped_config(tmp_path, monkeypatch):
    """The language mapping has to be wired into stream(), not merely exist:
    IndicVoices publishes "kannada", never "kn"."""
    shard = _write_shard(tmp_path / "s.parquet", [{
        "audio_filepath": {"bytes": _fake_audio_bytes(), "path": "a.wav"},
        "text": "ನಮಸ್ಕಾರ", "speaker_id": "S1", "district": "D",
    }])
    corpus, seen = _corpus_over(monkeypatch, shard)

    items = list(corpus.stream("kn-IN", limit=1))

    assert seen["config"] == "kannada", f"asked for {seen.get('config')!r}"
    assert len(items) == 1


def test_a_shard_whose_columns_are_wrong_raises_rather_than_yielding_nothing():
    """Silence is the dangerous failure here: an empty corpus produces an
    empty scored.json, and the calibration reports having measured nothing
    rather than that it could not read the data. The column names have
    genuinely changed once already."""
    import pytest as _pytest
    from dhvani.corpus import SCHEMA_PROBE_ROWS
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    shard = _write_shard(tmp / "bad.parquet", [
        {"wrong_audio": "x", "wrong_text": "y"}
        for _ in range(SCHEMA_PROBE_ROWS + 10)
    ])

    class _C:
        pass
    from dhvani.corpus import IndicVoicesCorpus
    corpus = IndicVoicesCorpus(dataset_id="test/dataset")
    corpus.shard_names = lambda config: ["s.parquet"]
    corpus.local_shard = lambda name: shard

    with _pytest.raises(ValueError) as exc:
        list(corpus.stream("hi-IN", limit=10))
    message = str(exc.value).lower()
    assert "schema mismatch" in message
    assert "actual row keys" in message


def test_a_small_shard_that_yields_items_does_not_raise(tmp_path, monkeypatch):
    """Fewer rows than the probe threshold must not be mistaken for a
    broken schema."""
    shard = _write_shard(tmp_path / "s.parquet", [{
        "audio_filepath": {"bytes": _fake_audio_bytes(), "path": f"{i}.wav"},
        "text": f"वाक्य {i}", "speaker_id": f"S{i}", "district": "D",
    } for i in range(3)])
    corpus, _ = _corpus_over(monkeypatch, shard)

    items = list(corpus.stream("hi-IN", limit=10))
    assert len(items) == 3


def test_stream_stops_at_the_limit_without_fetching_more_shards(tmp_path, monkeypatch):
    """Each shard is ~0.5 GB, so honouring the limit is a disk and bandwidth
    guarantee, not just a correctness one."""
    shard = _write_shard(tmp_path / "s.parquet", [{
        "audio_filepath": {"bytes": _fake_audio_bytes(), "path": f"{i}.wav"},
        "text": f"वाक्य {i}",
    } for i in range(20)])

    from dhvani.corpus import IndicVoicesCorpus
    corpus = IndicVoicesCorpus(dataset_id="test/dataset")
    fetched = []
    corpus.shard_names = lambda config: ["a.parquet", "b.parquet", "c.parquet"]
    corpus.local_shard = lambda name: (fetched.append(name), shard)[1]

    items = list(corpus.stream("hi-IN", limit=5))

    assert len(items) == 5
    assert fetched == ["a.parquet"], f"downloaded more than needed: {fetched}"
