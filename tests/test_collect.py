import numpy as np
import pytest

from dhvani.calibrate import collect
from dhvani.corpus import FakeCorpus
from dhvani.store import Store


class StubTier0:
    """A Tier 0 stub whose variant_key depends on lang, like the real one.

    I6: a stub with a hardcoded variant_key cannot show the bug where one
    Hindi-configured backend is reused for every language -- the cache keys
    would look identical either way.
    """

    name = "tier0"

    def __init__(self, lang="hi"):
        self.lang = lang
        self.calls = 0

    @property
    def variant_key(self):
        return f"tier0|{self.lang}|m"

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        self.calls += 1
        return {"text": f"नमस्ते world {self.lang}",
                "signals": {"ctc_rnnt_disagreement": 0.4}}


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


@pytest.fixture
def pcm_dir(tmp_path):
    """Where phase 1 parks the audio phase 2 sends to Tier 1 inline."""
    return str(tmp_path / "pcm")


def test_collect_returns_one_scored_item_per_utterance(store, pcm_dir):
    out = collect(_corpus(3), lambda lang: StubTier0(), store, ["hi-IN"], per_lang=3,
                 pcm_cache_dir=pcm_dir)
    assert len(out) == 3
    assert all(0.0 <= s["risk"] <= 1.0 for s in out)


def test_collect_persists_reference_and_hypothesis(store, pcm_dir):
    out = collect(_corpus(1), lambda lang: StubTier0(), store, ["hi-IN"], per_lang=1,
                  pcm_cache_dir=pcm_dir)
    sid = out[0]["segment_id"]
    assert store.get_reference(sid)["reference"] == "ref-0"
    assert store.get_hypothesis(sid, "tier0", "tier0|hi|m")["text"] == "नमस्ते world hi"


def test_collect_is_resumable_and_does_not_retranscribe(store, pcm_dir):
    """The property that makes a multi-hour run survivable."""
    tier0 = StubTier0()
    collect(_corpus(3), lambda lang: tier0, store, ["hi-IN"], per_lang=3,
                 pcm_cache_dir=pcm_dir)
    first = tier0.calls
    collect(_corpus(3), lambda lang: tier0, store, ["hi-IN"], per_lang=3,
                 pcm_cache_dir=pcm_dir)
    assert tier0.calls == first, "cached segments must not be re-transcribed"


def test_collect_scores_identically_on_a_cached_rerun(store, pcm_dir):
    tier0 = StubTier0()
    a = collect(_corpus(3), lambda lang: tier0, store, ["hi-IN"], per_lang=3,
                 pcm_cache_dir=pcm_dir)
    b = collect(_corpus(3), lambda lang: tier0, store, ["hi-IN"], per_lang=3,
                 pcm_cache_dir=pcm_dir)
    assert a == b


def test_collect_spans_requested_languages(store, pcm_dir):
    rng = np.random.default_rng(1)
    corpus = FakeCorpus([
        (0.3 * rng.standard_normal(32000), 16000, "h", "hi-IN", "s1", "d1"),
        (0.3 * rng.standard_normal(32000), 16000, "k", "kn-IN", "s2", "d2"),
    ])
    out = collect(corpus, lambda lang: StubTier0(), store, ["hi-IN", "kn-IN"], per_lang=1,
                  pcm_cache_dir=pcm_dir)
    assert {s["lang"] for s in out} == {"hi-IN", "kn-IN"}


def test_collect_carries_duration_for_later_pricing(store, pcm_dir):
    out = collect(_corpus(1), lambda lang: StubTier0(), store, ["hi-IN"], per_lang=1,
                  pcm_cache_dir=pcm_dir)
    assert out[0]["duration_ms"] == 2000


def test_collect_records_the_tier0_variant_it_stored_under(store, pcm_dir):
    """C1: phase 2 is a separate process and cannot re-derive this key.
    Whatever cache key the hypothesis went in under must travel in
    scored.json, or the escalate pass looks under the wrong one and skips
    every segment."""
    tier0 = StubTier0()
    out = collect(_corpus(2), lambda lang: tier0, store, ["hi-IN"], per_lang=2,
                  pcm_cache_dir=pcm_dir)
    assert all(s["tier0_variant"] == tier0.variant_key for s in out)
    for s in out:
        assert store.get_hypothesis(s["segment_id"], "tier0",
                                    s["tier0_variant"]) is not None


# --- C2: phase 2 sends the audio inline, so phase 1 must cache it ---

def test_collect_caches_pcm_for_every_segment(store, pcm_dir):
    """Tier1Chirp.transcribe() sends segment.pcm inline. Phase 2 runs in a
    separate process with no corpus in hand, so if phase 1 does not write
    the audio down there is nothing to send."""
    import os

    from dhvani.calibrate import load_pcm, pcm_cache_path

    out = collect(_corpus(3), lambda lang: StubTier0(), store, ["hi-IN"], per_lang=3,
                  pcm_cache_dir=pcm_dir)
    for s in out:
        assert os.path.exists(pcm_cache_path(pcm_dir, s["segment_id"]))
        pcm = load_pcm(pcm_dir, s["segment_id"])
        assert pcm.dtype == np.int16
        assert len(pcm) > 1, "must be the real audio, not a silence stub"


def test_cached_pcm_round_trips_to_the_same_segment_id(store, pcm_dir):
    """segment_id is SHA256 of these exact bytes, so a cache read that does
    not re-hash to the same id would mean the wrong audio was billed."""
    from dhvani.calibrate import load_pcm
    from dhvani.ids import segment_id as compute_id

    out = collect(_corpus(2), lambda lang: StubTier0(), store, ["hi-IN"], per_lang=2,
                  pcm_cache_dir=pcm_dir)
    for s in out:
        assert compute_id(load_pcm(pcm_dir, s["segment_id"])) == s["segment_id"]


def test_missing_cached_pcm_raises_naming_the_segment_and_path(tmp_path):
    """Never a silent skip and never a zero array: both were live in this
    code's history and both corrupt the table without saying so."""
    from dhvani.calibrate import PcmCacheMiss, load_pcm

    empty = str(tmp_path / "nothing")
    with pytest.raises(PcmCacheMiss) as exc:
        load_pcm(empty, "deadbeef" * 8)
    message = str(exc.value)
    assert "deadbeef" * 8 in message
    assert empty in message


def _corrupt_cache_entry(cache_dir, segment_id, keep):
    """Write a real .npy for segment_id, then truncate it to `keep` bytes.

    `keep` as a fraction of the file, or an absolute count. This is what a
    collect run killed mid-np.save leaves on disk -- and save_pcm() skips
    any path that already exists, so re-running collect steps straight over
    the wreckage instead of repairing it.
    """
    from dhvani.calibrate import pcm_cache_path, save_pcm

    save_pcm(cache_dir, segment_id, np.arange(5000, dtype=np.int16))
    path = pcm_cache_path(cache_dir, segment_id)
    whole = open(path, "rb").read()
    with open(path, "wb") as fh:
        fh.write(whole[:keep])
    return path


@pytest.mark.parametrize("keep,shape", [
    (0, "nothing at all"),
    (4, "half a magic string"),
    (128, "a complete header and no data"),
    (5000, "a complete header and half the data"),
])
def test_truncated_cached_pcm_raises_a_clean_diagnostic(tmp_path, keep, shape):
    """numpy's own errors here are opaque, inconsistent, and in one case
    actively misleading.

    The four truncations below raise three different exception types, and
    the 4-byte one reports "This file contains pickled (object) data" and
    suggests allow_pickle=True -- advice that is both wrong for a truncated
    array and dangerous to follow. None of them name the segment, name the
    cache, or say what to do. load_pcm() owns that diagnostic for the
    missing case already; it must own it for the unreadable case too.
    """
    from dhvani.calibrate import PcmCacheCorrupt, load_pcm

    cache = str(tmp_path / "pcm")
    segment_id = "deadbeef" * 8
    path = _corrupt_cache_entry(cache, segment_id, keep)

    with pytest.raises(PcmCacheCorrupt) as exc:
        load_pcm(cache, segment_id)

    message = str(exc.value)
    assert segment_id in message, f"must name the segment ({shape})"
    assert path in message, f"must name the file ({shape})"
    # The operationally load-bearing half: save_pcm() skips files that
    # already exist, so "just re-run collect" -- the remedy for a cache
    # MISS -- silently does nothing here. Saying so is the whole point.
    assert "delete" in message.lower(), f"must give the remedy ({shape})"


def test_a_corrupt_cache_entry_is_not_reported_as_a_missing_one(tmp_path):
    """Different remedies, so they must not collapse into one error.

    A miss is fixed by re-running collect. A corrupt entry is not -- and an
    operator told "re-run collect" for a file collect will skip is sent
    into a loop that cannot terminate.
    """
    from dhvani.calibrate import PcmCacheCorrupt, PcmCacheMiss, load_pcm

    cache = str(tmp_path / "pcm")
    segment_id = "deadbeef" * 8
    _corrupt_cache_entry(cache, segment_id, 128)

    with pytest.raises(PcmCacheCorrupt) as exc:
        load_pcm(cache, segment_id)
    assert not isinstance(exc.value, PcmCacheMiss)


def test_both_cache_failures_share_a_base_the_cli_can_catch(tmp_path):
    """cli_calibrate's escalate path turns a cache failure into a message
    and exit 3 rather than a traceback. It caught PcmCacheMiss by name, so
    a new sibling would have sailed straight past it -- both are now
    PcmCacheError, which is what that handler catches."""
    from dhvani.calibrate import PcmCacheCorrupt, PcmCacheError, PcmCacheMiss

    assert issubclass(PcmCacheMiss, PcmCacheError)
    assert issubclass(PcmCacheCorrupt, PcmCacheError)


def test_an_intact_cache_entry_still_loads(tmp_path):
    """Guard against the diagnostic swallowing the happy path: the new
    try/except must not turn a perfectly good load into an error."""
    from dhvani.calibrate import load_pcm, save_pcm

    cache = str(tmp_path / "pcm")
    segment_id = "deadbeef" * 8
    original = np.arange(5000, dtype=np.int16)
    save_pcm(cache, segment_id, original)

    assert np.array_equal(load_pcm(cache, segment_id), original)


# --- R9: byte-identical utterances are ONE segment ---

def test_duplicate_audio_yields_one_scored_entry(store, pcm_dir):
    """segment_id is content-addressed, so the same audio twice is one
    segment. Two scored entries would let it clear MIN_BUCKET_SAMPLES by
    itself and would weight its delta twice in the bucket mean."""
    rng = np.random.default_rng(7)
    raw = 0.3 * rng.standard_normal(32000)
    corpus = FakeCorpus([
        (raw, 16000, "same utterance", "hi-IN", "spk1", "d1"),
        (raw, 16000, "same utterance", "hi-IN", "spk2", "d2"),
    ])
    out = collect(corpus, lambda lang: StubTier0(), store, ["hi-IN"], per_lang=2,
                  pcm_cache_dir=pcm_dir)
    assert len(out) == 1
    assert len({s["segment_id"] for s in out}) == 1


# --- I6: one backend per language, not one Hindi backend for all three ---

def test_collect_builds_one_backend_per_language(store, pcm_dir):
    """The CLI built a single Tier0Conformer() -- default lang="hi" -- and
    collect() reused it for every language, so Kannada and Malayalam audio
    was decoded as Hindi."""
    made = []

    def make(lang):
        tier0 = StubTier0(lang=lang)
        made.append(tier0)
        return tier0

    rng = np.random.default_rng(11)
    corpus = FakeCorpus([
        (0.3 * rng.standard_normal(32000), 16000, "h", "hi-IN", "s1", "d1"),
        (0.3 * rng.standard_normal(32000), 16000, "k", "kn-IN", "s2", "d2"),
    ])
    out = collect(corpus, make, store, ["hi-IN", "kn-IN"], per_lang=1,
                  pcm_cache_dir=pcm_dir)

    assert [t.lang for t in made] == ["hi-IN", "kn-IN"]
    assert len({t.variant_key for t in made}) == 2
    assert len({s["tier0_variant"] for s in out}) == 2


def test_identical_audio_in_two_languages_gets_distinct_cache_entries(store, pcm_dir):
    """variant_key was identical across languages, so the store could not
    tell the three runs apart: the Hindi hypothesis was served for the
    Kannada one on a cache hit."""
    rng = np.random.default_rng(13)
    raw = 0.3 * rng.standard_normal(32000)

    hi = collect(FakeCorpus([(raw, 16000, "ref", "hi-IN", "s1", "d1")]),
                 lambda lang: StubTier0(lang=lang), store, ["hi-IN"],
                 per_lang=1, pcm_cache_dir=pcm_dir)
    kn = collect(FakeCorpus([(raw, 16000, "ref", "kn-IN", "s1", "d1")]),
                 lambda lang: StubTier0(lang=lang), store, ["kn-IN"],
                 per_lang=1, pcm_cache_dir=pcm_dir)

    sid = hi[0]["segment_id"]
    assert kn[0]["segment_id"] == sid, "identical audio must be one segment id"
    assert hi[0]["tier0_variant"] != kn[0]["tier0_variant"]
    assert (store.get_hypothesis(sid, "tier0", hi[0]["tier0_variant"])["text"]
            != store.get_hypothesis(sid, "tier0", kn[0]["tier0_variant"])["text"]), (
        "two languages must not share one cache entry for the same audio"
    )
