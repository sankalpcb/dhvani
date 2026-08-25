"""Calibration corpus sources.

Calibration bypasses the segmenter: one dataset utterance is one segment, so
each segment keeps its own reference transcript (spec §1.2). segment_id is
still SHA256 of normalized PCM, so the store, cache and fixture layers work
unchanged.

IndicVoicesCorpus imports `datasets` lazily, inside stream(), so importing
this module costs nothing and the test suite runs with the `data` extra
absent (goal G5).

A corpus is anything with stream(lang, limit) and a `dataset_id` string.
dataset_id is identity, not decoration: calibrate.collect() writes it into
each segment's source_id, so a second dataset collected into the same --db
stays distinguishable from the first.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from dhvani.audio import normalize
from dhvani.config import SAMPLE_RATE
from dhvani.ids import segment_id as compute_id

# Threshold for probing dataset schema. If this many rows have been examined
# without producing any items, the field name guesses are likely wrong and we
# should fail loud rather than silently yield nothing.
SCHEMA_PROBE_ROWS = 50

DEFAULT_DATASET = "ai4bharat/IndicVoices"

INDICVOICES_CONFIGS = {
    "as": "assamese", "bn": "bengali", "brx": "bodo", "doi": "dogri",
    "gu": "gujarati", "hi": "hindi", "kn": "kannada", "ks": "kashmiri",
    "kok": "konkani", "mai": "maithili", "ml": "malayalam", "mni": "manipuri",
    "mr": "marathi", "ne": "nepali", "or": "odia", "pa": "punjabi",
    "sa": "sanskrit", "sat": "santali", "sd": "sindhi", "ta": "tamil",
    "te": "telugu", "ur": "urdu",
}
"""Language code -> the config name IndicVoices actually publishes.

stream() used to derive this as lang.split("-")[0], asking the dataset for
"hi", "kn", "ml". IndicVoices names its configs by full language: "hindi",
"kannada", "malayalam". Every collect run against the real corpus failed on
the first language, for all three the project targets -- the harness had
only ever been exercised against FakeCorpus.

Worth knowing how that failure presented, because it does not point at the
cause: `datasets` reports a config it cannot resolve on a gated repo as
"Dataset 'ai4bharat/IndicVoices' is a gated dataset ... ask for access", so
a wrong config name looks exactly like a permissions problem.
"""


def indicvoices_config(lang: str) -> str:
    """Resolve "hi-IN" (or bare "hi") to the dataset's config name.

    Both spellings are accepted deliberately: Tier0Conformer speaks the bare
    code and the corpus speaks the full tag, and that mismatch has already
    caused one bug (I6 -- one Tier 0 backend per language).
    """
    code = lang.split("-")[0]
    try:
        return INDICVOICES_CONFIGS[code]
    except KeyError:
        raise ValueError(
            f"no IndicVoices config for language {lang!r}. "
            f"Available: {', '.join(sorted(INDICVOICES_CONFIGS.values()))}"
        ) from None
"""The corpus a calibration run streams unless --dataset says otherwise.

Named rather than inlined so IndicVoicesCorpus's default and the CLI's
advertised default cannot drift apart; a test pins the help text to it.
"""


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


def _decode_audio(audio: dict):
    """(samples, rate) from a dataset audio cell, or None if unusable.

    Two shapes, because the published data is not the shape this module was
    written against. `datasets` hands back a DECODED {array, sampling_rate}
    when its Audio feature is applied; a raw IndicVoices parquet row carries
    the ENCODED file as {bytes, path}. Decoding the bytes here with
    soundfile is also what keeps `torchcodec` out of the dependency set --
    datasets>=4 refuses to decode audio without it.
    """
    array = audio.get("array")
    if array is not None:
        return np.asarray(array), audio["sampling_rate"]

    blob = audio.get("bytes")
    if blob:
        import io

        import soundfile as sf
        samples, rate = sf.read(io.BytesIO(blob), dtype="float32")
        return np.asarray(samples), rate
    return None


def _extract_item_from_row(row: dict, lang: str) -> CorpusItem | None:
    """Extract a CorpusItem from a dataset row, or None if fields are missing.

    IndicVoices names its audio column `audio_filepath`, not `audio`, and
    carries the encoded file rather than a decoded array -- so every row
    failed the old extraction and collect() would have died on its schema
    probe even after the config name was fixed. Both column names and both
    audio shapes are accepted now; `audio` first, since that is what the
    other corpora and the tests supply.

    Reference text likewise: IndicVoices publishes `text` alongside
    `normalized` and `verbatim`. `text` is preferred and the others are
    fallbacks, so a shard missing one still yields ground truth.
    """
    audio = row.get("audio") or row.get("audio_filepath")
    reference = (row.get("text") or row.get("normalized")
                 or row.get("verbatim") or row.get("transcript") or "")
    if not audio or not reference.strip():
        return None

    decoded = _decode_audio(audio)
    if decoded is None:
        return None

    samples, rate = decoded
    return _make_item(
        samples, rate, reference, lang,
        row.get("speaker_id"), row.get("district"),
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

    def __init__(self, items, dataset_id: str = "fake"):
        # items: list of (raw_audio, src_rate, reference, lang, speaker_id, district)
        self._items = list(items)
        # Part of the corpus protocol alongside stream(): collect() records
        # it so the segments table says which corpus a row came from. The
        # default is deliberately not a plausible dataset name.
        self.dataset_id = dataset_id

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

    def __init__(self, dataset_id: str = DEFAULT_DATASET):
        self.dataset_id = dataset_id

    def shard_names(self, config: str) -> list[str]:
        """The parquet files for one language, in published order."""
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(self.dataset_id, repo_type="dataset")
        return sorted(f for f in files
                      if f.startswith(config + "/") and f.endswith(".parquet"))

    def local_shard(self, name: str) -> str:
        """Fetch one shard, returning its local path. Cached by the Hub."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(self.dataset_id, name, repo_type="dataset")

    def stream(self, lang: str, limit: int) -> Iterator[CorpusItem]:
        """Yield up to `limit` utterances for one language.

        Reads downloaded parquet shards rather than `datasets` streaming.
        Streaming was not merely slower -- it was unusable: 29 minutes of
        load_dataset(streaming=True) yielded ZERO rows for hindi, on a link
        that pulls a whole 0.5 GB shard in 39 seconds. Downloading the shard
        and reading it with pyarrow turns the same work into one 39-second
        fetch plus a local scan of 5,429 rows.

        It also drops a dependency rather than adding one: datasets>=4
        refuses to decode audio without `torchcodec`, and reading the
        encoded bytes straight out of the parquet sidesteps that entirely
        (see _decode_audio).

        Shards are fetched lazily, one at a time, and only until `limit` is
        met -- each is about 0.5 GB, so pulling all 83 for hindi would be
        50 GB nobody asked for.
        """
        import pyarrow.parquet as pq

        config = indicvoices_config(lang)
        count = 0
        rows_examined = 0
        sample_row = None

        for shard in self.shard_names(config):
            if count >= limit:
                return
            parquet = pq.ParquetFile(self.local_shard(shard))

            for batch in parquet.iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    if count >= limit:
                        return
                    rows_examined += 1
                    if sample_row is None:
                        sample_row = row

                    item = _extract_item_from_row(row, lang)
                    if item is None:
                        # Fail loud on a schema that yields nothing at all,
                        # rather than quietly returning an empty corpus --
                        # the column names really did change once already.
                        if rows_examined >= SCHEMA_PROBE_ROWS and count == 0:
                            raise ValueError(
                                f"Dataset schema mismatch: examined "
                                f"{rows_examined} rows from dataset "
                                f"'{self.dataset_id}' config '{config}', but "
                                f"no items produced. Expected fields: "
                                f"['audio' or 'audio_filepath', "
                                f"'text/normalized/verbatim/transcript']. "
                                f"Actual row keys: "
                                f"{list(sample_row) if sample_row else []}"
                            )
                        continue

                    yield item
                    count += 1
