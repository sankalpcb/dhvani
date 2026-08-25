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


def _extract_item_from_row(row: dict, lang: str) -> CorpusItem | None:
    """Extract a CorpusItem from a dataset row, or None if fields are missing.

    Tests the row against the expected field names: "audio", "text"/"transcript",
    "speaker_id", "district". Returns None if required fields are absent.
    """
    audio = row.get("audio")
    reference = row.get("text") or row.get("transcript") or ""
    if not audio or not reference.strip():
        return None
    return _make_item(
        audio["array"], audio["sampling_rate"], reference, lang,
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

    def stream(self, lang: str, limit: int) -> Iterator[CorpusItem]:
        from datasets import load_dataset  # lazy: keeps the `data` extra optional

        config = indicvoices_config(lang)  # "hi-IN" -> "hindi"
        ds = load_dataset(self.dataset_id, config, split="train", streaming=True)

        count = 0
        rows_examined = 0
        items_yielded = 0
        sample_row = None

        for row in ds:
            rows_examined += 1
            if sample_row is None:
                sample_row = row

            if count >= limit:
                return

            item = _extract_item_from_row(row, lang)
            if item is None:
                # After SCHEMA_PROBE_ROWS rows with zero items, raise an exception
                if rows_examined >= SCHEMA_PROBE_ROWS and items_yielded == 0:
                    expected_fields = ["audio", "text/transcript", "speaker_id", "district"]
                    actual_keys = list(sample_row.keys()) if sample_row else []
                    raise ValueError(
                        f"Dataset schema mismatch: examined {rows_examined} rows from "
                        f"dataset '{self.dataset_id}' config '{config}', but no items produced. "
                        f"Expected fields: {expected_fields}. Actual row keys: {actual_keys}"
                    )
                continue

            yield item
            items_yielded += 1
            count += 1

            # Once we've yielded at least one item, the schema is correct
            if items_yielded >= 1:
                # Reset the probe so we don't raise later
                rows_examined = 0
