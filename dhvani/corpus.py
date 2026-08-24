"""Calibration corpus sources.

Calibration bypasses the segmenter: one dataset utterance is one segment, so
each segment keeps its own reference transcript (spec §1.2). segment_id is
still SHA256 of normalized PCM, so the store, cache and fixture layers work
unchanged.

IndicVoicesCorpus imports `datasets` lazily, inside stream(), so importing
this module costs nothing and the test suite runs with the `data` extra
absent (goal G5).
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from dhvani.audio import normalize
from dhvani.config import SAMPLE_RATE
from dhvani.ids import segment_id as compute_id


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

    def __init__(self, items):
        # items: list of (raw_audio, src_rate, reference, lang, speaker_id, district)
        self._items = list(items)

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

    def __init__(self, dataset_id: str = "ai4bharat/IndicVoices"):
        self.dataset_id = dataset_id

    def stream(self, lang: str, limit: int) -> Iterator[CorpusItem]:
        from datasets import load_dataset  # lazy: keeps the `data` extra optional

        config = lang.split("-")[0]  # "hi-IN" -> "hi"
        ds = load_dataset(self.dataset_id, config, split="train", streaming=True)

        count = 0
        for row in ds:
            if count >= limit:
                return
            audio = row.get("audio")
            reference = row.get("text") or row.get("transcript") or ""
            if not audio or not reference.strip():
                continue
            yield _make_item(
                audio["array"], audio["sampling_rate"], reference, lang,
                row.get("speaker_id"), row.get("district"),
            )
            count += 1
