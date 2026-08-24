"""Calibration harness: measures the delta table the router needs.

Spec: docs/superpowers/specs/2026-08-24-dhvani-calibration-design.md

The router cannot pick its own calibration set — it selects by delta, and
delta is what is being measured (spec §1.1). So calibration escalates a
STRATIFIED sample across every risk bucket, deliberately including low-risk
ones, because that is the only way to discover the negative deltas that
invariant I3 exists to filter.
"""

from collections import defaultdict

from dhvani.router import bucket_of
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import Segment

# A bucket with fewer samples than this is OMITTED from the table rather
# than included with a noisy average. Omission degrades to "do not
# escalate"; a noisy average degrades to "escalate wrongly, and pay".
MIN_BUCKET_SAMPLES = 20

N_PER_BUCKET = 100


def histogram(scored: list[dict]) -> dict:
    """Bucket label -> count. Printed before any paid call so the risk
    distribution is visible while it is still free to act on."""
    counts: dict[str, int] = defaultdict(int)
    for item in scored:
        counts[bucket_of(item["risk"])] += 1
    return dict(sorted(counts.items()))


def stratify(scored: list[dict], n_per_bucket: int = N_PER_BUCKET) -> list[dict]:
    """Sample up to n_per_bucket from each risk bucket. Pure and deterministic.

    Selection sorts by segment_id, so a re-run picks the same segments and
    hits the content-addressed cache instead of paying again.
    """
    by_bucket: dict[str, list] = defaultdict(list)
    for item in scored:
        by_bucket[bucket_of(item["risk"])].append(item)

    chosen: list[dict] = []
    for bucket in sorted(by_bucket):
        members = by_bucket[bucket]
        if len(members) < MIN_BUCKET_SAMPLES:
            continue
        members.sort(key=lambda i: i["segment_id"])
        chosen.extend(members[:n_per_bucket])
    return chosen


def collect(corpus, tier0, store, langs, per_lang: int) -> list[dict]:
    """Phase 1: transcribe and score a corpus locally. Slow, free, resumable.

    One utterance is one segment (spec §1.2), so the segmenter is bypassed
    and every segment keeps its own reference. Already-transcribed segments
    are read from the store rather than re-run — that is what lets a
    multi-hour run be killed and restarted without losing work.
    """
    scored: list[dict] = []

    for lang in langs:
        for item in corpus.stream(lang, limit=per_lang):
            store.put_segment(item.segment_id, f"calib:{lang}", 0,
                              item.duration_ms, lang)
            store.put_reference(item.segment_id, item.reference, lang,
                                item.speaker_id, item.district)

            cached = store.get_hypothesis(item.segment_id, "tier0", tier0.variant_key)
            if cached is None:
                segment = Segment(item.segment_id, 0, item.duration_ms, item.pcm)
                result = tier0.transcribe(segment)
                store.put_hypothesis(item.segment_id, "tier0", result["text"],
                                     result["signals"], 0.0, tier0.variant_key)
            else:
                result = {"text": cached["text"], "signals": cached["signals"]}

            features = extract(result["text"], result["signals"], item.duration_ms)
            scored.append({
                "segment_id": item.segment_id,
                "risk": compute_risk(features),
                "lang": lang,
                "duration_ms": item.duration_ms,
            })

    return scored
