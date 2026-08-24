"""Synchronous orchestration: segment, transcribe, score, route, band."""

from dataclasses import dataclass

import numpy as np

from dhvani.config import TAU_FLAG, TAU_SHIP
from dhvani.metrics import Timer
from dhvani.scorer import extract, risk as compute_risk
from dhvani.segmenter import segment as split


@dataclass(frozen=True)
class TrackEntry:
    segment_id: str
    t_start_ms: int
    t_end_ms: int
    text: str
    risk: float
    band: str


def band_of(risk: float) -> str:
    """Spec §6.2 output bands. Nothing below tau_flag ships silently."""
    if risk < TAU_SHIP:
        return "ship"
    if risk < TAU_FLAG:
        return "marked"
    return "review"


def run(pcm: np.ndarray, source_id: str, tier0, store,
        delta_table: dict, budget_usd: float,
        samples: dict | None = None) -> list[TrackEntry]:
    """Produce a caption track. Cached segments are never re-transcribed.

    samples, when given, collects Tier 0 transcription wall-clock timings
    (spec §9.1) into samples["tier0"], one entry per actual backend call
    (cache hits are not timed -- they are not the work being measured).
    Callers that pass nothing get exactly today's behavior.

    delta_table and budget_usd are accepted for interface stability but are
    unused in Phase 1: escalation is computed offline by report.frontier()
    (Task 11), not here. Phase 2 wires them into asynchronous Tier 1
    submission. run() itself never imports dhvani.router and never touches
    store attributes beyond the documented Store interface (put_segment,
    get_hypothesis, put_hypothesis) — an earlier draft built router
    Candidates and assigned an undocumented store.escalation_plan attribute;
    that was removed as untested and out of scope for Phase 1.

    Deterministic (invariant I5): identical pcm/source_id/tier0 responses
    always produce identical output, regardless of what is already cached.
    """
    segments = split(pcm)
    entries: list[TrackEntry] = []

    for seg in segments:
        store.put_segment(seg.segment_id, source_id, seg.t_start_ms, seg.t_end_ms)

        # The backend's variant identity (lang, model_id, ...) is part of
        # the cache key: identical PCM decoded under a different config is
        # a different hypothesis. See FIX ROUND 2 (I2/I3) in store.py.
        variant = tier0.variant_key
        cached = store.get_hypothesis(seg.segment_id, "tier0", variant_key=variant)
        if cached is None:
            with Timer() as t:
                result = tier0.transcribe(seg)
            if samples is not None:
                samples.setdefault("tier0", []).append(t.elapsed_ms)
            store.put_hypothesis(
                seg.segment_id, "tier0", result["text"],
                result["signals"], tier0.cost_per_call(seg),
                variant_key=variant,
            )
        else:
            result = {"text": cached["text"], "signals": cached["signals"]}

        duration = seg.t_end_ms - seg.t_start_ms
        features = extract(result["text"], result["signals"], duration)
        r = compute_risk(features)

        entries.append(TrackEntry(
            segment_id=seg.segment_id,
            t_start_ms=seg.t_start_ms,
            t_end_ms=seg.t_end_ms,
            text=result["text"],
            risk=r,
            band=band_of(r),
        ))

    return entries
