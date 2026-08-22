"""Synchronous orchestration: segment, transcribe, score, route, band."""

from dataclasses import dataclass

import numpy as np

from dhvani.config import TAU_FLAG, TAU_SHIP
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
        delta_table: dict, budget_usd: float) -> list[TrackEntry]:
    """Produce a caption track. Cached segments are never re-transcribed.

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

        cached = store.get_hypothesis(seg.segment_id, "tier0")
        if cached is None:
            result = tier0.transcribe(seg)
            store.put_hypothesis(
                seg.segment_id, "tier0", result["text"],
                result["signals"], tier0.cost_per_call(seg),
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
