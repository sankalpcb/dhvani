"""Versioned caption tracks and the idempotent merge that advances them.

Spec §7: Tier 1 results arrive up to 24h after submission, possibly out of
order, possibly partial, possibly twice. merge_entries() is what makes that
safe. It is a pure keyed replacement over segment_id, so:

  I1 (no loss)      -- entries are only ever replaced, never removed, and an
                       update for an unknown segment_id is ignored rather
                       than inserted.
  I2 (idempotent)   -- replacing an entry with the same values twice yields
                       the same track; merge(merge(t, u), u) == merge(t, u).
  I5 (determinism)  -- output is sorted by (t_start_ms, segment_id), so the
                       arrival order of updates cannot change the result.
"""

import json
from dataclasses import replace

from dhvani.pipeline import TrackEntry, band_of


def merge_entries(base: list[TrackEntry], updates: dict) -> list[TrackEntry]:
    """Fold arriving results into a track. Pure and order-independent.

    updates maps segment_id -> {"text": str, "risk": float}. The band is
    recomputed from the new risk rather than carried over, so an escalated
    segment can move out of the review band.

    base is not keyed by segment_id: byte-identical audio chunks legitimately
    share an id (segment_id is a hash of the pcm alone), so every entry whose
    segment_id appears in updates is replaced in place, preserving each
    entry's own timestamps. Output length is always len(base); an update
    naming a segment_id absent from base is ignored, never inserted.
    """
    merged = []
    for entry in base:
        upd = updates.get(entry.segment_id)
        if upd is None:
            merged.append(entry)
            continue
        risk = float(upd["risk"])
        merged.append(
            replace(entry, text=upd["text"], risk=risk, band=band_of(risk))
        )

    return sorted(merged, key=lambda e: (e.t_start_ms, e.segment_id))


def entries_to_json(entries: list[TrackEntry]) -> str:
    ordered = sorted(entries, key=lambda e: (e.t_start_ms, e.segment_id))
    return json.dumps([e.__dict__ for e in ordered],
                      ensure_ascii=False, indent=2, sort_keys=True)


def entries_from_json(payload: str) -> list[TrackEntry]:
    return [TrackEntry(**row) for row in json.loads(payload)]
