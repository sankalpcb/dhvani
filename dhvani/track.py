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
    """
    by_id = {e.segment_id: e for e in base}

    for segment_id, upd in updates.items():
        current = by_id.get(segment_id)
        if current is None:
            # A result for a segment this track does not contain. Ignoring it
            # protects I1: the merge can never grow the track.
            continue
        risk = float(upd["risk"])
        by_id[segment_id] = replace(
            current, text=upd["text"], risk=risk, band=band_of(risk)
        )

    return sorted(by_id.values(), key=lambda e: (e.t_start_ms, e.segment_id))


def entries_to_json(entries: list[TrackEntry]) -> str:
    ordered = sorted(entries, key=lambda e: (e.t_start_ms, e.segment_id))
    return json.dumps([e.__dict__ for e in ordered],
                      ensure_ascii=False, indent=2, sort_keys=True)


def entries_from_json(payload: str) -> list[TrackEntry]:
    return [TrackEntry(**row) for row in json.loads(payload)]
