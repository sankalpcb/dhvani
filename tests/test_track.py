import pytest
from dhvani.pipeline import TrackEntry
from dhvani.track import merge_entries, entries_to_json, entries_from_json


def _base():
    return [
        TrackEntry("a" * 64, 0, 3000, "one", 0.9, "review"),
        TrackEntry("b" * 64, 3000, 6000, "two", 0.5, "marked"),
    ]


def test_merge_replaces_text_and_recomputes_band():
    out = merge_entries(_base(), {"a" * 64: {"text": "fixed", "risk": 0.1}})
    assert out[0].text == "fixed"
    assert out[0].risk == 0.1
    assert out[0].band == "ship"


def test_merge_leaves_untouched_entries_alone():
    out = merge_entries(_base(), {"a" * 64: {"text": "fixed", "risk": 0.1}})
    assert out[1].text == "two"
    assert out[1].band == "marked"


def test_merge_is_idempotent():
    """Invariant I2: applying the same result twice is a no-op."""
    upd = {"a" * 64: {"text": "fixed", "risk": 0.1}}
    once = merge_entries(_base(), upd)
    twice = merge_entries(once, upd)
    assert once == twice


def test_merge_ignores_unknown_segment_ids():
    """A late result for a segment not in this track must not invent an entry."""
    out = merge_entries(_base(), {"z" * 64: {"text": "ghost", "risk": 0.0}})
    assert [e.segment_id for e in out] == ["a" * 64, "b" * 64]


def test_merge_never_loses_entries():
    """Invariant I1: every input segment appears exactly once in the output."""
    out = merge_entries(_base(), {"a" * 64: {"text": "fixed", "risk": 0.1}})
    ids = [e.segment_id for e in out]
    assert sorted(ids) == sorted(e.segment_id for e in _base())
    assert len(ids) == len(set(ids))


def test_merge_output_is_order_independent():
    """Two updates applied in either order give byte-identical output."""
    u1 = {"a" * 64: {"text": "A", "risk": 0.2}}
    u2 = {"b" * 64: {"text": "B", "risk": 0.3}}
    left = merge_entries(merge_entries(_base(), u1), u2)
    right = merge_entries(merge_entries(_base(), u2), u1)
    assert left == right


def test_merge_sorts_by_time_then_segment_id():
    shuffled = list(reversed(_base()))
    out = merge_entries(shuffled, {})
    assert [e.t_start_ms for e in out] == [0, 3000]


def test_json_round_trip():
    entries = _base()
    assert entries_from_json(entries_to_json(entries)) == entries


def test_json_is_stable_for_equal_input():
    """Invariant I5: same entries -> byte-identical payload."""
    assert entries_to_json(_base()) == entries_to_json(list(reversed(_base())))
