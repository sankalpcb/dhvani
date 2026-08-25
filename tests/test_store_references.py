import pytest
from dhvani.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_reference_round_trips(store):
    store.put_reference("a" * 64, "नमस्ते", "hi-IN", "spk1", "Pune")
    got = store.get_reference("a" * 64)
    assert got == {"reference": "नमस्ते", "lang": "hi-IN",
                   "speaker_id": "spk1", "district": "Pune"}


def test_put_reference_is_idempotent(store):
    assert store.put_reference("a" * 64, "first", "hi-IN") is True
    assert store.put_reference("a" * 64, "SECOND", "hi-IN") is False
    assert store.get_reference("a" * 64)["reference"] == "first"


def test_get_missing_reference_returns_none(store):
    assert store.get_reference("nope") is None


def test_speaker_and_district_are_optional(store):
    store.put_reference("b" * 64, "text", "kn-IN")
    got = store.get_reference("b" * 64)
    assert got["speaker_id"] is None and got["district"] is None


def test_references_do_not_disturb_hypotheses(store):
    """The new table must not interfere with the existing content-addressed cache."""
    store.put_segment("c" * 64, "vid1", 0, 3000)
    store.put_hypothesis("c" * 64, "tier0", "hyp", {}, 0.0)
    store.put_reference("c" * 64, "ref", "hi-IN")
    assert store.get_hypothesis("c" * 64, "tier0")["text"] == "hyp"
    assert store.get_reference("c" * 64)["reference"] == "ref"
