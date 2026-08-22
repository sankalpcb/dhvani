import pytest
from dhvani.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_job_is_idempotent(store):
    assert store.put_job("j1", "tier1", "v1", ["a", "b"]) is True
    assert store.put_job("j1", "tier1", "v1", ["DIFFERENT"]) is False
    assert store.get_job("j1")["segment_ids"] == ["a", "b"]


def test_new_job_starts_pending_with_zero_attempts(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    job = store.get_job("j1")
    assert job["state"] == "pending"
    assert job["attempts"] == 0


def test_get_missing_job_returns_none(store):
    assert store.get_job("nope") is None


def test_set_job_state_round_trips(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    store.set_job_state("j1", "done")
    assert store.get_job("j1")["state"] == "done"


def test_set_job_state_rejects_unknown_state(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    with pytest.raises(ValueError, match="unknown state"):
        store.set_job_state("j1", "banana")


def test_bump_job_attempts_increments_and_returns(store):
    store.put_job("j1", "tier1", "v1", ["a"])
    assert store.bump_job_attempts("j1") == 1
    assert store.bump_job_attempts("j1") == 2
    assert store.get_job("j1")["attempts"] == 2


def test_open_jobs_excludes_settled_and_is_ordered(store):
    for jid, state in [("j3", "pending"), ("j1", "done"), ("j2", "running")]:
        store.put_job(jid, "tier1", "v1", ["a"])
        store.set_job_state(jid, state)
    assert [j["job_id"] for j in store.open_jobs()] == ["j2", "j3"]


def test_put_track_is_idempotent(store):
    assert store.put_track("vid1", 1, "p1", "[]", 0.0) is True
    assert store.put_track("vid1", 1, "p1", '["DIFFERENT"]', 0.0) is False
    assert store.get_track("vid1", 1)["content_json"] == "[]"


def test_latest_track_version_starts_at_zero_and_advances(store):
    assert store.latest_track_version("vid1") == 0
    store.put_track("vid1", 1, "p1", "[]", 0.0)
    store.put_track("vid1", 2, "p1", "[]", 0.0)
    assert store.latest_track_version("vid1") == 2
    assert store.latest_track_version("other") == 0
