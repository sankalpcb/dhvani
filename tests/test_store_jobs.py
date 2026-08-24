import pytest
from dhvani.store import Store


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_job_is_idempotent(store):
    assert store.put_job("j1", "tier1", "v1", ["a", "b"], "vid1") is True
    assert store.put_job("j1", "tier1", "v1", ["DIFFERENT"], "vid1") is False
    assert store.get_job("j1")["segment_ids"] == ["a", "b"]


def test_new_job_starts_pending_with_zero_attempts(store):
    store.put_job("j1", "tier1", "v1", ["a"], "vid1")
    job = store.get_job("j1")
    assert job["state"] == "pending"
    assert job["attempts"] == 0


def test_get_missing_job_returns_none(store):
    assert store.get_job("nope") is None


def test_set_job_state_round_trips(store):
    store.put_job("j1", "tier1", "v1", ["a"], "vid1")
    store.set_job_state("j1", "done")
    assert store.get_job("j1")["state"] == "done"


def test_set_job_state_rejects_unknown_state(store):
    store.put_job("j1", "tier1", "v1", ["a"], "vid1")
    with pytest.raises(ValueError, match="unknown state"):
        store.set_job_state("j1", "banana")


def test_bump_job_attempts_increments_and_returns(store):
    store.put_job("j1", "tier1", "v1", ["a"], "vid1")
    assert store.bump_job_attempts("j1") == 1
    assert store.bump_job_attempts("j1") == 2
    assert store.get_job("j1")["attempts"] == 2


def test_open_jobs_excludes_settled_and_is_ordered(store):
    for jid, state in [("j3", "pending"), ("j1", "done"), ("j2", "running")]:
        store.put_job(jid, "tier1", "v1", ["a"], "vid1")
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


# --- C1: jobs belong to a source; open_jobs() must be able to say which ---

def test_put_job_records_the_source_id(store):
    store.put_job("j1", "tier1", "v1", ["a"], "vid1")
    assert store.get_job("j1")["source_id"] == "vid1"


def test_open_jobs_filters_to_one_source_and_stays_ordered(store):
    """C1: reconcile() polls and settles whatever open_jobs() hands it, so
    a job for another video showing up here is what let one source settle
    -- and permanently lose -- another source's paid results."""
    store.put_job("j3", "tier1", "v1", ["a"], "vid1")
    store.put_job("j1", "tier1", "v1", ["b"], "vid2")
    store.put_job("j2", "tier1", "v1", ["c"], "vid1")

    assert [j["job_id"] for j in store.open_jobs("vid1")] == ["j2", "j3"]
    assert [j["job_id"] for j in store.open_jobs("vid2")] == ["j1"]
    # No source_id given: every open job, ordered by job_id, as before.
    assert [j["job_id"] for j in store.open_jobs()] == ["j1", "j2", "j3"]


def test_open_jobs_for_an_unknown_source_is_empty(store):
    store.put_job("j1", "tier1", "v1", ["a"], "vid1")
    assert store.open_jobs("nope") == []
