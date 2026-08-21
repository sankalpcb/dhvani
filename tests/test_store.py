import pytest
from dhvani.store import Store, BudgetExceeded


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def test_put_hypothesis_is_idempotent(store):
    """Invariant I2: applying the same result twice is a no-op."""
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    first = store.put_hypothesis("abc", "tier0", "hello", {"x": 1}, 0.0)
    second = store.put_hypothesis("abc", "tier0", "DIFFERENT", {"x": 2}, 0.0)
    assert first is True
    assert second is False
    assert store.get_hypothesis("abc", "tier0")["text"] == "hello"


def test_get_missing_hypothesis_returns_none(store):
    assert store.get_hypothesis("nope", "tier0") is None


def test_signals_round_trip(store):
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    store.put_hypothesis("abc", "tier0", "hello", {"entropy": 0.5}, 0.0)
    assert store.get_hypothesis("abc", "tier0")["signals"] == {"entropy": 0.5}


def test_spend_accumulates(store):
    store.record_spend("tier1", 1.5)
    store.record_spend("tier1", 2.25)
    assert store.total_spend() == pytest.approx(3.75)


def test_check_budget_allows_under_ceiling(store):
    store.record_spend("tier1", 1.0)
    store.check_budget(0.5)  # must not raise


def test_check_budget_fails_closed_at_ceiling(store):
    """Invariant I4: total spend never exceeds the configured budget."""
    store.record_spend("tier1", 19.9)
    with pytest.raises(BudgetExceeded, match="would exceed"):
        store.check_budget(0.5)
