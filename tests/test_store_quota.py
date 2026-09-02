"""The daily request quota (design §4.1).

This mirrors tests/test_store.py's spend-ledger tests deliberately. The
daily cap is the same problem as the USD ceiling in a different currency,
and it must not reintroduce the FIX ROUND 2 (C1) bug that a separate
check-then-record allows.
"""

import threading

import pytest

from dhvani.quota import QuotaExhausted
from dhvani.store import Store


def test_reserving_up_to_the_cap_succeeds_and_one_more_raises():
    with Store(":memory:") as s:
        for _ in range(3):
            s.reserve_quota("tier2", "2026-09-02", cap=3)

        assert s.quota_used("tier2", "2026-09-02") == 3
        with pytest.raises(QuotaExhausted):
            s.reserve_quota("tier2", "2026-09-02", cap=3)


def test_a_days_budget_is_independent_of_other_days():
    """The cap resets; it does not accumulate across days."""
    with Store(":memory:") as s:
        s.reserve_quota("tier2", "2026-09-02", cap=1)
        with pytest.raises(QuotaExhausted):
            s.reserve_quota("tier2", "2026-09-02", cap=1)

        s.reserve_quota("tier2", "2026-09-03", cap=1)  # tomorrow is fresh
        assert s.quota_used("tier2", "2026-09-02") == 1
        assert s.quota_used("tier2", "2026-09-03") == 1


def test_tiers_do_not_share_a_budget():
    with Store(":memory:") as s:
        s.reserve_quota("tier2", "2026-09-02", cap=1)
        s.reserve_quota("tier9", "2026-09-02", cap=1)
        assert s.quota_used("tier2", "2026-09-02") == 1


def test_reservations_survive_a_restart(tmp_path):
    """In-memory counting would silently refill the quota on every run."""
    db = str(tmp_path / "q.db")
    with Store(db) as s:
        s.reserve_quota("tier2", "2026-09-02", cap=2)
        s.reserve_quota("tier2", "2026-09-02", cap=2)

    with Store(db) as s:
        assert s.quota_used("tier2", "2026-09-02") == 2
        with pytest.raises(QuotaExhausted):
            s.reserve_quota("tier2", "2026-09-02", cap=2)


def test_landing_exactly_on_the_cap_is_allowed():
    with Store(":memory:") as s:
        s.reserve_quota("tier2", "2026-09-02", cap=5, n=5)
        assert s.quota_used("tier2", "2026-09-02") == 5


def test_a_refused_reservation_writes_nothing():
    with Store(":memory:") as s:
        s.reserve_quota("tier2", "2026-09-02", cap=1)
        with pytest.raises(QuotaExhausted):
            s.reserve_quota("tier2", "2026-09-02", cap=1, n=99)
        assert s.quota_used("tier2", "2026-09-02") == 1


def test_concurrent_reservations_never_exceed_the_cap(tmp_path):
    """N threads, N handles, one DB, cap of 1 -- exactly one may win.

    This is the FIX ROUND 2 (C1) scenario transposed to quota. Verified
    against a deliberately broken two-step implementation before being
    trusted: with SELECT-then-INSERT as separate statements this test
    reports several winners.
    """
    db = str(tmp_path / "shared.db")
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker():
        with Store(db) as s:
            barrier.wait()
            try:
                s.reserve_quota("tier2", "2026-09-02", cap=1)
                result = "ok"
            except QuotaExhausted:
                result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a worker thread deadlocked"

    assert len(outcomes) == n_threads
    assert outcomes.count("ok") == 1, f"expected 1 winner, got {outcomes.count('ok')}"
    with Store(db) as check:
        assert check.quota_used("tier2", "2026-09-02") == 1
