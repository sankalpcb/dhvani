import threading

import pytest
from dhvani.config import MAX_SPEND_USD
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


# --- Fix round 2, I2/I3: the cache key carries variant and POLICY_ID ---

def test_two_backend_variants_do_not_share_a_cache_entry(store):
    """Demonstrated defect: a run with lang="hi" followed by a run with
    lang="ml" returned the Hindi transcript from cache, making --lang
    silently a no-op. segment_id hashes PCM alone, so the variant has to
    be part of the hypothesis key."""
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    store.put_hypothesis("abc", "tier0", "hindi text", {}, 0.0,
                         variant_key="lang=hi")

    assert store.get_hypothesis("abc", "tier0", variant_key="lang=ml") is None
    assert store.get_hypothesis("abc", "tier0",
                                variant_key="lang=hi")["text"] == "hindi text"


def test_both_variants_coexist_and_stay_idempotent(store):
    """The variant is folded into the tier column, so PRIMARY KEY
    (segment_id, tier) still enforces idempotency per variant -- the key
    is not weakened, only made more specific."""
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    assert store.put_hypothesis("abc", "tier0", "hindi", {}, 0.0,
                                variant_key="lang=hi") is True
    assert store.put_hypothesis("abc", "tier0", "malayalam", {}, 0.0,
                                variant_key="lang=ml") is True
    assert store.put_hypothesis("abc", "tier0", "DIFFERENT", {}, 0.0,
                                variant_key="lang=hi") is False

    assert store.get_hypothesis("abc", "tier0", variant_key="lang=hi")["text"] == "hindi"
    assert store.get_hypothesis("abc", "tier0", variant_key="lang=ml")["text"] == "malayalam"


def test_bumping_policy_id_invalidates_cached_hypotheses(store, monkeypatch):
    """POLICY_ID's own docstring, spec §3.1 and invariant I5 all describe
    it as the cache-invalidation mechanism. Before this fix it had zero
    call sites, so bumping it invalidated nothing."""
    store.put_segment("abc", "vid1", 0, 3000, "hi")
    store.put_hypothesis("abc", "tier0", "stale", {}, 0.0, variant_key="lang=hi")
    assert store.get_hypothesis("abc", "tier0", variant_key="lang=hi") is not None

    monkeypatch.setattr("dhvani.config.POLICY_ID", "p-bumped")
    assert store.get_hypothesis("abc", "tier0", variant_key="lang=hi") is None


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


# --- Fix round 2, C1: the ceiling must survive concurrent reservations ---

def test_two_step_check_then_record_is_not_atomic(tmp_path):
    """Characterizes the C1 defect that reserve_spend() exists to close.

    check_budget() and record_spend() are two separate autocommitted
    transactions. Two Store handles on the same DB file can both read the
    same stale total, both conclude there is room, and both then record --
    so the ledger ends up OVER the ceiling with neither call having
    raised. This is not hypothetical: dhvani.cli's --db defaults to a
    fixed shared path, so two concurrent runs share exactly this file.

    This test deliberately pins the broken behavior of the two-step
    sequence rather than asserting it is safe: it is the reason callers
    that spend money must use reserve_spend() instead. If someone later
    makes the two-step path atomic too, this test failing is the correct
    signal to delete it.
    """
    db = str(tmp_path / "shared.db")
    with Store(db) as a, Store(db) as b:
        a.record_spend("tier1", MAX_SPEND_USD - 0.40)  # ledger at 19.60

        # Interleaved exactly as dhvani.backends.base used to call it.
        a.check_budget(0.30)   # sees 19.60, projects 19.90 -- passes
        b.check_budget(0.30)   # sees 19.60 too (stale) -- also passes
        a.record_spend("tier1", 0.30)
        b.record_spend("tier1", 0.30)

        assert a.total_spend() > MAX_SPEND_USD, (
            "expected the non-atomic two-step path to breach the ceiling"
        )


def test_interleaved_reserve_spend_cannot_breach_the_ceiling(tmp_path):
    """The same interleaving as above, but through reserve_spend().

    Two Store handles on one DB file, the ledger at 19.60, each
    reservation costing 0.30: the first fits (19.90), the second does not
    (20.20 > 20.00) and must raise. The ledger must never end above the
    ceiling.
    """
    db = str(tmp_path / "shared.db")
    with Store(db) as a, Store(db) as b:
        a.record_spend("tier1", MAX_SPEND_USD - 0.40)

        a.reserve_spend("tier1", 0.30)
        with pytest.raises(BudgetExceeded):
            b.reserve_spend("tier1", 0.30)

        assert a.total_spend() <= MAX_SPEND_USD
        assert b.total_spend() == pytest.approx(MAX_SPEND_USD - 0.10)


def test_concurrent_reserve_spend_never_breaches_the_ceiling(tmp_path):
    """Real concurrency: N threads, N independent Store handles, one DB.

    Every thread waits on a barrier so the reservations are issued as
    close to simultaneously as the runtime allows, which is precisely the
    window the old check-then-record sequence lost money in. The ledger
    starts at 19.60 with a 20.00 ceiling and each reservation costs 0.30,
    so at most ONE thread can legally win. Whatever the interleaving, the
    final total must never exceed MAX_SPEND_USD and the number of
    successful reservations must equal the number of rows actually
    written.
    """
    db = str(tmp_path / "shared.db")
    n_threads = 8
    cost = 0.30

    with Store(db) as seed:
        seed.record_spend("tier1", MAX_SPEND_USD - 0.40)  # ledger at 19.60

    barrier = threading.Barrier(n_threads)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker():
        # Each thread owns its own connection: sqlite3 objects are not
        # shareable across threads, and separate handles are exactly the
        # scenario C1 describes.
        with Store(db) as s:
            barrier.wait()
            try:
                s.reserve_spend("tier1", cost)
                result = "ok"
            except BudgetExceeded:
                result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a worker thread deadlocked"

    granted = outcomes.count("ok")
    assert len(outcomes) == n_threads
    assert granted == 1, f"exactly one reservation may fit, got {granted}"

    with Store(db) as check:
        total = check.total_spend()
    assert total <= MAX_SPEND_USD, f"ledger breached the ceiling: {total}"
    assert total == pytest.approx(MAX_SPEND_USD - 0.40 + granted * cost)


def test_reserve_spend_allows_landing_exactly_on_the_ceiling(tmp_path):
    """projected == MAX_SPEND_USD is allowed; only > is refused.

    Same boundary as check_budget's strict '>' comparison, pinned so the
    atomic path cannot silently drift to '>=' and start refusing a spend
    that exactly exhausts the budget.
    """
    with Store(str(tmp_path / "t.db")) as s:
        s.record_spend("tier1", MAX_SPEND_USD - 0.5)  # 19.5, exact in binary
        s.reserve_spend("tier1", 0.5)                 # 19.5 + 0.5 == 20.0
        assert s.total_spend() == pytest.approx(MAX_SPEND_USD)


def test_reserve_spend_refuses_a_hair_over_the_ceiling(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        s.record_spend("tier1", MAX_SPEND_USD - 0.5)
        with pytest.raises(BudgetExceeded, match="would exceed"):
            s.reserve_spend("tier1", 0.5000001)
        assert s.total_spend() == pytest.approx(MAX_SPEND_USD - 0.5)


def test_reserve_spend_records_nothing_when_it_refuses(tmp_path):
    """A refused reservation must leave the ledger byte-identical."""
    with Store(str(tmp_path / "t.db")) as s:
        s.record_spend("tier1", MAX_SPEND_USD)
        before = s.conn.execute("SELECT COUNT(*) AS n FROM spend").fetchone()["n"]
        with pytest.raises(BudgetExceeded):
            s.reserve_spend("tier1", 0.01)
        after = s.conn.execute("SELECT COUNT(*) AS n FROM spend").fetchone()["n"]
        assert after == before
