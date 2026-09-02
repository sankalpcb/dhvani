"""Rate limiting and daily quota (design §4).

No test here sleeps. TokenBucket takes its clock as a parameter, so a
"one minute later" assertion is a variable assignment rather than a
minute of wall time -- goal G5 of the Tier 2 design.
"""

import pytest

from dhvani.quota import TokenBucket


class FakeClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_a_fresh_bucket_allows_exactly_its_capacity():
    clock = FakeClock()
    bucket = TokenBucket(capacity=3, refill_per_sec=1.0, now=clock)

    assert [bucket.take() for _ in range(3)] == [True, True, True]
    assert bucket.take() is False


def test_tokens_refill_at_the_stated_rate():
    clock = FakeClock()
    bucket = TokenBucket(capacity=3, refill_per_sec=2.0, now=clock)
    for _ in range(3):
        bucket.take()
    assert bucket.take() is False

    clock.advance(0.5)  # 0.5s at 2/s == exactly one token

    assert bucket.take() is True
    assert bucket.take() is False


def test_idle_time_does_not_accumulate_past_capacity():
    """An hour idle must not buy an hour's worth of burst."""
    clock = FakeClock()
    bucket = TokenBucket(capacity=2, refill_per_sec=1.0, now=clock)
    clock.advance(3600)

    assert [bucket.take() for _ in range(3)] == [True, True, False]


# --- QuotaGate: the two limits composed (design §4) ---

from dhvani.quota import QuotaExhausted, QuotaGate, RateLimited  # noqa: E402
from dhvani.store import Store  # noqa: E402


def _gate(store, cap=2, capacity=2, clock=None, day="2026-09-02"):
    clock = clock or FakeClock()
    return QuotaGate(
        store=store, tier="tier2", cap=cap,
        bucket=TokenBucket(capacity=capacity, refill_per_sec=1.0, now=clock),
        today=lambda: day,
    ), clock


def test_acquire_consumes_daily_quota():
    with Store(":memory:") as s:
        gate, _ = _gate(s)
        gate.acquire()
        assert s.quota_used("tier2", "2026-09-02") == 1


def test_the_daily_cap_is_hard():
    with Store(":memory:") as s:
        gate, _ = _gate(s, cap=1, capacity=99)
        gate.acquire()
        with pytest.raises(QuotaExhausted):
            gate.acquire()


def test_rate_limiting_does_not_burn_daily_quota():
    """The whole reason pacing is checked BEFORE the daily reservation.

    A 429 consumes nothing at the vendor, so it must consume nothing in
    the ledger either -- otherwise going too fast would permanently
    destroy quota for calls that were never made.
    """
    with Store(":memory:") as s:
        gate, _ = _gate(s, cap=100, capacity=1)
        gate.acquire()

        with pytest.raises(RateLimited):
            gate.acquire()

        assert s.quota_used("tier2", "2026-09-02") == 1, "pacing burned daily quota"


def test_pacing_recovers_but_the_daily_cap_does_not():
    with Store(":memory:") as s:
        gate, clock = _gate(s, cap=2, capacity=1)
        gate.acquire()
        clock.advance(10)
        gate.acquire()

        clock.advance(10)  # tokens available again
        with pytest.raises(QuotaExhausted):
            gate.acquire()  # ...but the day's budget is gone
