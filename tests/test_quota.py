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


# --- the quota day boundary (spike, 2026-09-02) ---

from datetime import datetime, timezone  # noqa: E402

from dhvani.quota import quota_day  # noqa: E402


def _at(iso):
    return lambda: datetime.fromisoformat(iso)


def test_the_quota_day_follows_pacific_not_utc():
    """Google resets RPD at midnight PACIFIC (ai.google.dev/gemini-api/docs/
    rate-limits). Keying by UTC rolls our counter over ~8 hours early, so in
    that window we reset a budget Google has not -- letting up to TWICE the
    cap through in one Google day. The design's claim that a wrong boundary
    could only ever be conservative was exactly backwards.
    """
    # 01:30 UTC on the 2nd is still 18:30 Pacific on the 1st.
    assert quota_day(_at("2026-09-02T01:30:00+00:00")) == "2026-09-01"


def test_the_day_rolls_over_at_pacific_midnight():
    # 06:59 UTC = 23:59 PDT on the 1st; 07:00 UTC = 00:00 PDT on the 2nd.
    assert quota_day(_at("2026-09-02T06:59:00+00:00")) == "2026-09-01"
    assert quota_day(_at("2026-09-02T07:00:00+00:00")) == "2026-09-02"


def test_the_boundary_tracks_daylight_saving():
    """Pacific is UTC-7 in summer and UTC-8 in winter, so a fixed offset
    would drift by an hour for half the year."""
    assert quota_day(_at("2026-01-15T07:30:00+00:00")) == "2026-01-14"  # PST
    assert quota_day(_at("2026-07-15T07:30:00+00:00")) == "2026-07-15"  # PDT
