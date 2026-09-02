"""Rate limiting and daily quota for the free-tier Tier 2 backend.

Design §4: the free tier imposes two limits with DIFFERENT failure
semantics, and they get different mechanisms because of it.

  daily cap      a consumable. Over-running it is unrecoverable until
                 midnight, so it fails CLOSED, persisted, through one
                 atomic statement (Store.reserve_quota).

  per-minute     pacing. Exceeding it returns a retryable 429 and
                 consumes nothing, so it fails SOFT, in process, here.

TokenBucket deliberately holds no persistent state. Losing its tokens on
restart is correct: a process that just started genuinely has not sent
anything recently. The daily counter must NOT be modelled this way, which
is why it lives in the store instead.
"""

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


class QuotaExhausted(RuntimeError):
    """The daily free-tier request budget for a tier is spent.

    Not an error in the usual sense: design §3 makes exhaustion a normal,
    expected condition that degrades gracefully. Callers mark the segment
    and carry on rather than aborting the run.
    """


class TokenBucket:
    """Classic token bucket with an injectable clock.

    `now` is a zero-argument callable returning monotonic seconds. Tests
    pass a fake so that rate limiting can be asserted at its exact
    boundary without sleeping.
    """

    def __init__(self, capacity: int, refill_per_sec: float, now=time.monotonic):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if refill_per_sec <= 0:
            raise ValueError(f"refill_per_sec must be positive, got {refill_per_sec}")
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self._now = now
        self._tokens = float(capacity)
        self._last = now()

    def take(self, n: int = 1) -> bool:
        """Consume n tokens. Returns False rather than blocking or raising."""
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    def _refill(self) -> None:
        now = self._now()
        elapsed = now - self._last
        self._last = now
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)


class RateLimited(RuntimeError):
    """Too many requests too fast. Soft: nothing was consumed, retry later."""


# Google resets requests-per-day quotas at midnight PACIFIC, and applies
# limits per project rather than per API key.
# https://ai.google.dev/gemini-api/docs/rate-limits  (checked 2026-09-02)
QUOTA_TZ = ZoneInfo("America/Los_Angeles")

# Conservative pacing figure, not a measured ceiling -- the vendor does
# not publish free-tier RPM. The bucket smooths bursts; the daily cap is
# what actually fails closed.
GEMINI_RPM_DEFAULT = 15
"""MEASURED 2026-09-02, not assumed. A live run returned:

    Quota exceeded for metric:
    generativelanguage.googleapis.com/generate_content_free_tier_requests,
    limit: 15, model: gemini-3.5-flash-lite

so the free tier allows 15 requests per MINUTE per project per model. This
was previously flagged unpublished and guessed at; the guess happened to be
right, and is now a measurement.
"""

PACING_SAFETY = 0.8
"""Issue at 80% of the vendor's stated rate.

Landing exactly on a documented limit is how you discover the vendor counts
its window differently than you do -- a rolling minute against a fixed one,
or a clock that is not yours. 20% is cheap insurance on a free tier where
the only cost of going slower is waiting.
"""


def paced_bucket(rpm: int = GEMINI_RPM_DEFAULT, safety: float = PACING_SAFETY,
                 now=None) -> "TokenBucket":
    """A bucket that paces against a FIXED-WINDOW vendor quota.

    Capacity 1 -- no burst -- which is the whole point, and was learned by
    getting it wrong. TokenBucket(capacity=rpm, refill=rpm/60) starts FULL,
    so it fires `rpm` requests in seconds; a per-minute quota is then spent
    before the first minute has elapsed and everything after 429s. A live
    calibration pass died at segment 30 of 100 exactly this way.

    A token bucket's capacity IS its burst allowance. Against a limit
    expressed per fixed window, the only safe burst is one.
    """
    kwargs = {"now": now} if now is not None else {}
    return TokenBucket(capacity=1, refill_per_sec=(rpm * safety) / 60.0, **kwargs)


def quota_day(now=None) -> str:
    """The quota day key, on Google's reset boundary.

    SETTLED by the spike of 2026-09-02, which overturned this function.
    It keyed by UTC, on the documented reasoning that a wrong boundary
    "can never over-spend -- a stale day key only makes the gate more
    conservative". That was exactly backwards, and worth spelling out
    because the same mistake is easy to repeat:

    UTC midnight is 16:00-17:00 Pacific the PREVIOUS day, so a UTC key
    rolls over roughly eight hours EARLY. In that window the gate resets a
    counter Google has not, and happily authorizes a second full day's
    worth of requests inside one Google day -- up to 2x the cap. The error
    was in the over-spending direction, which is the direction this
    project's whole reservation discipline exists to prevent.

    ZoneInfo rather than a fixed offset: Pacific is UTC-7 in summer and
    UTC-8 in winter, so a hardcoded offset would be wrong for half the
    year and would drift the boundary by an hour rather than fixing it.

    `now` is a zero-argument callable returning an aware datetime, so the
    boundary is testable without waiting for midnight.
    """
    at = now() if now is not None else datetime.now(timezone.utc)
    return at.astimezone(QUOTA_TZ).strftime("%Y-%m-%d")


class QuotaGate:
    """Authorize one free-tier call, or refuse it.

    Composes the two limits of design §4, and the ORDER is the point.
    Pacing is checked first because it is free and leaves no trace; the
    daily reservation is taken second because it is persistent and
    unrecoverable. Reserving first and then discovering we are going too
    fast would burn a day's quota on a call that was never made.
    """

    def __init__(self, store, tier: str, cap: int, bucket: TokenBucket, today=quota_day):
        self.store = store
        self.tier = tier
        self.cap = cap
        self.bucket = bucket
        self._today = today

    def acquire(self, n: int = 1) -> None:
        """Claim n requests. Raises RateLimited (soft) or QuotaExhausted (hard)."""
        if not self.bucket.take(n):
            raise RateLimited(
                f"{self.tier} exceeded its per-second pacing budget; retry shortly"
            )
        self.store.reserve_quota(self.tier, self._today(), self.cap, n)

    def remaining(self) -> int:
        """Requests left in today's budget. Never negative."""
        return max(0, self.cap - self.store.quota_used(self.tier, self._today()))


def unpaced_bucket() -> "TokenBucket":
    """A bucket that never refuses, for modes that call no vendor.

    Replay reads fixtures; there is no rate to limit and no quota to
    protect. Expressed as a bucket rather than as a None special-case so
    QuotaGate keeps exactly one code path and the daily reservation still
    happens -- a replay run should still be counted, just never delayed.
    """
    return TokenBucket(capacity=1_000_000, refill_per_sec=1_000_000.0)
