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


def utc_today() -> str:
    """The quota day key, as a UTC date string.

    UNVERIFIED (design §8): Google's reset boundary is assumed to be UTC
    midnight and has not been confirmed against the vendor. Getting this
    wrong shifts the reset by hours and can never over-spend -- a stale
    day key only makes the gate more conservative -- so it is a safe
    assumption to ship while the spike is outstanding.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class QuotaGate:
    """Authorize one free-tier call, or refuse it.

    Composes the two limits of design §4, and the ORDER is the point.
    Pacing is checked first because it is free and leaves no trace; the
    daily reservation is taken second because it is persistent and
    unrecoverable. Reserving first and then discovering we are going too
    fast would burn a day's quota on a call that was never made.
    """

    def __init__(self, store, tier: str, cap: int, bucket: TokenBucket, today=utc_today):
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
