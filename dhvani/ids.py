"""Content-addressed segment identity, and the cache identity built on it.

segment_id() hashes PCM alone, which is correct: it identifies the audio.
It does NOT identify a *hypothesis about* that audio — the same bytes
decoded with a different lang, a different model_id, or under a different
POLICY_ID produce different text. hypothesis_key() and variant_slug()
supply that second half of the identity, for the SQLite cache and for
fixture paths respectively.
"""

import hashlib
import re

import numpy as np

from dhvani import config


def segment_id(pcm: np.ndarray) -> str:
    """SHA256 of normalized PCM bytes.

    pcm must be mono int16 at config.SAMPLE_RATE — see audio.normalize().
    """
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm.dtype}")
    if pcm.ndim != 1:
        raise ValueError(f"expected mono 1-D array, got shape {pcm.shape}")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def _policy_variant(variant_key: str) -> str:
    """POLICY_ID + backend variant: everything besides the audio and the
    tier that determines what a hypothesis says.

    config.POLICY_ID is read at call time, not bound at import time, so
    bumping it takes effect immediately (and so tests can monkeypatch it).
    """
    return f"{config.POLICY_ID}|{variant_key}"


def hypothesis_key(tier: str, variant_key: str = "") -> str:
    """Cache key for the store's `tier` column.

    Fix round 2 (I2/I3): the store keyed hypotheses on (segment_id, tier)
    only, so a run with lang="hi" followed by one with lang="ml" served
    the Hindi transcript from cache — --lang was silently a no-op — and
    POLICY_ID, documented as the cache-invalidation mechanism, had zero
    call sites and invalidated nothing.

    The composite goes into the same `tier` column, so PRIMARY KEY
    (segment_id, tier) keeps enforcing idempotency; the key is made more
    specific, never weakened.
    """
    return f"{tier}:{_policy_variant(variant_key)}"


def variant_slug(variant_key: str = "") -> str:
    """A single filesystem-safe path component for a backend variant.

    Fixture paths need the same identity as the cache key, but variant
    strings carry model ids containing '/' and other separators. The slug
    is a sanitized, length-capped prefix (so a human can tell fixture
    directories apart) plus a digest of the UNsanitized string (so two
    variants that sanitize to the same text still get different
    directories). Deterministic: the same variant and POLICY_ID always
    produce the same slug.
    """
    full = _policy_variant(variant_key)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", full).strip("-.") or "default"
    digest = hashlib.sha256(full.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:48]}-{digest}"
