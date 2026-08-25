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


def source_id(name: str, pcm: np.ndarray) -> str:
    """Identity for one audio source: a legible name plus a content digest.

    dhvani/cli.py used os.path.basename(audio) alone, so a/clip.wav and
    b/clip.wav were ONE source inside a single --db -- one track history and
    one job namespace. Escalating one could surface in the other's output,
    and reconcile() polled across both as if they were the same video.

    Shaped like variant_slug() and for the same reason: a sanitized,
    length-capped prefix so a human reading the segments or jobs table can
    tell which file a row came from, plus a digest so two sources that
    sanitize to the same text still differ. The digest covers the
    UNsanitized name as well as the audio, because sanitizing is lossy --
    "a b.wav" and "a/b.wav" collapse to the same safe text.

    Audio identity comes from segment_id(), so it is the same hash over the
    same normalized bytes that identifies every segment, and it inherits
    that function's rejection of float or multi-channel input. Two files
    whose audio decodes to identical PCM therefore agree here exactly when
    they already agree at the segment level.

    Only the basename is passed in, never the directory: re-running the
    same clip from a different working directory must find its own track
    rather than start a new one.

    Note what this does NOT do: renaming a file changes its source_id and
    so starts a fresh track history. That costs a stale row, not money --
    hypotheses are keyed by (segment_id, tier, variant), not by source, so
    escalate() still sees the old results as already paid for and will not
    re-buy them.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "source"
    digest = hashlib.sha256(
        segment_id(pcm).encode("utf-8") + b"\x00" + name.encode("utf-8")
    ).hexdigest()[:12]
    return f"{safe[:48]}-{digest}"


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
