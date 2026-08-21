"""Content-addressed segment identity."""

import hashlib

import numpy as np


def segment_id(pcm: np.ndarray) -> str:
    """SHA256 of normalized PCM bytes.

    pcm must be mono int16 at config.SAMPLE_RATE — see audio.normalize().
    """
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got {pcm.dtype}")
    if pcm.ndim != 1:
        raise ValueError(f"expected mono 1-D array, got shape {pcm.shape}")
    return hashlib.sha256(pcm.tobytes()).hexdigest()
