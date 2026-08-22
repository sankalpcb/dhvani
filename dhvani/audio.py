"""Deterministic audio normalization. Any change here is cache-invalidating."""

import numpy as np
import scipy.signal

from dhvani.config import SAMPLE_RATE


def normalize(samples: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample float audio in [-1, 1] to mono int16 at SAMPLE_RATE.

    Deterministic: same input always produces byte-identical output.
    """
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError(f"expected 1-D or 2-D array, got shape {samples.shape}")

    samples = samples.astype(np.float64)

    if src_rate != SAMPLE_RATE:
        n_out = int(round(len(samples) * SAMPLE_RATE / src_rate))
        samples = scipy.signal.resample(samples, n_out)

    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).round().astype(np.int16)
