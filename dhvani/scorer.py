"""Deterministic risk scoring. No model artifact — weights come from config."""

from dataclasses import asdict, dataclass

from dhvani.config import RISK_WEIGHTS
from dhvani.signals import romanization_smell, script_mix_entropy

SHORT_SEGMENT_MS = 1500


@dataclass(frozen=True)
class Features:
    ctc_rnnt_disagreement: float
    mean_neg_logprob: float
    script_mix_entropy: float
    romanization_smell: float
    short_segment: float


def extract(text: str, decoder_signals: dict, duration_ms: int) -> Features:
    """Build a feature vector. Used identically at fit time and at inference
    time — importing this one function everywhere is what prevents skew."""
    return Features(
        ctc_rnnt_disagreement=float(decoder_signals.get("ctc_rnnt_disagreement", 0.0)),
        mean_neg_logprob=float(decoder_signals.get("mean_neg_logprob", 0.0)),
        script_mix_entropy=script_mix_entropy(text),
        romanization_smell=romanization_smell(text),
        short_segment=1.0 if duration_ms < SHORT_SEGMENT_MS else 0.0,
    )


def risk(f: Features) -> float:
    """Weighted sum of clamped features, in [0, 1]."""
    total = sum(
        RISK_WEIGHTS[name] * min(max(value, 0.0), 1.0)
        for name, value in asdict(f).items()
    )
    return min(max(total, 0.0), 1.0)
