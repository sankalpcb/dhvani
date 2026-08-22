"""Tier 0: local IndicConformer. Runs on 100% of segments at zero marginal cost.

The model is hybrid CTC-RNNT: two decoders over one shared encoder. Disagreement
between the heads is ensemble uncertainty for free, and is expected to be the
strongest single risk signal.

Day-one spike (spec §14): could not confirm whether both heads are actually
exposed by the real model — see scripts/spike_conformer.py and the Task 9
commit message for the blocked-on-environment result (the model repo is
gated on HuggingFace and access was not available in this environment). The
implementation below assumes the interface documented in the brief
(`model(wav, lang, decoding)` returning text for `decoding in ("ctc",
"rnnt")`); this must be re-verified before Tier 0 is trusted against the
real weights.
"""

import numpy as np

from dhvani.evaluator import plain_wer

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


def disagreement(ctc_text: str, rnnt_text: str) -> float:
    """Normalized edit distance between the two decoder heads, in [0, 1]."""
    if not ctc_text.strip() and not rnnt_text.strip():
        return 0.0
    return min(plain_wer(rnnt_text, ctc_text), 1.0)


def _load(model_id: str):
    import torch  # noqa: F401  (imported for side effects / availability check)
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_id, trust_remote_code=True)


class Tier0Conformer:
    name = "tier0"

    def __init__(self, model=None, lang: str = "hi", model_id: str = MODEL_ID):
        """Stores the injected model (or None) and the model_id. Does NOT
        call _load() — constructing a Tier0Conformer must import nothing
        and touch no network, so replay-mode callers (e.g. dhvani.cli) can
        build one with zero ML dependencies installed. The real, gated
        model is loaded lazily on first use via the `_model` property.
        """
        self._injected_model = model
        self._loaded_model = None
        self.lang = lang
        self.model_id = model_id

    @property
    def variant_key(self) -> str:
        """Everything about this backend that changes the output for
        byte-identical PCM. lang and model_id both do, so a hypothesis
        cached (or a fixture recorded) under one must never be served for
        the other — see FIX ROUND 2 (I2/I3) in backends/base.py."""
        return f"lang={self.lang};model_id={self.model_id}"

    @property
    def _model(self):
        """The model to call, loading and caching it on first access.

        An injected model (tests, or any caller that already has one) is
        used as-is. Otherwise _load(self.model_id) runs exactly once, on
        the first transcribe() call — never at construction time.
        """
        if self._injected_model is not None:
            return self._injected_model
        if self._loaded_model is None:
            self._loaded_model = _load(self.model_id)
        return self._loaded_model

    def cost_per_call(self, segment) -> float:
        return 0.0  # local inference

    def transcribe(self, segment) -> dict:
        wav = self._to_float(segment.pcm)
        ctc_text = str(self._model(wav, self.lang, "ctc"))
        rnnt_text = str(self._model(wav, self.lang, "rnnt"))
        return {
            "text": rnnt_text,
            "signals": {
                "ctc_rnnt_disagreement": disagreement(ctc_text, rnnt_text),
                "mean_neg_logprob": 0.0,  # not exposed by this model; see spec §14
            },
        }

    @staticmethod
    def _to_float(pcm: np.ndarray):
        float_pcm = pcm.astype(np.float32) / 32768.0
        try:
            import torch
        except ImportError:
            # torch lives in the optional `models` extra. A stubbed model
            # (as used by the whole test suite) never inspects this value's
            # type, so fall back to a plain ndarray rather than forcing
            # torch onto every environment that only exercises the stub
            # path. The real model, loaded via _load(), already requires
            # torch to import at all, so this branch is unreachable once a
            # real model is in play.
            return float_pcm[np.newaxis, :]
        return torch.from_numpy(float_pcm).unsqueeze(0)
