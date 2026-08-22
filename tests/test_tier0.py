import numpy as np
import pytest

from dhvani.backends.base import Backend
from dhvani.backends.tier0_conformer import Tier0Conformer, disagreement
from dhvani.segmenter import Segment


def _seg():
    return Segment(segment_id="a" * 64, t_start_ms=0, t_end_ms=3000,
                   pcm=np.zeros(16000, dtype=np.int16))


class StubModel:
    def __init__(self, ctc_text, rnnt_text):
        self.ctc_text, self.rnnt_text = ctc_text, rnnt_text

    def __call__(self, wav, lang, decoding):
        return self.ctc_text if decoding == "ctc" else self.rnnt_text


def test_satisfies_backend_protocol():
    assert isinstance(Tier0Conformer(model=StubModel("a", "a"), lang="hi"), Backend)


def test_tier0_is_free():
    assert Tier0Conformer(model=StubModel("a", "a"), lang="hi").cost_per_call(_seg()) == 0.0


def test_returns_rnnt_text_as_primary_hypothesis():
    b = Tier0Conformer(model=StubModel("ctc out", "rnnt out"), lang="hi")
    assert b.transcribe(_seg())["text"] == "rnnt out"


def test_agreeing_heads_give_zero_disagreement():
    b = Tier0Conformer(model=StubModel("same words here", "same words here"), lang="hi")
    assert b.transcribe(_seg())["signals"]["ctc_rnnt_disagreement"] == 0.0


def test_disagreeing_heads_give_positive_disagreement():
    b = Tier0Conformer(model=StubModel("alpha beta gamma", "alpha beta delta"), lang="hi")
    assert b.transcribe(_seg())["signals"]["ctc_rnnt_disagreement"] > 0.0


def test_disagreement_is_bounded():
    assert disagreement("a b c", "x y z") <= 1.0
    assert disagreement("", "") == 0.0
