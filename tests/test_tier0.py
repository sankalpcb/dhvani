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


def test_construction_with_no_model_succeeds_without_torch():
    """G5 regression guard. A stranger who clones the repo and runs the
    offline replay workflow has no torch/transformers installed and no HF
    token. Constructing Tier0Conformer() with no injected model must not
    import anything or touch the network — only an actual transcribe()
    call is allowed to do that. This project's own venv genuinely has
    neither torch nor transformers installed (see pyproject.toml's optional
    `models` extra), so this assertion not raising is real evidence, not a
    simulated one."""
    b = Tier0Conformer()
    assert b.lang == "hi"


def test_construction_never_calls_load(monkeypatch):
    """Stronger, environment-independent version of the guard above: force
    _load to blow up if it is ever invoked, and prove __init__ alone never
    reaches it. This stays a valid regression test even in an environment
    where the models extra happens to be installed."""
    import dhvani.backends.tier0_conformer as mod

    def _boom(model_id):
        raise AssertionError("__init__ must not call _load()")

    monkeypatch.setattr(mod, "_load", _boom)
    Tier0Conformer()  # must not raise


def test_lazy_load_runs_exactly_once_on_first_transcribe(monkeypatch):
    """The deferred _load() must still fire (and be cached) once an actual
    transcribe() happens on an uninjected instance — laziness must not
    silently turn into never-loads-at-all."""
    import dhvani.backends.tier0_conformer as mod

    calls = []

    def _fake_load(model_id):
        calls.append(model_id)
        return StubModel("ctc out", "rnnt out")

    monkeypatch.setattr(mod, "_load", _fake_load)
    b = Tier0Conformer(lang="hi")
    assert calls == [], "construction must not have loaded anything yet"

    b.transcribe(_seg())
    assert calls == [mod.MODEL_ID]

    b.transcribe(_seg())
    assert calls == [mod.MODEL_ID], "second call must reuse the cached model"
