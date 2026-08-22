import numpy as np
import pytest

from dhvani.backends.base import Backend
from dhvani.backends.tier1_chirp import (
    Tier1Chirp, USD_PER_MIN_DYNAMIC_BATCH, USD_PER_MIN_STANDARD,
)
from dhvani.segmenter import Segment


def _seg(ms=60000):
    return Segment(segment_id="b" * 64, t_start_ms=0, t_end_ms=ms,
                   pcm=np.zeros(16000, dtype=np.int16))


class StubClient:
    def __init__(self, text="chirp output"):
        self.text = text
        self.calls = 0

    def recognize_pcm(self, pcm, lang):
        self.calls += 1
        return self.text


def test_satisfies_backend_protocol():
    assert isinstance(Tier1Chirp(client=StubClient()), Backend)


def test_one_minute_costs_the_dynamic_batch_rate():
    assert Tier1Chirp(client=StubClient()).cost_per_call(_seg(60000)) == \
        pytest.approx(USD_PER_MIN_DYNAMIC_BATCH)


def test_cost_scales_with_duration():
    b = Tier1Chirp(client=StubClient())
    assert b.cost_per_call(_seg(30000)) == pytest.approx(USD_PER_MIN_DYNAMIC_BATCH / 2)


def test_standard_rate_is_the_documented_fallback():
    assert USD_PER_MIN_STANDARD == pytest.approx(0.016)
    assert USD_PER_MIN_STANDARD > USD_PER_MIN_DYNAMIC_BATCH


def test_transcribe_returns_text_and_empty_signals():
    out = Tier1Chirp(client=StubClient("नमस्ते")).transcribe(_seg())
    assert out["text"] == "नमस्ते"
    assert out["signals"] == {}


def test_transcribe_calls_the_client_once():
    client = StubClient()
    Tier1Chirp(client=client).transcribe(_seg())
    assert client.calls == 1


def test_construction_with_no_client_succeeds_without_google_cloud_speech():
    """G5 regression guard, mirroring Task 9's Tier0Conformer guard. A
    stranger who clones the repo and runs the offline replay workflow has
    no google-cloud-speech installed and no GCP credentials. Constructing
    Tier1Chirp() with no injected client must not import anything or touch
    the network -- only an actual transcribe() call is allowed to do that.
    This project's own venv genuinely has no google-cloud-speech installed
    (it lives behind the optional `cloud` extra), so this assertion not
    raising is real evidence, not a simulated one."""
    b = Tier1Chirp()
    assert b.lang == "hi-IN"


def test_construction_never_calls_default_client(monkeypatch):
    """Stronger, environment-independent version of the guard above: force
    _default_client to blow up if it is ever invoked, and prove __init__
    alone never reaches it. This stays a valid regression test even in an
    environment where the cloud extra happens to be installed."""
    import dhvani.backends.tier1_chirp as mod

    def _boom():
        raise AssertionError("__init__ must not call _default_client()")

    monkeypatch.setattr(mod, "_default_client", _boom)
    Tier1Chirp()  # must not raise
