import numpy as np
import pytest

from dhvani.backends.base import Backend
from dhvani.backends.tier1_chirp import (
    Tier1Chirp, USD_PER_MIN_DYNAMIC_BATCH, USD_PER_MIN_STANDARD,
    BILLING_INCREMENT_SEC,
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
    # 60000ms is already a whole multiple of BILLING_INCREMENT_SEC (15s),
    # so rounding up (fix round 1) does not change this figure.
    assert Tier1Chirp(client=StubClient()).cost_per_call(_seg(60000)) == \
        pytest.approx(USD_PER_MIN_DYNAMIC_BATCH)


def test_cost_scales_with_duration():
    # 30000ms is also a whole multiple of 15s, so this still holds after
    # fix round 1's rounding was introduced.
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


def test_variant_key_captures_everything_that_changes_the_output():
    """Fix round 2 (I2): lang and recognizer both change what Chirp
    returns for byte-identical PCM, so both must be in the variant
    identity that keys the cache and the fixture path."""
    base = Tier1Chirp(client=StubClient(), lang="hi-IN")
    other_lang = Tier1Chirp(client=StubClient(), lang="ml-IN")
    other_recognizer = Tier1Chirp(client=StubClient(), lang="hi-IN",
                                  recognizer="projects/p/locations/global/recognizers/r")

    assert base.variant_key == Tier1Chirp(client=StubClient(), lang="hi-IN").variant_key
    assert base.variant_key != other_lang.variant_key
    assert base.variant_key != other_recognizer.variant_key


def test_construction_with_no_client_succeeds():
    """G5 regression guard, mirroring Task 9's Tier0Conformer guard. A
    stranger who clones the repo and runs the offline replay workflow has
    no google-cloud-speech installed and no GCP credentials. Constructing
    Tier1Chirp() with no injected client must not import anything or touch
    the network -- only an actual transcribe() call is allowed to do that.

    This test used to be named ..._without_google_cloud_speech and to claim
    that this project's venv genuinely had none installed, so that the
    assertion below was "real evidence, not a simulated one". The `cloud`
    extra IS installed here now, and application default credentials exist
    on this machine, so that evidence is gone: passing here no longer says
    anything about the stranger's environment, only that construction is
    cheap. The environment-independent guarantee is the next test's job --
    it forces _default_client to blow up if reached, which holds whatever
    is installed. Kept as a cheap smoke check, no longer as proof."""
    b = Tier1Chirp()
    assert b.lang == "hi-IN"


def test_construction_never_calls_default_client(monkeypatch):
    """Stronger, environment-independent version of the guard above: force
    _default_client to blow up if it is ever invoked, and prove __init__
    alone never reaches it. This stays a valid regression test even in an
    environment where the cloud extra happens to be installed."""
    import dhvani.backends.tier1_chirp as mod

    def _boom(recognizer=""):
        raise AssertionError("__init__ must not call _default_client()")

    monkeypatch.setattr(mod, "_default_client", _boom)
    Tier1Chirp()  # must not raise


# --- Fix round 1, Finding 1: billing rounds up to BILLING_INCREMENT_SEC ---

def test_cost_rounds_up_to_the_billing_increment():
    """A segment shorter than one billing increment must be billed as a
    full increment, not its exact wall-clock duration. Google Cloud
    Speech-to-Text has historically billed synchronous/batch requests in
    whole 15s increments per request, and charging exact duration would
    understate cost -- Recorded (base.py) checks this exact number against
    the USD 20 ceiling before every live call, so understating it would
    let real spend breach the ceiling while the ledger still reads under
    budget. This is the whole point of the fix."""
    b = Tier1Chirp(client=StubClient())
    sub_increment_cost = b.cost_per_call(_seg(2000))
    full_increment_cost = b.cost_per_call(_seg(BILLING_INCREMENT_SEC * 1000))
    exact_duration_cost = USD_PER_MIN_DYNAMIC_BATCH * (2000 / 60000.0)

    assert sub_increment_cost == pytest.approx(full_increment_cost)
    assert sub_increment_cost == pytest.approx(
        USD_PER_MIN_DYNAMIC_BATCH * BILLING_INCREMENT_SEC / 60.0
    )
    assert sub_increment_cost != pytest.approx(exact_duration_cost)


def test_cost_does_not_round_up_an_exact_multiple():
    """A duration that already lands exactly on a billing increment
    boundary must not be bumped to the next one -- rounding up is ceiling
    division, not "always add an increment"."""
    b = Tier1Chirp(client=StubClient())
    exact = b.cost_per_call(_seg(BILLING_INCREMENT_SEC * 1000))
    assert exact == pytest.approx(USD_PER_MIN_DYNAMIC_BATCH * BILLING_INCREMENT_SEC / 60.0)

    just_over = b.cost_per_call(_seg(BILLING_INCREMENT_SEC * 1000 + 1))
    assert just_over == pytest.approx(
        USD_PER_MIN_DYNAMIC_BATCH * (2 * BILLING_INCREMENT_SEC) / 60.0
    )


# --- Fix round 1, Finding 2: the recognizer parameter must do something ---

def test_recognizer_path_uses_configured_recognizer_when_given():
    """When a caller configures a real recognizer resource ID, it must be
    used verbatim rather than silently ignored in favor of the wildcard
    path. _recognizer_path has no google-cloud imports, so it is testable
    directly without the `cloud` extra installed -- it is exactly the
    logic _default_client's real request-building calls at transcribe
    time."""
    import dhvani.backends.tier1_chirp as mod

    configured = "projects/my-proj/locations/global/recognizers/my-recognizer"
    assert mod._recognizer_path(configured) == configured


def test_recognizer_path_falls_back_to_wildcard_when_empty(monkeypatch):
    """The default (empty recognizer) falls back to the wildcard-per-project
    path, built via _project() and pinned to the configured REGION.

    This previously asserted `locations/global`. scripts/spike_chirp.py
    (2026-08-24) proved that wrong: `global` hosts no chirp model of any
    generation, so a global recognizer path can never resolve one. Changed
    deliberately, not to make a failure go away."""
    import dhvani.backends.tier1_chirp as mod

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    assert mod._recognizer_path("") == (
        f"projects/test-project/locations/{mod.SPEECH_LOCATION}/recognizers/_"
    )


def test_recognizer_path_is_never_global(monkeypatch):
    """Regression guard for the spike's finding: `global` holds no chirp
    model, so reverting the recognizer path to it would make every live
    Tier 1 call fail with 'model does not exist in the location named
    global'. Pins the region away from that value."""
    import dhvani.backends.tier1_chirp as mod

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    assert mod.SPEECH_LOCATION != "global"
    assert "locations/global" not in mod._recognizer_path("")


def test_configured_model_is_not_the_withdrawn_chirp_3():
    """chirp_3 returned 403 'no longer generally available' for hi-IN in
    asia-south1, and does not exist in us-central1 or europe-west4."""
    import dhvani.backends.tier1_chirp as mod

    assert mod.SPEECH_MODEL != "chirp_3"


def test_recognizer_path_fails_closed_without_project_when_empty(monkeypatch):
    """The empty-recognizer fallback still fails closed if
    GOOGLE_CLOUD_PROJECT is unset -- unchanged _project() behavior, now
    reachable through the new _recognizer_path indirection too."""
    import dhvani.backends.tier1_chirp as mod

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(RuntimeError):
        mod._recognizer_path("")
