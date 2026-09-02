"""Tier 2 Gemini repair (design §5).

The backend is a Backend, not a special case: it exposes transcribe()
so Recorded wraps it unchanged for record/replay. That is sound because
its input -- the best upstream hypothesis -- is itself content-addressed
and keyed (segment_id, tier), so under a fixed POLICY_ID the segment_id
determines the input and fixtures cannot collide.
"""

import pytest

from dhvani.backends.tier2_gemini import Tier2Gemini


class FakeSegment:
    def __init__(self, segment_id="seg1", t_start_ms=0, t_end_ms=4000):
        self.segment_id = segment_id
        self.t_start_ms = t_start_ms
        self.t_end_ms = t_end_ms
        self.pcm = b""


class FakeGemini:
    """Records what it was asked and returns a canned repair."""

    def __init__(self, reply="साढ़े तीन हज़ार"):
        self.reply = reply
        self.calls = []

    def repair(self, text, lang):
        self.calls.append((text, lang))
        return self.reply


def test_repair_sends_the_upstream_hypothesis_and_returns_the_repaired_text():
    client = FakeGemini()
    backend = Tier2Gemini(
        hypothesis_source=lambda sid: "3456",
        client=client,
        lang="hi-IN",
    )

    out = backend.transcribe(FakeSegment())

    assert client.calls == [("3456", "hi-IN")]
    assert out["text"] == "साढ़े तीन हज़ार"


def test_tier2_calls_are_free():
    backend = Tier2Gemini(hypothesis_source=lambda sid: "x", client=FakeGemini())
    assert backend.cost_per_call(FakeSegment()) == 0.0


def test_an_empty_hypothesis_is_not_sent():
    """There is nothing to repair, and quota is too scarce to spend on it."""
    client = FakeGemini()
    backend = Tier2Gemini(hypothesis_source=lambda sid: "   ", client=client)

    out = backend.transcribe(FakeSegment())

    assert client.calls == [], "spent a request repairing nothing"
    assert out["text"] == ""


def test_variant_key_covers_model_and_prompt():
    """Both change the output for byte-identical PCM, so both must key
    the fixture -- backends/base.py FIX ROUND 2 (I2/I3)."""
    a = Tier2Gemini(hypothesis_source=lambda s: "x", client=FakeGemini(), model="m1")
    b = Tier2Gemini(hypothesis_source=lambda s: "x", client=FakeGemini(), model="m2")

    assert a.variant_key != b.variant_key
    assert "m1" in a.variant_key
    assert "prompt=" in a.variant_key


def test_constructing_it_touches_no_network_and_imports_no_sdk():
    """G4: the suite must pass with google-genai absent. Constructing the
    backend must therefore not build a client -- mirroring Tier1Chirp."""
    backend = Tier2Gemini(hypothesis_source=lambda s: "x")

    def explode(*a, **k):
        raise AssertionError("built a live client without being asked to")

    import dhvani.backends.tier2_gemini as mod
    original, mod._default_client = mod._default_client, explode
    try:
        assert backend.name == "tier2"
        assert "prompt=" in backend.variant_key
    finally:
        mod._default_client = original
