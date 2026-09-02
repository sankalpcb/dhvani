"""Calibration phase 3: measure a Tier 2 delta (design G6).

The Tier 1 analogue is escalate_selected(). This differs in two ways that
matter, and both are asserted here:

  the scarce resource is QUOTA, not dollars, so exhaustion must stop the
  pass and return the rows already earned rather than raise;

  the BASELINE is what Tier 2 actually repaired -- Tier 1's text when Tier 1
  ran, Tier 0's when it did not -- so a row carries before_text.
"""

import pytest

from dhvani.calibrate import repair_selected
from dhvani.quota import QuotaGate, TokenBucket
from dhvani.store import Store

T0V = "tier0|hi|m"
T1V = "tier1|hi-IN"


class FakeTier2:
    name = "tier2"
    variant_key = "model=fake;lang=hi-IN;prompt=t"

    def __init__(self, reply=None):
        self.seen = []
        self._reply = reply or (lambda text: f"repaired({text})")

    def cost_per_call(self, segment):
        return 0.0

    def transcribe(self, segment):
        text = self._source(segment.segment_id)
        self.seen.append(text)
        return {"text": self._reply(text), "signals": {}}


class Seg:
    def __init__(self, sid):
        self.segment_id = sid
        self.t_start_ms = 0
        self.t_end_ms = 2000
        self.pcm = b""


def _fixture(n=3, cap=10, with_tier1=False):
    store = Store(":memory:")
    selected = []
    for i in range(n):
        sid = f"s{i}"
        store.put_reference(sid, f"reference {i}", "hi-IN", f"spk{i}", "D")
        store.put_hypothesis(sid, "tier0", f"tier0 {i}", {}, 0.0, T0V)
        if with_tier1:
            store.put_hypothesis(sid, "tier1", f"tier1 {i}", {}, 0.0, T1V)
        selected.append({"segment_id": sid, "risk": 0.05, "duration_ms": 2000,
                         "tier0_variant": T0V})
    gate = QuotaGate(store=store, tier="tier2", cap=cap,
                     bucket=TokenBucket(capacity=999, refill_per_sec=1.0),
                     today=lambda: "2026-09-02")
    return store, selected, gate


def _run(store, selected, gate, backend, **kw):
    hyps: dict = {}
    backend._source = hyps.get
    segments = {s["segment_id"]: Seg(s["segment_id"]) for s in selected}
    return repair_selected(selected, backend, store, gate, hyps, segments, **kw)


def test_rows_carry_the_repaired_text_and_its_baseline():
    store, selected, gate = _fixture(n=2)
    with store:
        rows = _run(store, selected, gate, FakeTier2())

    assert len(rows) == 2
    r = rows[0]
    assert r["tier2_text"] == "repaired(tier0 0)"
    assert r["before_text"] == "tier0 0"
    assert r["reference"] == "reference 0"
    assert r["tier0_text"] == "tier0 0"


def test_the_baseline_is_tier1_when_tier1_ran():
    """Otherwise Tier 2 is credited with Tier 1's improvement as well."""
    store, selected, gate = _fixture(n=2, with_tier1=True)
    with store:
        rows = _run(store, selected, gate, FakeTier2(), tier1_variant=T1V)

    assert rows[0]["before_text"] == "tier1 0"
    assert rows[0]["tier2_text"] == "repaired(tier1 0)"
    # tier0_text is still recorded, so a reader can see the whole chain.
    assert rows[0]["tier0_text"] == "tier0 0"


def test_quota_exhaustion_stops_the_pass_and_keeps_what_it_earned():
    """Graceful degradation applies to calibration too: a partial
    measurement is worth more than a raised exception, provided the caller
    can tell it was partial."""
    store, selected, gate = _fixture(n=5, cap=2)
    with store:
        rows = _run(store, selected, gate, FakeTier2())

    assert len(rows) == 2, "kept calling past the quota"


def test_a_cached_repair_costs_no_quota():
    store, selected, gate = _fixture(n=2)
    with store:
        _run(store, selected, gate, FakeTier2())
        used = store.quota_used("tier2", "2026-09-02")
        backend = FakeTier2()
        rows = _run(store, selected, gate, backend)

        assert store.quota_used("tier2", "2026-09-02") == used
        assert backend.seen == [], "re-repaired work already cached"
        assert len(rows) == 2, "cached rows must still be measured"


def test_a_segment_without_a_reference_is_skipped():
    """No ground truth means no meaningful delta -- same rule as Tier 1."""
    store, selected, gate = _fixture(n=2)
    with store:
        store.conn.execute("DELETE FROM references_ WHERE segment_id = 's0'")
        store.conn.commit()
        rows = _run(store, selected, gate, FakeTier2())

    assert [r["reference"] for r in rows] == ["reference 1"]


def test_repairs_are_persisted_for_a_resumed_run():
    store, selected, gate = _fixture(n=2)
    with store:
        _run(store, selected, gate, FakeTier2())
        got = store.get_hypothesis("s0", "tier2",
                                   variant_key=FakeTier2.variant_key)
        assert got["text"] == "repaired(tier0 0)"
        assert got["cost_usd"] == 0.0, "Tier 2 is free; the ledger must say so"


# --- write_table generalized to a second tier ---

import json  # noqa: E402

from dhvani.calibrate import MIN_BUCKET_SAMPLES, NoMeasuredBuckets, write_table  # noqa: E402


def _rows(n, tier="tier2", risk=0.05):
    return [{"risk": risk, "reference": "one two three",
             "tier0_text": "one two wrong", "before_text": "one two wrong",
             f"{tier}_text": "one two three"} for _ in range(n)]


def test_write_table_can_write_a_tier2_table(tmp_path):
    out = tmp_path / "t.json"
    table = write_table(_rows(MIN_BUCKET_SAMPLES), _rows(MIN_BUCKET_SAMPLES),
                        str(out), 0.0, ["hi-IN"], tier="tier2")

    assert set(table) == {"tier2"}
    written = json.loads(out.read_text())
    assert "tier2" in written
    assert written["meta"]["tier"] == "tier2"
    assert written["meta"]["spend_usd"] == 0.0


def test_the_thin_bucket_floor_applies_to_tier2_too(tmp_path):
    """The floor is what stops a noisy average being published as measured.
    A free tier is not exempt -- cheapness is not evidence."""
    with pytest.raises(NoMeasuredBuckets):
        write_table(_rows(MIN_BUCKET_SAMPLES - 1), _rows(MIN_BUCKET_SAMPLES - 1),
                    str(tmp_path / "t.json"), 0.0, ["hi-IN"], tier="tier2")


def test_write_table_still_defaults_to_tier1(tmp_path):
    rows = [{"risk": 0.05, "reference": "a b", "tier0_text": "a x",
             "tier1_text": "a b"} for _ in range(MIN_BUCKET_SAMPLES)]
    table = write_table(rows, rows, str(tmp_path / "t.json"), 1.0, ["hi-IN"])
    assert set(table) == {"tier1"}


def test_the_repair_phase_needs_no_audio(tmp_path, monkeypatch, capsys):
    """Tier 2 is text-in, text-out. Requiring phase 1's PCM cache would
    couple it to audio it never reads -- and would make phase 3 impossible
    whenever that cache is gone, which is exactly the state this project is
    in. Tier 1 needs the audio because it sends PCM inline; Tier 2 does not.
    """
    import json as _json

    from dhvani.cli_calibrate import main as calib_main
    from dhvani.store import Store as _Store

    db = tmp_path / "c.db"
    scored = []
    with _Store(str(db)) as s:
        for i in range(MIN_BUCKET_SAMPLES):
            sid = f"{i:064d}"
            s.put_reference(sid, "एक दो तीन", "hi-IN", f"spk{i}", f"D{i}")
            s.put_hypothesis(sid, "tier0", "1 2 3", {}, 0.0, "t0v")
            scored.append({"segment_id": sid, "risk": 0.05, "lang": "hi-IN",
                           "duration_ms": 2000, "tier0_variant": "t0v"})
    sfile = tmp_path / "scored.json"
    sfile.write_text(_json.dumps(scored))

    # No PCM cache exists anywhere near tmp_path -- that is the point.
    monkeypatch.chdir(tmp_path)

    fixtures = tmp_path / "fx"
    from dhvani.backends.tier2_gemini import Tier2Gemini
    from dhvani.ids import variant_slug
    backend = Tier2Gemini(hypothesis_source=lambda s: "", lang="hi-IN", model="")
    fdir = fixtures / "tier2" / variant_slug(backend.variant_key)
    fdir.mkdir(parents=True)
    for row in scored:
        (fdir / f"{row['segment_id']}.json").write_text(
            _json.dumps({"text": "एक दो तीन", "signals": {}}))

    code = calib_main(["repair", "--db", str(db), "--scored-in", str(sfile),
                       "--out", str(tmp_path / "t2.json"),
                       "--mode", "replay", "--fixtures", str(fixtures)])

    assert code == 0, capsys.readouterr().err
    table = _json.loads((tmp_path / "t2.json").read_text())
    assert table["meta"]["tier"] == "tier2"
    assert table["tier2"]["0.0-0.1"] > 0.0, "repair should score an improvement"


def test_pacing_waits_rather_than_abandoning_the_batch():
    """RateLimited is SOFT -- nothing was consumed, the call simply came too
    soon. Breaking out of the batch on it silently measures fewer segments
    than were selected, which is how a bucket quietly falls under the
    publication floor and a run reports having measured nothing.

    Only QuotaExhausted, the hard limit, may stop the pass.
    """
    from dhvani.quota import QuotaGate, TokenBucket

    clock = type("C", (), {"t": 0.0, "__call__": lambda s: s.t})()
    waits = []

    def fake_sleep(seconds):
        waits.append(seconds)
        clock.t += seconds

    store = Store(":memory:")
    selected = []
    for i in range(20):
        sid = f"s{i}"
        store.put_reference(sid, f"ref {i}", "hi-IN", f"spk{i}", "D")
        store.put_hypothesis(sid, "tier0", f"t0 {i}", {}, 0.0, T0V)
        selected.append({"segment_id": sid, "risk": 0.05,
                         "duration_ms": 2000, "tier0_variant": T0V})

    # Capacity 5, so 15 of the 20 must wait for a refill.
    gate = QuotaGate(store=store, tier="tier2", cap=1000,
                     bucket=TokenBucket(capacity=5, refill_per_sec=1.0, now=clock),
                     today=lambda: "2026-09-02")
    hyps = {}
    backend = FakeTier2()
    backend._source = hyps.get
    segments = {s["segment_id"]: Seg(s["segment_id"]) for s in selected}

    with store:
        rows = repair_selected(selected, backend, store, gate, hyps, segments,
                               sleep=fake_sleep)

        assert len(rows) == 20, f"pacing lost {20 - len(rows)} segments"
        assert waits, "never waited; the bucket should have been exhausted"


def test_quota_exhaustion_still_stops_immediately():
    """The hard limit must NOT be waited out -- it does not refill today."""
    store, selected, gate = _fixture(n=5, cap=2)
    waits = []
    with store:
        rows = _run(store, selected, gate, FakeTier2(),
                    sleep=lambda s: waits.append(s))

    assert len(rows) == 2
    assert waits == [], "slept waiting for a daily quota that refills tomorrow"
