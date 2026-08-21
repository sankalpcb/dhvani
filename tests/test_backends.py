import json
import numpy as np
import pytest

from dhvani.backends.base import Recorded, FixtureMissing
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def cost_per_call(self, segment):
        return 0.5

    def transcribe(self, segment):
        self.calls += 1
        return {"text": f"hello-{self.calls}", "signals": {}}


class RaisingBackend:
    """Backend whose transcribe() always raises, to prove spend is recorded
    before the call so a crash cannot leave money spent but unrecorded."""

    name = "fake"

    def cost_per_call(self, segment):
        return 0.5

    def transcribe(self, segment):
        raise RuntimeError("boom: simulated API crash")


def _seg(sid="a" * 64):
    return Segment(segment_id=sid, t_start_ms=0, t_end_ms=3000,
                   pcm=np.zeros(10, dtype=np.int16))


def test_record_mode_writes_a_fixture(tmp_path):
    inner = FakeBackend()
    with Store(str(tmp_path / "t.db")) as store:
        b = Recorded(inner, mode="record", fixture_dir=str(tmp_path), store=store)
        out = b.transcribe(_seg())
        assert out["text"] == "hello-1"
        written = tmp_path / "fake" / f"{'a' * 64}.json"
        assert json.loads(written.read_text())["text"] == "hello-1"


def test_replay_mode_reads_fixture_without_calling_inner(tmp_path):
    inner = FakeBackend()
    with Store(str(tmp_path / "t.db")) as store:
        Recorded(inner, "record", str(tmp_path), store).transcribe(_seg())
    assert inner.calls == 1

    replayer = Recorded(inner, "replay", str(tmp_path), None)
    assert replayer.transcribe(_seg())["text"] == "hello-1"
    assert inner.calls == 1, "replay must not invoke the live backend"


def test_replay_hard_fails_on_missing_fixture(tmp_path):
    """Silent fallback to live is how a test run quietly spends money."""
    b = Recorded(FakeBackend(), "replay", str(tmp_path), None)
    with pytest.raises(FixtureMissing, match="no fixture"):
        b.transcribe(_seg())


def test_replay_costs_nothing(tmp_path):
    inner = FakeBackend()
    with Store(str(tmp_path / "t.db")) as store:
        Recorded(inner, "record", str(tmp_path), store).transcribe(_seg())
    b = Recorded(inner, "replay", str(tmp_path), None)
    assert b.cost_per_call(_seg()) == 0.0


def test_live_mode_records_spend(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        b = Recorded(FakeBackend(), "live", str(tmp_path), store)
        b.transcribe(_seg())
        assert store.total_spend() == pytest.approx(0.5)


def test_live_mode_fails_closed_at_budget_ceiling(tmp_path):
    with Store(str(tmp_path / "t.db")) as store:
        store.record_spend("fake", 19.8)
        b = Recorded(FakeBackend(), "live", str(tmp_path), store)
        with pytest.raises(BudgetExceeded):
            b.transcribe(_seg())


def test_live_mode_records_spend_before_call_so_crashes_cannot_undercount(tmp_path):
    """RULING (overrides brief): record_spend() must happen BEFORE
    inner.transcribe(), not after. A crash inside the API call must still
    leave the spend recorded, so total_spend() can never under-count and
    the USD 20 ceiling can never be breached on restart. This is pessimistic
    accounting: a failed call over-counts, which fails safe."""
    with Store(str(tmp_path / "t.db")) as store:
        b = Recorded(RaisingBackend(), "live", str(tmp_path), store)
        with pytest.raises(RuntimeError, match="boom"):
            b.transcribe(_seg())
        assert store.total_spend() == pytest.approx(0.5)


def test_live_mode_without_store_raises_value_error(tmp_path):
    """A live-mode wrapper with store=None would make paid calls with zero
    budget enforcement and zero spend accounting — the USD 20 ceiling would
    simply be absent. This must be rejected at construction time."""
    with pytest.raises(ValueError):
        Recorded(FakeBackend(), "live", str(tmp_path), store=None)


def test_record_mode_without_store_raises_value_error(tmp_path):
    """record mode also invokes the paid inner backend (it calls, then
    saves the response), so it has the exact same unmetered-spend hole as
    live mode and must be rejected the same way."""
    with pytest.raises(ValueError):
        Recorded(FakeBackend(), "record", str(tmp_path), store=None)


def test_replay_mode_without_store_still_constructs(tmp_path):
    """Replay never calls anything, so store=None is the intended, safe
    configuration for replay and must not regress."""
    b = Recorded(FakeBackend(), "replay", str(tmp_path), store=None)
    assert b.mode == "replay"
    assert b.store is None
