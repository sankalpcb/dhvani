import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.tier1_chirp import cost_for_duration_ms
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.segmenter import Segment
from dhvani.store import Store, BudgetExceeded


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        from dhvani.backends.tier1_chirp import cost_for_duration_ms
        return cost_for_duration_ms(segment.t_end_ms - segment.t_start_ms)

    def transcribe(self, segment):
        return {"text": "escalated", "signals": {}}


@pytest.fixture
def store(tmp_path):
    with Store(str(tmp_path / "t.db")) as s:
        yield s


def _entries():
    return [TrackEntry("a" * 64, 0, 3000, "raw", 0.65, "review"),
            TrackEntry("b" * 64, 3000, 6000, "raw", 0.05, "ship")]


def _segments():
    """Real Segment objects — Tier1Chirp.transcribe() reads segment.pcm."""
    return {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in _entries()}


SEGMENTS = _segments()
TABLE = {"tier1": {"0.6-0.7": 18.0}}


def _live_stack(store, text="escalated"):
    """The production shape: SyncAsyncAdapter(Recorded(sync backend)).

    Spend only lands when the call actually happens, which for
    SyncAsyncAdapter is at poll() -- so a test about MONEY has to drive the
    poll, not just escalate(). Tests about deduplication assert on jobs and
    call counts instead, which is the more direct claim anyway.
    """
    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp
    inner = Tier1Chirp(client=ExplodingClient(), lang="hi-IN")
    inner.transcribe = lambda seg: {"text": text, "signals": {}}
    return SyncAsyncAdapter(Recorded(inner, "live", "fixtures", store))


def test_zero_budget_submits_nothing(store):
    assert escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                    store, TABLE, budget_usd=0.0) is None


def test_empty_delta_table_submits_nothing(store):
    """No measured improvement means no candidate has positive delta."""
    assert escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                    store, {}, budget_usd=10.0) is None


def test_escalation_registers_a_job_with_the_selected_segments(store):
    job_id = escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, TABLE, budget_usd=10.0)
    assert job_id is not None
    job = store.get_job(job_id)
    assert job["segment_ids"] == ["a" * 64]
    assert job["state"] == "pending"
    assert job["variant_key"] == "tier1|hi-IN"


def test_low_risk_segments_are_not_escalated(store):
    job_id = escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, TABLE, budget_usd=10.0)
    assert "b" * 64 not in store.get_job(job_id)["segment_ids"]


def test_spend_lands_when_the_call_happens_not_when_it_is_planned(store):
    """escalate() used to reserve the batch itself, which double-charged:
    Recorded reserves again at poll time, where the request actually goes
    out. escalate() now only CHECKS the ceiling, so nothing is on the
    ledger until the call is made."""
    backend = _live_stack(store)
    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    assert job_id is not None
    assert store.total_spend() == 0.0, "planning a batch is not spending"

    _drain_once(backend, store)
    assert store.total_spend() == pytest.approx(cost_for_duration_ms(3000))


def test_escalation_fails_closed_at_the_ceiling(store):
    # The only candidate ever selected here is "a" (risk 0.65, bucket
    # "0.6-0.7", the sole key in TABLE); "b" (risk 0.05) has no matching
    # bucket, delta 0.0, and is excluded by plan()'s invariant I3. That
    # single 3000ms segment is priced by cost_for_duration_ms(3000): V2
    # bills rounded up to 1s at $0.004/min. Reserving
    # the brief's literal $19.999 first leaves $0.001 of headroom — more
    # than the batch actually costs — so it would NOT breach the ceiling
    # (19.999 + 0.00075 == 19.99975 <= 20.0, legally under the "boundary:
    # projected == MAX_SPEND_USD is allowed" rule in Store.reserve_spend).
    # Reserve just enough that the batch's real cost tips it over instead.
    from dhvani.backends.tier1_chirp import cost_for_duration_ms
    almost_full = 20.0 - cost_for_duration_ms(3000) + 0.0001
    store.reserve_spend("tier1", almost_full)
    with pytest.raises(BudgetExceeded):
        escalate("vid1", _entries(), SEGMENTS, SyncAsyncAdapter(StubSync()),
                 store, TABLE, budget_usd=10.0)


def test_resubmitting_the_same_batch_is_idempotent(store):
    """Re-escalating an unfinished track adds no work and no charge.

    This used to assert `first == second`, which passed for the wrong
    reason: job_id is content-addressed, so the second call re-entered
    submit(), got the same id back, no-opped against INSERT OR IGNORE --
    and reserved the batch's cost a second time on the way there. Equal
    job ids were never the invariant; "the store gains nothing and the
    ledger gains nothing" is, so assert that instead.
    """
    backend = SyncAsyncAdapter(StubSync())
    first = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    spend_after_first = store.total_spend()

    second = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    assert second is None
    assert store.total_spend() == pytest.approx(spend_after_first)
    # The first job is untouched and still the only one: the resubmission
    # neither duplicated it nor disturbed the batch reconcile() will poll.
    assert [job["job_id"] for job in store.open_jobs()] == [first]
    assert store.get_job(first)["segment_ids"] == ["a" * 64]


def test_resubmitting_an_in_flight_batch_reserves_no_additional_spend(store):
    """The double reservation carried forward from phase 3.

    escalate() reserved the whole batch's cost before it knew whether the
    batch was new. submit() is content-addressed and put_job() is INSERT OR
    IGNORE, so re-running --escalate while a job is still outstanding
    registers no new work -- but it charged for that work again, once per
    invocation, with nothing bounding the total but the $20 ceiling itself.
    """
    backend = _live_stack(store)
    escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    _drain_once(backend, store)
    after_first = store.total_spend()
    assert after_first > 0.0, "the first escalation must really cost something"

    escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    _drain_once(backend, store)

    assert store.total_spend() == pytest.approx(after_first)


def test_repeated_escalation_of_an_in_flight_batch_stays_bounded(store):
    """"Unbounded across reconcile passes" stated as a test: the operator
    loop is `--escalate --reconcile` re-run until the track converges, and
    every pass before convergence used to add another full batch charge."""
    backend = _live_stack(store)
    escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    _drain_once(backend, store)
    after_first = store.total_spend()

    for _ in range(10):
        escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    assert store.total_spend() == pytest.approx(after_first)


def test_resubmitting_a_delivered_batch_reserves_no_additional_spend(store):
    """A settled segment is already paid for. Same backend, same variant,
    same audio yields the same answer, so re-escalating it buys nothing --
    exactly the rule pipeline.run() already applies to Tier 0 via
    store.get_hypothesis() before calling tier0.transcribe()."""
    backend = SyncAsyncAdapter(StubSync())
    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    after_first = store.total_spend()

    # Settle it the way reconcile() does: a hypothesis per segment under
    # this backend's (name, variant_key), then the job out of open_jobs().
    for segment_id in store.get_job(job_id)["segment_ids"]:
        store.put_hypothesis(segment_id, backend.name, "escalated", {}, 0.0,
                             backend.variant_key)
    store.set_job_state(job_id, "done")

    assert escalate("vid1", _entries(), SEGMENTS, backend, store,
                    TABLE, 10.0) is None
    assert store.total_spend() == pytest.approx(after_first)


def test_re_escalating_a_dead_lettered_batch_makes_it_pollable_again(store):
    """The last way escalate() could spend money and get nothing back.

    Once the batch is deduplicated, a segment reaches submit() only if it
    has no hypothesis and no open job -- which is exactly the state
    reconcile() leaves behind when it dead-letters a job whose results
    never arrived. Resubmitting that batch produces the same
    content-addressed job_id, so put_job()'s INSERT OR IGNORE no-ops and
    the row stays state="failed". open_jobs() never surfaces it again, so
    reconcile() never polls it -- while reserve_spend() has already
    charged for the call, and in live mode the call really was made.

    put_job() returns False precisely to report this, and nobody looked.
    An explicit re-escalation is an operator paying for another attempt,
    so it must get one.
    """
    backend = SyncAsyncAdapter(StubSync())
    first = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    # How reconcile() leaves a job it has given up on: no segment of it was
    # ever delivered, so no hypothesis exists for any of them.
    store.set_job_state(first, "failed")
    assert store.open_jobs() == [], "dead-lettered jobs drop out of the queue"

    second = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    # Same segments means the same content-addressed id -- there is no
    # second row to create, only the existing one to reopen.
    assert second == first
    assert [job["job_id"] for job in store.open_jobs()] == [first]


def test_a_hypothesis_from_another_variant_does_not_suppress_escalation(store):
    """The exclusion is keyed on (name, variant_key), not on segment_id
    alone. A Tier 0 hypothesis for the same segment is a different tier and
    must not be mistaken for Tier 1 work already done -- that would switch
    escalation off entirely for every segment the pipeline has transcribed,
    which is all of them."""
    backend = SyncAsyncAdapter(StubSync())
    for entry in _entries():
        store.put_hypothesis(entry.segment_id, "tier0", "raw", {}, 0.0,
                             "tier0|hi")

    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    assert job_id is not None
    assert store.get_job(job_id)["segment_ids"] == ["a" * 64]


def _multi_entries():
    """Three entries whose risk buckets all carry positive delta, so the
    router genuinely selects 3 candidates instead of the single-candidate
    path every other test in this file exercises."""
    return [TrackEntry("c" * 64, 0, 3000, "raw", 0.65, "review"),
            TrackEntry("d" * 64, 3000, 6000, "raw", 0.75, "review"),
            TrackEntry("e" * 64, 6000, 9000, "raw", 0.85, "review")]


def _multi_segments():
    return {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in _multi_entries()}


MULTI_SEGMENTS = _multi_segments()
MULTI_TABLE = {"tier1": {"0.6-0.7": 10.0, "0.7-0.8": 10.0, "0.8-0.9": 10.0}}


def test_multi_candidate_batch_reserves_summed_cost_in_one_call(store):
    """Regression guard for reserving per-candidate in a loop: the batch's
    total cost must be reserved with a single reserve_spend() call, not N
    partial ones."""
    calls = []
    original_check = store.check_budget

    def spy(cost_usd):
        calls.append(cost_usd)
        return original_check(cost_usd)

    store.check_budget = spy

    job_id = escalate("vid1", _multi_entries(), MULTI_SEGMENTS, SyncAsyncAdapter(StubSync()),
                      store, MULTI_TABLE, budget_usd=10.0)

    assert job_id is not None
    per_segment_cost = cost_for_duration_ms(3000)
    assert len(calls) == 1, "the batch must be checked once, not per candidate"
    assert calls[0] == pytest.approx(3 * per_segment_cost)


def test_multi_candidate_reservation_is_atomic_on_failure(store):
    """When the ledger has room for only part of the batch, nothing may be
    partially reserved and no job may be created -- the failure must be
    all-or-nothing. This fails against a per-candidate reservation loop,
    which would reserve the first 2 of 3 candidates before raising on the
    3rd, permanently burning budget for a batch that was never submitted."""
    per_segment_cost = cost_for_duration_ms(3000)
    # Leave room for exactly 2 of the 3 candidates, not all 3.
    pre_reserved = 20.0 - (2 * per_segment_cost) - 0.0000001
    store.reserve_spend("tier1", pre_reserved)
    total_before = store.total_spend()

    with pytest.raises(BudgetExceeded):
        escalate("vid1", _multi_entries(), MULTI_SEGMENTS, SyncAsyncAdapter(StubSync()),
                 store, MULTI_TABLE, budget_usd=10.0)

    assert store.total_spend() == pytest.approx(total_before)
    assert store.open_jobs() == []


def test_segments_still_missing_after_a_partial_delivery_are_re_escalated(store):
    """The exclusion must not become a blanket "escalate once, ever".

    A batch that came back incomplete leaves some segments with a Tier 1
    hypothesis and some without. Once the job is off open_jobs(), a second
    escalation must submit exactly the still-missing ones -- and charge for
    exactly those, not for the whole batch again and not for nothing.
    """
    backend = SyncAsyncAdapter(StubSync())

    job_id = escalate("vid1", _multi_entries(), MULTI_SEGMENTS, backend, store,
                      MULTI_TABLE, 10.0)
    assert len(store.get_job(job_id)["segment_ids"]) == 3

    # Exactly one of the three came back; the job is then dead-lettered by
    # reconcile() after MAX_JOB_ATTEMPTS of persistently partial delivery.
    delivered = store.get_job(job_id)["segment_ids"][0]
    store.put_hypothesis(delivered, backend.name, "escalated", {}, 0.0,
                         backend.variant_key)
    store.set_job_state(job_id, "failed")

    second = escalate("vid1", _multi_entries(), MULTI_SEGMENTS, backend, store,
                      MULTI_TABLE, 10.0)

    assert second is not None
    assert delivered not in store.get_job(second)["segment_ids"]
    assert len(store.get_job(second)["segment_ids"]) == 2


# --- C2: replay must cost nothing and must never reach a live backend ---

class ExplodingClient:
    """Any call to this is a live billed API call that must never happen."""

    def recognize_pcm(self, pcm, lang):
        raise AssertionError(
            "replay mode reached the live Chirp client -- "
            "replay must never fall back to live"
        )


def _replay_tier1(store, fixtures):
    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp
    inner = Tier1Chirp(client=ExplodingClient(), lang="hi-IN")
    return SyncAsyncAdapter(Recorded(inner, "replay", str(fixtures), store))


def test_replay_escalation_reserves_no_money(store, tmp_path):
    """escalate() priced every candidate with cost_for_duration_ms()
    unconditionally, so a replay run -- which makes no paid calls at all --
    still reserved real dollars against the $20 ceiling for calls that
    never happened. Pricing now goes through backend.cost_per_call(), and
    Recorded.cost_per_call() returns 0.0 in replay."""
    backend = _replay_tier1(store, tmp_path / "fixtures")
    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    assert job_id is not None, "replay must still plan and submit the batch"
    assert store.total_spend() == 0.0


def test_replay_escalation_never_calls_the_live_client(store, tmp_path):
    """The end-to-end guarantee, not just the price: driving a replay-mode
    escalation through to results reads fixtures and never touches the
    injected client (which raises if called)."""
    from dhvani.ids import variant_slug
    from dhvani.backends.tier1_chirp import Tier1Chirp

    variant = Tier1Chirp(lang="hi-IN").variant_key
    fixtures = tmp_path / "fixtures"
    fixture_dir = fixtures / "tier1" / variant_slug(variant)
    fixture_dir.mkdir(parents=True)
    import json as _json
    for seg_id in SEGMENTS:
        (fixture_dir / f"{seg_id}.json").write_text(
            _json.dumps({"text": "from-fixture", "signals": {}}))

    backend = _replay_tier1(store, fixtures)
    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    results = backend.poll(job_id)  # would raise AssertionError if live
    assert all(r["text"] == "from-fixture" for r in results.values())
    assert store.total_spend() == 0.0


def test_live_escalation_still_reserves_the_real_cost(store):
    """The replay fix must not make live calls free: a live-mode backend
    still prices through Tier1Chirp.cost_per_call, i.e. the one and only
    cost model, cost_for_duration_ms()."""
    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp

    backend = _live_stack(store)
    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    assert job_id is not None
    # Exactly one candidate ("a", 3000ms) is ever selected -- see
    # test_escalation_fails_closed_at_the_ceiling for why. The charge lands
    # at poll(), where Recorded makes the request.
    _drain_once(backend, store)
    assert store.total_spend() == pytest.approx(cost_for_duration_ms(3000))


# --- C3, second occurrence: the async path reserved twice ---

def _drain_once(backend, store, source="vid1"):
    from dhvani.reconcile import reconcile
    from dhvani.config import POLICY_ID
    from dhvani.track import entries_to_json
    if store.latest_track_version(source) == 0:
        store.put_track(source, 1, POLICY_ID, entries_to_json(_entries()), 0.0)
    return reconcile(source, backend, store)


def test_the_async_path_charges_each_call_exactly_once(store):
    """escalate() reserves the batch, and Recorded.transcribe() reserves
    again when the call actually happens at poll time -- so a live run was
    billed twice in the ledger for one Chirp request.

    This is C3 a second time. calibrate.escalate_selected had exactly this
    shape and was fixed by making Recorded the only reserver; the CLI's
    async path kept both, and nothing caught it because no test drove
    escalate() THROUGH a poll with a live Recorded in the stack. The
    existing live test asserts spend right after escalate(), before
    SyncAsyncAdapter has called transcribe() at all.

    Observed for real: recording one 4440ms Tier 1 fixture reserved
    $0.00067 against a true price of $0.00033.
    """
    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp

    inner = Tier1Chirp(client=ExplodingClient(), lang="hi-IN")
    inner.transcribe = lambda seg: {"text": "escalated", "signals": {}}
    backend = SyncAsyncAdapter(Recorded(inner, "live", "fixtures", store))

    job_id = escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)
    assert job_id is not None
    _drain_once(backend, store)

    one_call = cost_for_duration_ms(3000)
    assert store.total_spend() == pytest.approx(one_call), (
        "one Chirp request must appear once in the ledger, not twice"
    )


def test_the_ceiling_still_fails_closed_before_a_batch_is_submitted(store):
    """Removing escalate()'s reservation must not remove its guard: a batch
    that cannot be afforded must never reach submit(), or the job exists
    with no budget to service it."""
    from dhvani.backends.base import Recorded
    from dhvani.backends.tier1_chirp import Tier1Chirp

    inner = Tier1Chirp(client=ExplodingClient(), lang="hi-IN")
    backend = SyncAsyncAdapter(Recorded(inner, "live", "fixtures", store))
    store.reserve_spend("tier1", 20.0 - cost_for_duration_ms(3000) + 0.0001)

    with pytest.raises(BudgetExceeded):
        escalate("vid1", _entries(), SEGMENTS, backend, store, TABLE, 10.0)

    assert store.open_jobs() == [], "no job may be created for an unaffordable batch"
