"""The 1,000-injected-failure campaign (spec §9.2).

The headline claim format the spec asks for is:

    "Matches full-Chirp caption quality at N x lower cost per audio-hour,
     with zero data loss across 1000 injected failures."

tests/test_invariants.py asserts I1 under six hand-picked fault combinations.
Six is not a thousand, and the claim was unbacked until this file existed.

What is counted is INJECTIONS, not trials. A trial can inject several
failures (a truncated delivery that is also reordered), and one that draws
an empty fault set injects none. Counting trials and calling them failures
would be the easy way to make the number look right, so the campaign
instruments the backend and asserts on what it actually did.

Two outcomes are legitimate and the campaign accepts both, because the
faults differ in kind:

  survivable   partial / duplicate / reorder -- the drain converges and
               every segment carries its escalated text.
  fatal        timeout / rate_limit / server_error raise on EVERY poll, so
               the job exhausts MAX_JOB_ATTEMPTS and is dead-lettered. The
               track is left untouched.

I1 is what must hold in BOTH: every input segment appears exactly once,
never lost, never duplicated, never half-merged. "Zero data loss" is a
statement about segments, not about whether escalation succeeded.
"""

import random

import numpy as np
import pytest

from dhvani.backends.async_base import SyncAsyncAdapter
from dhvani.backends.chaos import FAULTS, ChaosBackend, TransientError
from dhvani.config import POLICY_ID
from dhvani.escalate import escalate
from dhvani.pipeline import TrackEntry
from dhvani.reconcile import reconcile
from dhvani.segmenter import Segment
from dhvani.store import Store
from dhvani.track import entries_from_json, entries_to_json

TARGET_INJECTIONS = 1000

N = 8
ENTRIES = [TrackEntry(chr(97 + i) * 64, i * 3000, (i + 1) * 3000,
                      "raw", 0.65, "review") for i in range(N)]
SEGMENTS = {e.segment_id: Segment(e.segment_id, e.t_start_ms, e.t_end_ms,
                                  np.zeros(10, dtype=np.int16))
            for e in ENTRIES}
TABLE = {"tier1": {"0.6-0.7": 18.0}}
FATAL = {"timeout", "rate_limit", "server_error"}


class StubSync:
    name = "tier1"
    variant_key = "tier1|hi-IN"

    def cost_per_call(self, segment):
        return 0.00075

    def transcribe(self, segment):
        return {"text": f"fixed-{segment.segment_id[:4]}", "signals": {}}


class CountingChaos(ChaosBackend):
    """ChaosBackend that records every failure it actually injects.

    Subclassed rather than reimplemented so the campaign exercises the same
    injector the rest of the suite does. A parallel implementation could
    drift and quietly stop injecting what it claims to.
    """

    def __init__(self, inner, faults=(), seed=0):
        super().__init__(inner, faults=faults, seed=seed)
        self.injected = 0

    def poll(self, job_id):
        before_partial = len(self._partial_delivered)
        try:
            results = super().poll(job_id)
        except TransientError:
            self.injected += 1          # a raised poll is one injection
            raise
        if results is None:
            return None
        # A delivery happened. Count the mutations that were applied to it.
        if len(self._partial_delivered) > before_partial:
            self.injected += 1
        if "reorder" in self.faults:
            self.injected += 1
        if "duplicate" in self.faults:
            self.injected += 1
        return results


def _drain(backend, store, max_passes=20):
    version = store.latest_track_version("vid1")
    converged = False
    for _ in range(max_passes):
        try:
            version = reconcile("vid1", backend, store)
        except TransientError:
            continue
        if not any(job["tier"] == backend.name
                   and job["variant_key"] == backend.variant_key
                   for job in store.open_jobs()):
            converged = True
            break
    return version, converged


def _trial(faults, seed):
    """One escalate-and-drain under `faults`. Returns (texts, injected)."""
    with Store(":memory:") as store:
        store.put_track("vid1", 1, POLICY_ID, entries_to_json(ENTRIES), 0.0)
        backend = CountingChaos(SyncAsyncAdapter(StubSync()),
                                faults=faults, seed=seed)
        escalate("vid1", ENTRIES, SEGMENTS, backend, store, TABLE, 10.0)
        version, _ = _drain(backend, store)
        entries = entries_from_json(
            store.get_track("vid1", version)["content_json"])
    return entries, backend.injected


SURVIVABLE = tuple(f for f in FAULTS if f not in FATAL)


def _plan(campaign_seed=20260902, trials=400):
    """Deterministic (faults, seed) pairs. Same plan on every run (I5).

    STRATIFIED by regime, deliberately. Drawing 1-3 faults uniformly from
    all six put ~80% of trials in the fatal regime, because a set is fatal
    if it contains ANY of the three raising faults. That satisfies the
    injection count while under-exercising the half that matters: a
    dead-lettered job proves "nothing merged, nothing lost", which is a
    weaker property than "everything merged exactly once under a delivery
    that was truncated, shuffled and duplicated".

    Alternating the regimes keeps both well covered, and the test asserts
    the balance rather than trusting this comment.
    """
    rng = random.Random(campaign_seed)
    plan = []
    for i in range(trials):
        if i % 2 == 0:
            # Survivable only: the merge actually runs and must be exact.
            k = rng.randint(1, len(SURVIVABLE))
            faults = rng.sample(SURVIVABLE, k)
        else:
            # At least one fatal, optionally mixed with survivable ones.
            faults = rng.sample(sorted(FATAL), rng.randint(1, 2))
            if rng.random() < 0.5:
                faults += rng.sample(SURVIVABLE, rng.randint(1, 2))
        plan.append((tuple(sorted(faults)), rng.randrange(10_000)))
    return plan


def test_zero_data_loss_across_a_thousand_injected_failures():
    """Spec §9.2. Every segment survives every injection, exactly once."""
    injected = 0
    trials = 0
    survivable_runs = 0
    dead_lettered = 0

    for faults, seed in _plan():
        entries, n = _trial(faults, seed)
        injected += n
        trials += 1

        ids = [e.segment_id for e in entries]
        # I1, the actual claim: nothing lost, nothing duplicated.
        assert len(ids) == N, f"segment count changed under {faults}: {len(ids)}"
        assert len(set(ids)) == N, f"duplicate segment under {faults}"
        assert sorted(ids) == sorted(SEGMENTS), f"identity changed under {faults}"

        texts = [e.text for e in entries]
        expected = [f"fixed-{e.segment_id[:4]}" for e in entries]
        if set(faults) & FATAL:
            # Every poll raised; the job dead-letters and nothing merges.
            assert texts == ["raw"] * N, f"partial merge under fatal {faults}"
            dead_lettered += 1
        else:
            # Survivable: the drain converged and every segment carries its
            # escalated text. A half-merged track would fail here -- which is
            # the failure mode identity checks alone cannot see.
            assert texts == expected, f"incomplete merge under {faults}"
            survivable_runs += 1

        if injected >= TARGET_INJECTIONS and trials >= 200:
            break

    assert injected >= TARGET_INJECTIONS, (
        f"campaign injected only {injected} failures across {trials} trials; "
        f"the §9.2 claim needs at least {TARGET_INJECTIONS}"
    )
    # Both regimes must actually have been exercised, or the campaign proves
    # only one half of the property while reporting a big number.
    # Both regimes must be MEANINGFULLY covered, not merely present. An
    # earlier version of this campaign drew faults uniformly and landed
    # 152 dead-lettered against 55 survivable -- it hit 1000 injections
    # while barely exercising the merge path, which is the half where loss
    # could actually occur.
    assert survivable_runs >= trials * 0.4, (
        f"merge path under-exercised: {survivable_runs} survivable of "
        f"{trials} trials"
    )
    assert dead_lettered >= trials * 0.4, (
        f"failure path under-exercised: {dead_lettered} dead-lettered of "
        f"{trials} trials"
    )


def test_the_campaign_is_deterministic():
    """I5. A headline number that changes run to run is not a measurement."""
    a = [_trial(f, s)[1] for f, s in _plan()[:25]]
    b = [_trial(f, s)[1] for f, s in _plan()[:25]]
    assert a == b


def test_the_plan_covers_every_fault_kind():
    """A thousand injections of one fault would satisfy the count and prove
    almost nothing."""
    seen = set()
    for faults, _ in _plan():
        seen.update(faults)
    assert seen == set(FAULTS), f"never exercised: {set(FAULTS) - seen}"
