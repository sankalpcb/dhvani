# Dhvani Tier 2 Repair — Design

**Status:** Design approved, pending implementation plan
**Date:** 2026-09-02
**Author:** Sankalp Badrinath
**Implements:** `2026-08-21-dhvani-design.md` §3 `repair_tier2` and §13 M5 — deferred from
Phase 2, the last unbuilt milestone

---

## 1. Problem

`repair_tier2` appears in the architecture diagram and the component contract table. No module
implements it. `delta_for(risk, "tier2", table)` already returns `0.0` and a test asserts it,
so the router has a Tier 2 hook wired to nothing.

M5's demo, per §13, is **graceful degradation at quota exhaustion**. Gemini's free tier allows
1,000 requests/day, and the Phase 2 plan deferred this milestone precisely because that cap
"is a hard wall and deserves its own token-bucket plus graceful degradation design".

### 1.1 The inertness trap, and why Tier 2 must not inherit it

Spec §3 draws `repair_tier2` strictly downstream of `asr_tier1`. Under the measured delta
table that placement makes Tier 2 **permanently unreachable**: both Tier 1 deltas are negative,
`plan()` excludes non-positive delta by construction (I3), so nothing escalates to Tier 1 and
therefore nothing would ever reach a post-Tier-1 repair stage. M5 would ship demonstrable only
behind `samples/demo-delta-table.json`, the deliberately fake table.

Phase 2 already had to build that crutch once. Building a second milestone that needs it would
make "the router refuses to escalate on a hunch" true of the code and false of the deliverable.

**Resolution: Tier 2 repairs the best available hypothesis.** It takes whichever hypothesis
exists for a segment — Tier 1's when present, Tier 0's otherwise — and returns repaired text.
This is what the spec's own contract already describes: `(segment, [hypothesis]) -> text` takes
a *list*, a signature only meaningful for a component that sees more than one upstream opinion.
Tier 2 composes with Tier 1 when Tier 1 runs, and degrades to repairing Tier 0 when it does
not.

### 1.2 The asymmetry that makes this milestone cheap

Measuring the Tier 1 delta cost $0.10, required a billing account, a spend ceiling, an atomic
reservation and a live-call confirmation flag. **Tier 2 is free.** 150 calibration utterances
is 150 requests against a 1,000/day quota.

Every piece of machinery Dhvani built to keep money safe is therefore *not* what Tier 2 needs.
What it needs is the same discipline applied to a different scarce resource — one that is not
denominated in dollars and that refills at midnight.

---

## 2. Goals and non-goals

### Goals

- **G1** `repair_tier2` implemented against the spec's contract, repairing the best available
  hypothesis rather than requiring Tier 1.
- **G2** Daily quota is never exceeded, including across concurrent processes and across a
  crash and restart.
- **G3** Quota exhaustion degrades gracefully: the run completes, the segment ships with its
  unrepaired text and a marker, and the repair is retried on a later run.
- **G4** The full suite still passes with no ML deps, no cloud SDK, no `google-genai`, no
  credentials and no network (goal G5 of the parent spec, unchanged).
- **G5** No test sleeps to exercise rate limiting. Time is injected.
- **G6** A smoke measurement (~20 utterances) proves the calibration path end to end and
  records replay fixtures.

### Non-goals

- **N1** No publishable `tier2` delta table entry from this work. ~20 utterances spread across
  risk buckets leaves every individual bucket under `MIN_BUCKET_SAMPLES = 20`, and
  `calibrate.py` already drops thin buckets rather than publishing a noisy average. The smoke
  run therefore proves the path and records fixtures without emitting a headline number. A
  full calibration is a later operational step.
- **N2** No prompt engineering research. One prompt, versioned, changed only with `POLICY_ID`.
- **N3** No streaming, no function calling, no multi-turn. One request, one repaired string.
- **N4** No paid Gemini tier. If the free quota is exhausted, the answer is to wait, never to
  enable billing.

---

## 3. Architecture

```
          ┌──────────────────────────────────────────┐
          │ best_hypothesis(segment_id)              │
          │   tier1 if present, else tier0           │
          └──────────────────────────────────────────┘
                              │
                              ▼
router.plan(tier="tier2") ──► escalate ──► QuotaGate ──┬── reserved ──► Tier2Gemini
   delta-gated (I3),                                   │                 (batched)
   same as tier1                                       │                     │
                                                       │                     ▼
                                                       │              tier2 hypothesis
                                                       │                     │
                                                       └── exhausted ──► job stays open
                                                                 │           │
                                                          mark segment       │
                                                       repair_unavailable    │
                                                                 │           │
                                                                 ▼           ▼
                                                         track v(n) ───► reconcile() ───► track v(n+1)
                                                         ships now          later run
```

Tier 2 is a **second asynchronous tier**. It reuses `jobs`, `open_jobs()`,
`set_job_state()`, `bump_job_attempts()` and `reconcile()` unchanged. Invariants I1–I6 extend
to it for free: the merge is keyed `(segment_id, tier)`, so re-applying a Tier 2 result is a
no-op at the database level exactly as it is for Tier 1.

This is the payoff of the Phase 2 design. Adding a tier costs a backend, a gate and a
selection rule — not a second reconciliation engine.

---

## 4. The two limits, and why they get different mechanisms

The free tier imposes two constraints with **different failure semantics**, and conflating
them is the central design error to avoid.

| | daily request cap | per-minute rate |
|---|---|---|
| Nature | a **consumable** | **pacing** |
| Over-running it | unrecoverable until midnight | returns a retryable 429 |
| Resource consumed on failure | yes — the request counted | no |
| Correct posture | fail **closed**, persist, atomic | fail **soft**, retry with backoff |
| Mechanism | `Store.reserve_quota()` | in-process token bucket |

### 4.1 Daily cap — `Store.reserve_quota()`

Mirrors `reserve_spend()` deliberately, because it is the same problem in a different currency:

```sql
INSERT INTO quota (tier, day, used)
SELECT ?, ?, ?
WHERE (SELECT COALESCE(SUM(used), 0) FROM quota WHERE tier = ? AND day = ?) + ? <= ?
```

One statement doing check-and-insert. This inherits the fix already recorded in
`backends/base.py` (FIX ROUND 2, C1): a separate `check` then `record` lets two Store handles
on the same DB read the same stale total, both pass, and both write. `reserve_spend()` was
consolidated for exactly this reason and `reserve_quota()` must not reintroduce the bug.

Reservation happens **before** the call, so a crash between reserving and calling over-counts
by one request. Over-counting quota fails safe in the same direction as over-reserving spend.

`day` is a date string on **Google's** reset boundary, which the spike of 2026-09-02 settled:
requests-per-day quotas reset at **midnight Pacific**, and limits apply **per project, not per
API key** ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).

An earlier draft of this section keyed the day by UTC and reasoned that a wrong boundary
"never over-spends, because a stale `day` key only makes the gate more conservative". **That
was exactly backwards.** UTC midnight is 16:00–17:00 Pacific the previous day, so a UTC key
rolls over about eight hours early; in that window the gate resets a counter Google has not,
and authorizes a second full day's worth of requests inside one Google day — up to 2× the cap,
in the over-spending direction the whole reservation discipline exists to prevent. Fixed in
`quota.quota_day()`, which uses `ZoneInfo` rather than a fixed offset because Pacific shifts an
hour with daylight saving.

### 4.2 Per-minute pacing — `TokenBucket`

In-process, dependency-free, and **injected with a clock**:

```
TokenBucket(capacity, refill_per_sec, now=time.monotonic)
  .take(n) -> bool
```

No test sleeps (G5). Tests advance a fake clock and assert on the boundary. Losing bucket
state on restart is correct here — a restart genuinely has not sent anything recently.

---

## 5. Components

| Component | Contract | New? |
|---|---|---|
| `backends/tier2_gemini.py` | `Tier2Gemini.repair(segment, hypotheses) -> str` | new |
| `dhvani/quota.py` | `TokenBucket`, `QuotaGate` | new |
| `Store.reserve_quota` | `(tier, day, n) -> None`, raises `QuotaExhausted` | new |
| `Store.quota_used` | `(tier, day) -> int` | new |
| `dhvani/repair.py` | select candidates, gate, submit, mark | new |
| `router.plan` | unchanged — already tier-agnostic | no change |
| `reconcile` | unchanged — already keyed by `(segment_id, tier)` | no change |

`router.plan()` needs no modification, and this is worth stating precisely because it is
stronger than it sounds. Tier 2 candidates carry `cost_usd = 0.0`, and a zero-cost candidate is
the one case a ratio-greedy selector normally breaks on — `delta / cost` divides by zero.

**That case is already handled.** `_ratio()` returns `math.inf` for `cost_usd <= 0.0`, so a
free positive-delta candidate ranks ahead of everything priced, with ties broken on
`segment_id` then `tier` for determinism (I5). This was not written for Tier 2: FIX ROUND 3
(C2) introduced it because replay mode prices every candidate at zero and the old
`cost_usd > 0.0` filter made `--mode replay` unable to escalate at all.

So the router requires **zero changes** to support a free tier, and the policy it already
implements — free improvement is strictly the best thing to buy, take it first — is the right
one for Tier 2 by coincidence rather than by design. The implementation plan must not include
a step for this; it exists and is tested.

### 5.1 New storage

```sql
CREATE TABLE quota (
  tier       TEXT NOT NULL,
  day        TEXT NOT NULL,          -- UTC date, YYYY-MM-DD
  used       INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX idx_quota_tier_day ON quota (tier, day);
```

Append-only, like `spend`. Summing rows rather than updating one keeps the reservation a
single INSERT-guarded-by-SELECT and avoids a read-modify-write.

### 5.2 Marking an unrepaired segment

The band contract (§6.2) says nothing ships silently. A segment whose repair was skipped for
quota carries `repair_unavailable` in its track entry. This is **not** a new band — the band
still reflects risk. It is a separate field, because "we did not get to improve this" is a
different fact from "this is risky".

---

## 6. Cost control

Tier 2 is free, so `cost_usd = 0.0` is recorded in `hypotheses` and in the spend ledger.

**The ledger stays truthful about money and gains nothing false.** Quota is tracked in its own
table because it is a genuinely different resource: dollars accumulate against a lifetime
ceiling, requests reset daily. Modelling quota as a fake dollar amount would corrupt
`total_spend()`, which guards the USD 20 ceiling and is asserted by existing tests.

`Recorded` still wraps Tier 2 for record/replay, so `make test` never reaches the network and
a missing fixture is still a hard `FixtureMissing` (G4). Replay never falls back to live.

---

## 7. Testing

| Property | How |
|---|---|
| Daily cap never exceeded | reserve N times at the cap, assert N+1 raises `QuotaExhausted` |
| Concurrent reservation is atomic | two Store handles, same DB, same day — total never exceeds cap |
| Reservation survives restart | reserve, drop the handle, reopen, assert count persists |
| Exhaustion degrades | run with cap 0; assert exit 0, track produced, segment marked |
| Exhausted work is retried | reconcile on a later day, assert repair lands in v(n+1) |
| Rate limiting without sleeping | fake clock, assert bucket boundary exactly |
| Zero-cost ranking | free candidate with positive delta outranks a priced one |
| Determinism (I5) | same inputs, same plan, same job id |
| Idempotent merge (I2) | apply the same Tier 2 result twice, version does not advance |
| No live call in replay | tripwire on the client factory, as `tier1_chirp` already does |
| G4 holds | `google-genai` absent, suite green |

The chaos suite gains Tier 2 as a second async tier under the existing fault injections
(partial delivery, reordering, duplication, transient error).

---

## 8. Risks and open questions

| Risk | Mitigation |
|---|---|
| ~~Model id is unverified~~ | **Spike 2026-09-02: namespace confirmed, choice still open.** Current stable ids are `gemini-3.7-flash`, `3.6-flash`, `3.5-flash`, `2.5-flash`, `2.5-flash-lite` and siblings; `gemini-2.0-flash` is **shut down**. Guessing would have produced a retired id. Still read from `GEMINI_MODEL`; pick a Flash-class model when a key exists. |
| ~~Reset boundary assumed UTC~~ | **Spike 2026-09-02: CLOSED, and the assumption was wrong in the dangerous direction.** Midnight Pacific, per project. See §4.1. |
| **The daily cap is not knowable from documentation.** Google no longer publishes free-tier RPD/RPM; they are per-project facts visible only in AI Studio, and were reportedly cut sharply in Dec 2025. | Treat `GEMINI_DAILY_QUOTA` as a LOCAL ceiling, not the vendor's. Being wrong high is now safe: `repair()` treats the vendor's own refusal as degradation, not a crash. Set `--repair-quota` from AI Studio when a key exists. |
| Repair makes text worse | This is what I3 is for. The measured delta decides, and a negative tier2 delta correctly disables the tier — the Tier 1 result already demonstrates the harness reporting an unwelcome answer honestly. |
| LLM output is nondeterministic, and I5 requires determinism | Determinism is required of the **plan**, not the vendor. Temperature pinned to 0 and the response cached content-addressed; replay makes tests fully deterministic. The plan is deterministic regardless of what the model returns. |
| Prompt becomes an untracked variable | The prompt is part of `variant_key`, so changing it invalidates fixtures rather than silently altering cached results. Bumping it is a `POLICY_ID` change. |
| Free tier changes or disappears | N4 stands: the tier degrades to disabled, which §3 already handles as the exhausted path. |

**Open question deferred to the plan:** whether the smoke run repairs Tier 0 output from
`results/scored.json` directly (cheapest, reuses existing data) or re-runs a short pipeline.
The former is preferred and is the assumption unless the spike contradicts it.
