# Dhvani Calibration Harness — Design

**Status:** Design approved, pending implementation plan
**Date:** 2026-08-24
**Author:** Sankalp Badrinath
**Implements:** the missing half of `2026-08-21-dhvani-design.md` §6 — `delta_table.json`,
deferred from Phase 1 and again from Phase 2

---

## 1. Problem

`dhvani/delta_table.py::build(rows)` exists and is tested. Nothing produces its `rows`.

Without a measured table, `router.delta_for()` returns `0.0` for every bucket, so
`router.plan()` selects nothing and `escalate()` returns `None`. The routing policy — the
project's headline component — is inert. The frontier report renders, but every row reads zero
escalations.

This harness produces the table.

### 1.1 The circularity that shapes the whole design

**The router cannot choose which segments get Chirp.** It selects by `delta`, and `delta` is
precisely what is being measured. Calibration must escalate a **stratified sample across every
risk bucket**, deliberately including low-risk ones — that is the only way to discover the
*negative* deltas that invariant I3 exists to filter out. A run that escalates only high-risk
segments yields a table that can never tell the router when escalation makes things worse.

### 1.2 The alignment problem

`segmenter.segment()` splits audio into 2-8s VAD chunks. Dataset references are **per
utterance**. Splitting a 30s utterance into five chunks leaves one reference for five segments,
making `to_wer(reference, tier0_text)` meaningless and silently poisoning every delta.

**Resolution: calibration bypasses the segmenter.** Each dataset utterance is one segment.
`segment_id` remains `SHA256(normalized PCM)`, so the cache, store, and fixture layers are
unchanged.

**Accepted caveat, stated rather than hidden:** calibration segments are utterance-shaped while
production segments are VAD-shaped, so their duration and risk distributions may differ. The
measured deltas carry that assumption. Forced alignment would remove it, at the cost of a large
subsystem for a second-order gain. Out of scope.

---

## 2. Goals and non-goals

### Goals
- G1. Produce a committed `delta_table.json` measured on real Indic speech.
- G2. Show the risk histogram **before** any paid call.
- G3. Total external spend for a full run ≤ USD 10 (expected ~$0.75), inside the USD 20 ceiling.
- G4. Re-runnable when `RISK_WEIGHTS` changes; a re-run over unchanged segments costs nothing.
- G5. The test suite still runs with **no** ML dependencies, **no** cloud SDK, **no**
  credentials, and **no** network.

### Non-goals
- N1. No model training. Measurement only.
- N2. No forced alignment (see §1.2).
- N3. No staleness *enforcement*. The table records what it was measured under; nothing
  refuses to run on a mismatch. Explicitly declined.
- N4. No new metric. `to_wer` is used as-is, limitations included (spec §1.3.1).

---

## 3. Architecture

```
PHASE 1 — collect          (slow, free, local)
  IndicVoices stream (kn-IN, ml-IN, hi-IN)
      -> audio.normalize          -> segment_id = SHA256(pcm)
      -> Tier0Conformer           -> text + ctc_rnnt_disagreement
      -> scorer.extract / risk
      -> Store: segment + tier0 hypothesis + reference
  emits: risk histogram + per-bucket counts   <- inspect BEFORE spending

PHASE 2 — escalate         (fast, paid, remote)
  stratified sample across ALL 10 risk buckets
      -> upload only those segments to GCS
      -> Tier1Chirp, DYNAMIC_BATCHING
      -> Store: tier1 hypothesis
      -> rows {risk, reference, tier0_text, tier1_text}
      -> delta_table.build  -> delta_table.json
```

Phases are separate commands. The separation is load-bearing: the slow-and-free part is
decoupled from the fast-and-paid part, and the risk distribution is visible before any spend.
Each phase resumes independently via the content-addressed cache.

---

## 4. Components

| File | Responsibility | External deps |
|---|---|---|
| `dhvani/corpus.py` | stream IndicVoices -> `CorpusItem(pcm, reference, speaker_id, district, lang)` | `datasets` (optional extra) |
| `dhvani/calibrate.py` | both phases, stratification, row assembly | none — all injected |
| `dhvani/store.py` (modify) | + `references` table, `put_reference`, `get_reference` | none |
| `dhvani/cli_calibrate.py` | `collect` / `escalate` subcommands | wires the real deps |

**Injection rule.** `calibrate.py` constructs no external dependency. The corpus source, the
Tier 0 backend, and the Tier 1 backend all arrive as parameters. The CLI wires real ones; tests
wire fakes. This is the same seam that keeps the existing 259 tests running without `torch` or
the cloud SDK, and it is what preserves G5.

### 4.1 New storage

The Store has `segments`, `hypotheses`, `spend`, `jobs`, `tracks` — nowhere for ground truth.

```sql
CREATE TABLE IF NOT EXISTS references_ (
  segment_id  TEXT PRIMARY KEY,
  reference   TEXT NOT NULL,
  lang        TEXT NOT NULL,
  speaker_id  TEXT,
  district    TEXT,
  created_at  INTEGER NOT NULL
);
```

Named `references_` because `references` is a SQL reserved word. Empty in production;
populated only by calibration. `put_reference` is `INSERT OR IGNORE`, matching the idempotency
pattern the rest of the schema already uses.

---

## 5. Stratified sampling

### 5.0 Collected is not escalated

Two different numbers, easily conflated:

| | count | cost |
|---|---|---|
| Phase 1 **collects** | ~1000 per language x 3 = **~3000** | free, but hours of local compute |
| Phase 2 **escalates** | <= 100 per bucket x 10 = **<= 1000** | ~$0.75 at 15s billing rounding |

Only the stratified subset is ever sent to Chirp. At the authorized USD 10 that leaves roughly
**13x headroom**.

Spend that headroom on Phase 1, not Phase 2. If buckets come back thin, the fix is collecting
*more* segments (free, slow) so the buckets fill — not escalating more per bucket (paid, and
already statistically sufficient at 100). Raising `N_PER_BUCKET` only helps if the buckets
actually contain that many segments.

### 5.1 The sampling rule

Target **N = 100 segments per risk bucket**, 10 buckets, capped by what Phase 1 produced.
Selection within a bucket is deterministic — sorted by `segment_id`, seeded — so a re-run picks
the same segments and hits the cache rather than re-paying.

### 5.2 The distribution is the unknown

`ctc_rnnt_disagreement` carries weight 0.4667, by far the largest. On clean read speech the two
decoder heads will largely agree, so risk is expected to skew hard toward low buckets, leaving
`0.7-0.8` and above sparse or empty.

That is a finding, not a defect to engineer around. If production traffic shares the
distribution, those buckets are empty there too. What matters is that Phase 1 reports the
histogram before Phase 2 spends.

### 5.3 Bucket guards

- **Empty bucket** -> absent from the table -> `delta_for()` returns `0.0` -> the router never
  escalates it. Safe by construction; this is already `delta_for`'s behaviour.
- **Thin bucket** (n < `MIN_BUCKET_SAMPLES = 20`) -> **omitted, not included noisily.**
  Omission degrades to "do not escalate". A noisy average degrades to "escalate wrongly, and
  pay for it."

### 5.4 Table contents

`build()`'s existing contract is untouched. The written JSON wraps it with provenance:

```json
{
  "tier1": {"0.3-0.4": 4.2, "0.6-0.7": 18.1},
  "meta": {
    "policy_id": "p1-2026-08-22",
    "risk_weights": {"ctc_rnnt_disagreement": 0.4667, "...": "..."},
    "bucket_n": {"0.3-0.4": 143, "0.6-0.7": 61},
    "languages": ["kn-IN", "ml-IN", "hi-IN"],
    "segments_escalated": 812,
    "spend_usd": 0.6090,
    "measured_at": "2026-08-24"
  }
}
```

Nothing enforces `meta` (non-goal N3), but a stale table becomes visible rather than silent.
`delta_for()` reads `table["tier1"]` and ignores `meta`.

---

## 6. Cost control

Three independent guards, because this project has had five distinct defects in its spend path:

1. `reserve_spend` — atomic check-and-insert against the USD 20 ceiling. Unchanged.
2. **Pre-flight estimate** — Phase 2 prints `segments x cost_for_duration_ms` and requires
   `--confirm` before its first paid call.
3. `--dry-run` — stratify, print the histogram and estimate, exit before any paid call.

A calibration run that silently spends is the failure mode being designed out.

**Partial-failure rule:** if `BudgetExceeded` raises mid-run, `delta_table.json` is **not
written**. A partial table is worse than none, because the router would trust it.

---

## 7. Testing

Everything external is injected, so no test touches the network, a model, or a paid API.

| Unit | Nature | Approach |
|---|---|---|
| `stratify(segments, n_per_bucket, seed)` | pure | determinism, cap, thin-bucket omission, empty-bucket absence |
| row assembly | pure | shape and ordering of `{risk, reference, tier0_text, tier1_text}` |
| `references_` table | SQLite | re-insert is a no-op |
| Phase 1 collect | I/O | `FakeCorpus` + `StubTier0`; no download, no model |
| Phase 2 escalate | paid | `Recorded` in replay mode against committed fixtures |
| Cost gate | money | `BudgetExceeded` mid-run leaves the table unwritten |
| Split discipline | correctness | speaker- and district-disjointness asserted on the **selected** set |

### 7.1 Two properties pinned explicitly

- **Resumability.** Kill Phase 1 halfway, restart, assert **zero** re-transcription for
  already-seen segments. This is what makes a multi-hour run survivable.
- **Idempotent spend.** Re-running Phase 2 over the same stratified sample hits cached `tier1`
  hypotheses and reserves **nothing** further.

---

## 8. Operational notes

- **Region.** Tier 1 runs `chirp_2` in `europe-west4`; `asia-south1` offers no usable chirp
  model (main design doc, `2026-08-21-dhvani-design.md` §14.2). Indian-language audio is therefore transcribed outside India — a
  data-residency point to state, not discover.
- **The `models` extra must be reinstalled** for Phase 1 (Tier 0 needs the real model), and
  should be removed afterwards so G5 stays demonstrable. G5 is verified before and after, not
  during.
- **GCS bucket** is created by Phase 2 and may be deleted after; uploaded audio is a
  reproducible derivative of the corpus.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Risk distribution too skewed to populate mid buckets | Phase 1 histogram reveals it before any spend; widen the corpus or revisit `RISK_WEIGHTS` |
| Tier 0 runtime dominates (hours on CPU/MPS) | Resumable by construction; Phase 1 can be run in slices |
| IndicVoices utterances longer than expected, raising per-segment billing | Pre-flight estimate uses real durations, not assumptions |
| `to_wer`'s loanword blindness (main design doc §1.3.1) understates true improvement | Documented; deltas are directionally valid, magnitudes conservative |
| Chirp returns empty text for some segments | Treated as a legitimate Tier 1 result — a large *negative* delta, which is exactly what I3 must filter |

---

## 10. Open questions

- `MIN_BUCKET_SAMPLES = 20` is a judgement call, not a derived figure. If Phase 1 shows a
  narrow distribution, a lower floor may be warranted to populate more buckets.
- Whether to keep the GCS bucket between runs. Keeping it makes re-runs faster; deleting it
  keeps the footprint clean. Deferred to first use.
