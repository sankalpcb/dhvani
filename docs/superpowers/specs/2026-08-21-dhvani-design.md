# Dhvani — A Confidence-Gated Caption Cascade for Code-Mixed Indic Speech

**Status:** Design approved, pending implementation plan
**Date:** 2026-08-21
**Author:** Sankalp Badrinath
**Context:** Portfolio project targeting Software Engineer, YouTube (Bengaluru)

---

## 1. Problem

### 1.1 Code-mixing is the native register, not an edge case

A large share of Indian speech is *intra-sententially code-mixed*:

> मैंने उस **bug** को **fix** कर दिया, लेकिन **deployment** अभी **pending** है

Under Myers-Scotton's Matrix Language Frame model, one language (here Hindi) supplies the
morphosyntactic frame — word order, case marking, verb morphology — while the other (English)
contributes content words slotted into that frame. This is rule-governed, not noise.

This matters at scale: YouTube India has ~500M users, its largest market by user count, and
roughly 90% of consumption is in regional languages. Code-mixed audio is not a long tail.

### 1.2 Four distinct ASR failure modes

Conflating these is why the problem stays open:

1. **Language-ID is decided once per utterance.** The API takes a single language code
   (`hi-IN`, `ml-IN`). That choice is wrong for every embedded English span.
2. **Script rendering ambiguity.** In `ml-IN` mode, "deployment" has two defensible
   renderings: Latin `deployment` or Malayalam `ഡിപ്ലോയ്‌മെന്റ്`. Models pick
   unpredictably, sometimes varying *within a single utterance*. This is a publicly
   reported Chirp 3 defect (see References).
3. **Language-model bias toward monolinguality.** Code-mixed strings are low-probability
   under both monolingual LMs, dragging decoding toward a hallucinated monolingual reading.
4. **Acoustic mismatch.** English spoken with Indian phonology sits off-distribution for
   both English and Indic acoustic models.

### 1.3 WER actively hides the failure

`WER = (S + D + I) / N` treats all substitutions as equal cost. Given reference
`deployment अभी pending है`:

| Hypothesis | Edits | Meaning | WER |
|---|---|---|---|
| `ഡിപ്ലോയ്‌മെന്റ് अभी pending है` | 1 sub | intact | identical |
| `डिब्बा अभी pending है` ("box is still pending") | 1 sub | destroyed | identical |

WER simultaneously **over-penalizes benign script variance** and **under-penalizes semantic
corruption**. Prior art addresses this with `toWER`, `MER`, and `TER`. We adopt `toWER`; we
do not claim to invent it.

### 1.4 The product failure this project actually attacks

YouTube auto-captions average ~60-70% accuracy in the wild. The system has internal
confidence signals. **The viewer receives none of them.** A caption renders identically
whether the model is 99% or 12% confident.

For a deaf or hard-of-hearing viewer this is the core harm: they cannot distinguish a
correct caption from a confidently wrong one. A hearing viewer catches ASR errors from the
audio underneath; a deaf viewer has no such channel. The error is invisible to exactly the
person who depends on it.

**Thesis:** the goal is not better ASR. It is *knowing when the ASR is wrong, spending
expensive compute only there, and never silently shipping an untrusted caption.*

The accessibility argument and the cost argument are the same argument: you cannot run an
expensive model over 720,000 hours/day of uploads, but you can run one over the ~15% of
segments where a cheap model is uncertain.

---

## 2. Goals and non-goals

### Goals
- G1. Match a full-Chirp caption quality baseline at materially lower cost per audio-hour.
- G2. Never ship a caption below a configured quality threshold without marking it.
- G3. Provably lose no data across async escalation under injected failure.
- G4. Stay under **USD 20** total external spend.
- G5. Full test suite runnable by a stranger with no cloud credentials.

### Non-goals
- N1. Training or fine-tuning any model. This is a systems project.
- N2. Beating published WER baselines. We compete on cost and correctness, not modelling.
- N3. Standing up production GCP infrastructure. We ship a written scaling sketch instead.
- N4. Real-time/live captioning. Async batch is the target.

---

## 3. Architecture

```
audio
  │
  ▼
┌─────────────┐   VAD split into caption-sized segments (2-8s)
│  segmenter  │   segment_id = SHA256(normalized PCM)
└─────────────┘
  │
  ▼
┌─────────────┐   IndicConformer 600M, local. Cost: $0.
│  asr_tier0  │   Runs on 100% of segments.
└─────────────┘
  │
  ▼
┌─────────────┐   deterministic risk in [0,1]. No model artifact.
│   scorer    │
└─────────────┘
  │
  ▼
┌─────────────┐   ← BUDGET B (USD per audio-hour)
│   router    │   PURE FUNCTION. Emits escalation plan.
└─────────────┘
  │
  ├───────────── risk < tau_ship ─────────────────────────┐
  │                                                       │
  ▼                                                       │
┌─────────────┐  Chirp 3 dynamic batch, $0.003/min        │
│  asr_tier1  │  ASYNC — up to 24h turnaround             │
└─────────────┘                                           │
  │                                                       │
  ▼                                                       │
┌─────────────┐  Gemini script normalization +            │
│ repair_tier2│  code-mix restoration. Free tier.         │
└─────────────┘                                           │
  │                                                       │
  ▼                                                       ▼
┌────────────────────────────────────────────────────────────┐
│ quality gate → versioned caption track                     │
│                + per-segment confidence                    │
│                + flagged-for-review list (never silent)    │
└────────────────────────────────────────────────────────────┘
```

### 3.1 Why content-addressing is load-bearing

`segment_id = SHA256(normalized PCM)` means the same audio is never transcribed twice.
This is simultaneously:
- the correct production design (intros, music beds, stock B-roll repeat across videos),
- the mechanism that makes idempotent merges trivial (§6),
- and what keeps the project inside its USD 20 budget across dozens of iterations.

Normalization before hashing: resample to 16kHz mono PCM16, strip container metadata.
Normalization must be deterministic and version-pinned; a change to it is a cache-invalidating
breaking change and must bump `POLICY_ID`.

---

## 4. Component contracts

| Component | Contract | Depends on | Network |
|---|---|---|---|
| `segmenter` | `audio -> [(segment_id, t0, t1, pcm)]` | Silero VAD | no |
| `asr_tier0` | `segment -> (text, signals)` | IndicConformer 600M | no |
| `asr_tier1` | `[segment] -> job_handle`; `job_handle -> [text]` | Chirp 3 STT v2 | yes, async |
| `repair_tier2` | `(segment, [hypothesis]) -> text` | Gemini | yes, batched |
| `scorer` | `(text, signals) -> risk in [0,1]` | config only | no |
| `router` | `(risks, budget, delta_table) -> plan` | nothing | **no — pure** |
| `store` | content-addressed cache + job state + spend ledger | SQLite | no |
| `evaluator` | `(hyps, refs) -> {WER, toWER, MER, risk-coverage, cost}` | jiwer | no |

`router` and `scorer` are pure functions. They hold the most interesting logic and are
trivially unit-testable, which makes TDD viable for the core of the system.

---

## 5. The risk function

Deterministic, weights fixed by a one-time offline grid search, committed as config:

```
risk = w1 * ctc_rnnt_disagreement     # dual-head disagreement, free ensemble signal
     + w2 * mean_neg_logprob
     + w3 * script_mix_entropy        # Unicode script-block entropy within segment
     + w4 * romanization_smell        # tokens valid in neither lexicon
     + w5 * short_segment_indicator   # duration < 1.5s
```

**No model artifact, no training loop.** A config file, a pure function, and a measured
`delta_table` checked into the repo.

### 5.1 On `ctc_rnnt_disagreement`

IndicConformer is a hybrid CTC-RNNT model: two decoders over one shared encoder.
Disagreement between heads is ensemble uncertainty at zero extra compute — expected to be
the strongest single signal.

**Verify on day one that both heads are exposed by the inference API.** If only one is,
this feature is unavailable and the remaining weights must be re-fit. Treat as a
day-one spike, not an assumption.

### 5.2 On `script_mix_entropy`

Entropy over Unicode script blocks within a segment. A hypothesis flipping script repeatedly
inside one utterance is the direct fingerprint of failure mode 1.2(2). Costs microseconds.

---

## 6. Routing policy

Budget-constrained escalation, i.e. knapsack:

```
maximize   sum over selected (segment, tier) of delta_t(risk_i)
subject to sum of cost_t <= B
```

Solved greedily by `delta / cost` descending. `delta_t(risk_bucket)` is **measured
empirically once** against a held-out calibration split and committed as `delta_table.json`:

```json
{"tier1": {"0.0-0.1": 0.4, "0.1-0.2": 1.9, "0.6-0.7": 18.2}}   // abbreviated; all 10 buckets present
```

Values are toWER points reduced, per risk bucket.

### 6.1 Graceful degradation

- `B = 0` → pure Tier 0 with heavy flagging. System still produces output.
- `B` large → approaches full-Chirp quality.
- The policy adapts to whatever budget it is handed. This is the headline demo.

### 6.2 Output bands

| Risk | Action |
|---|---|
| `risk < tau_ship` | ship normally |
| `tau_ship <= risk < tau_flag` | ship **with visual uncertainty marking** |
| `risk >= tau_flag` | do not ship silently; surface to creator for review |

`tau_ship` is chosen from the risk-coverage curve to hit a target selective risk
(e.g. shipped captions have `toWER <= 10%`). The middle band is where the accessibility
contribution becomes concrete.

---

## 7. Async reconciliation

Chirp dynamic batch has up to 24h turnaround, so escalation is asynchronous:

1. `t=0` — publish track v1 from Tier 0, low-confidence segments flagged.
2. `t=0` — submit escalation batch; persist `job_handle` and member `segment_id`s.
3. `t=+Nh` — results arrive, possibly out of order, possibly partial, possibly duplicated.
4. Merge by `segment_id`; bump track version; republish.

Idempotency falls out of the schema: `hypotheses` is keyed `(segment_id, tier)`, so
re-applying a batch result is a no-op at the database level rather than in application logic.

---

## 8. Storage schema (SQLite for POC)

```sql
CREATE TABLE segments (
  segment_id   TEXT PRIMARY KEY,          -- SHA256(normalized PCM)
  source_id    TEXT NOT NULL,
  t_start_ms   INTEGER NOT NULL,
  t_end_ms     INTEGER NOT NULL,
  duration_ms  INTEGER NOT NULL,
  lang_hint    TEXT,
  created_at   INTEGER NOT NULL
);

CREATE TABLE hypotheses (
  segment_id   TEXT NOT NULL,
  tier         TEXT NOT NULL,             -- tier0 | tier1 | tier2
  text         TEXT NOT NULL,
  signals_json TEXT NOT NULL,
  cost_usd     REAL NOT NULL,
  created_at   INTEGER NOT NULL,
  PRIMARY KEY (segment_id, tier)          -- makes merges idempotent
);

CREATE TABLE jobs (
  job_id       TEXT PRIMARY KEY,
  tier         TEXT NOT NULL,
  state        TEXT NOT NULL,             -- pending | running | done | failed
  segment_ids  TEXT NOT NULL,             -- JSON array
  submitted_at INTEGER NOT NULL,
  settled_at   INTEGER,
  attempts     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE tracks (
  source_id    TEXT NOT NULL,
  version      INTEGER NOT NULL,
  policy_id    TEXT NOT NULL,
  content_json TEXT NOT NULL,
  to_wer       REAL,                      -- measured toWER, null until evaluated
  cost_usd     REAL NOT NULL,
  created_at   INTEGER NOT NULL,
  PRIMARY KEY (source_id, version)
);

CREATE TABLE spend (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  tier         TEXT NOT NULL,
  cost_usd     REAL NOT NULL,
  created_at   INTEGER NOT NULL
);
```

---

## 9. Evaluation

### 9.1 Metrics (systems-first)

| Metric | Why |
|---|---|
| Cost per audio-hour at each quality target | the headline claim |
| Cache hit rate | dedup effectiveness, the cost lever at scale |
| Throughput (audio-hours per wall-clock hour, single machine) | capacity |
| p50/p99 latency per tier | serving behaviour |
| Quota-exceeded failure count under sustained load | must be zero |
| Lost/duplicated segments across 1000 injected failures | must be zero |
| Escalation rate | policy behaviour |
| `toWER` on shipped captions | the one quality number, proves the cost claim means something |

### 9.2 Headline claim format

> Matches full-Chirp caption quality at **N× lower cost per audio-hour**, with zero data
> loss across 1000 injected failures.

### 9.3 Datasets

All CC-BY-4.0 or MIT, from HuggingFace. No scraping, no ToS exposure.

- **IndicVoices** — 7,348h spontaneous speech, 1,639h transcribed, 22 languages,
  16,237 speakers, 145 districts. Contains natural code-switching.
- **Kathbath** — 1,684h, 12 languages, professionally labeled.
- **Vistaar** — 59 benchmarks across 12 languages. Used to situate our numbers against
  published baselines, not to beat them (see non-goal N2).

**Languages:** Kannada (role is Bengaluru-based), Malayalam (documented Chirp 3 defect),
Hindi (largest corpus, strongest baselines).

### 9.4 Split discipline

Splits MUST be **speaker-disjoint and district-disjoint**. IndicVoices spans 16,237 speakers
across 145 districts; if a speaker appears in both fit and evaluation splits, the risk
function learns speaker identity as a difficulty proxy and all reported numbers are fiction.

---

## 10. Test strategy

### 10.1 Record/replay

```
DHVANI_MODE=record   call live API, write fixtures/{backend}/{segment_id}.json
DHVANI_MODE=replay   read fixture; HARD FAIL if missing
DHVANI_MODE=live     call live API, no fixture write
```

Two rules, both fail-closed:
- **Replay never falls back to live on a cache miss.** Silent fallback is how a test run
  quietly spends money.
- **The live client enforces a hard spend ceiling.** Cumulative cost is read from the
  `spend` ledger; any call that would exceed `MAX_SPEND_USD` is refused.

### 10.2 Pyramid

| Layer | Cost | Frequency |
|---|---|---|
| Unit — router, risk fn, script entropy, segment_id stability, merge idempotency | $0 | constant |
| Contract — adapters conform to `Backend` | $0 replay | per commit |
| Integration — full pipeline on fixtures | $0 | CI |
| Chaos — `ChaosBackend` injects timeouts, 429, 500, partial batch, duplicate delivery, reorder | $0 | CI |

### 10.3 Invariants

- **I1 · No loss** — every input segment appears exactly once in the final track.
- **I2 · Idempotent merge** — applying a batch result twice is a no-op.
- **I3 · No negative-value escalation** — the router never escalates a segment whose risk
  bucket has measured `delta <= 0`.
  *Note:* the naive form ("escalation always improves quality") is false — Tier 1 loses to
  Tier 0 on some segments. The bucket-level formulation is the honest one.
- **I4 · Budget respected** — total spend never exceeds configured budget.
- **I5 · Determinism** — same input + same `POLICY_ID` produces byte-identical output.
- **I6 · Convergence** — once all async jobs settle, the track equals what a fully
  synchronous pipeline would have produced.

### 10.4 CI

GitHub Actions runs the full unit + integration + chaos suite in replay mode: no
credentials, no cloud account, no spend. Fixtures are committed. Live contract tests live
in a separate, manually-triggered workflow.

Consequence: a reviewer can clone the repo and get a green suite with zero setup.

---

## 11. Budget

| Set | Size | Cost | Frequency |
|---|---|---|---|
| Smoke | 20 segments (~2 min) | ~$0.006 | constant |
| Dev | 500 segments (~40 min) | ~$0.12 | per change |
| Benchmark | 3 langs × 3h | ~$1.62 | ~5 runs |
| Held-out final | 3h | ~$0.54 | **exactly once** |

**Projected total: ~$10-13.** After the first pass everything is fixtured; re-runs cost $0.

**Rates and gotchas (verified 2026-08-21):**
- Speech-to-Text v2 **dynamic batch: $0.003/min** (24h turnaround). Standard tier is
  $0.016/min — do not use it for benchmarking.
- Free tier: 60 STT minutes/month, permanent.
- The $300 / 90-day trial credits **do** cover Speech-to-Text.
- **Gemini API is excluded from the $300 trial credits** (as of March 2026). It has its own
  free tier of 1,000 requests/day, which is sufficient here.
- **Free-trial billing accounts cannot attach GPUs to VMs.** Tier 0 runs locally or on
  free Colab.

**Day-one verification:** confirm Chirp 3 supports the dynamic-batch processing strategy.
If it is standard-tier only, benchmark cost rises to ~$14.40 — still inside $20, but the
margin is gone.

---

## 12. Production scaling sketch (written, not built)

Deliverable is an architecture document with capacity and cost math at 720,000 audio-hours/day:

| POC | Production |
|---|---|
| SQLite `segments`/`hypotheses` | Bigtable keyed by `segment_id` |
| In-process queue | Pub/Sub for the escalation queue |
| `jobs` table | Cloud Tasks |
| Local invocation | Cloud Run (scale-to-zero) for the API |
| Local files | GCS for audio and fixtures |
| `make bench` | Dataflow for batch reprocessing |

Include: shard-key analysis for `segment_id` (uniform by construction — SHA256), hot-key
behaviour for popular deduplicated segments, and cost-per-million-audio-hours at several
budget settings.

---

## 13. Milestones

| # | Deliverable | Demo |
|---|---|---|
| M0 | Skeleton, store, segmenter, `segment_id` stability tests | hashes are stable across runs |
| M1 | Tier 0 + record/replay + spend ceiling | first captions produced, $0 on re-run |
| M2 | Risk function + evaluator (`toWER`) | first quality numbers |
| M3 | Router + `delta_table` | first cost/quality frontier plot |
| M4 | Async Tier 1 + reconciliation + chaos suite | I1-I6 green under injected failure |
| M5 | Tier 2 repair + quota-aware rate limiter | graceful degradation at quota exhaustion |
| M6 | `make bench` report + scaling sketch | the resume artifact |

M4 is the differentiating milestone. If time runs short, cut M5 before M4.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| IndicConformer does not expose both CTC and RNNT heads | Day-one spike. If absent, re-fit weights without that feature; document the loss. |
| Chirp 3 lacks dynamic-batch support | Day-one spike. Fallback: standard tier, ~$14.40, still inside budget. |
| Chirp outperforms the cascade at every budget point | This is a legitimate finding, not a failure. Report it honestly; the systems contribution (dedup, reconciliation, chaos suite) stands independently. |
| Scope creep into ML | Non-goal N1 is binding. No training loops. |
| Held-out set contamination | Run exactly once, at M6. Enforced by a checked-in run counter. |

---

## 15. Open questions

- Which IndicConformer variant: the 600M multilingual model, or per-language checkpoints?
  Multilingual is simpler to operate; per-language may score better. Decide at M1 by
  measurement, not preference.
- Should `tau_ship` be global or per-language? Start global; revisit at M3 if per-language
  risk distributions diverge materially.

---

## 16. References

- Myers-Scotton, *Matrix Language Frame* model — code-mixing structure
- Das & Gambäck (2014), Code-Mixing Index (CMI)
- Google Research, *Transliteration based approaches to improve code-switched speech
  recognition performance* — `toWER` prior art
- AI4Bharat: IndicVoices (arXiv:2403.01926), Kathbath, Vistaar, IndicConformer
- Chirp 3 `ml-IN` code-mixing defect report:
  https://discuss.google.dev/t/chirp-3-ml-in-unpredictable-code-mixing-english-words-transliterated-to-malayalam-script-and-vice-versa/388859
- Google Cloud Speech-to-Text v2 pricing (dynamic batch tier)
