import json

import pytest

from dhvani.calibrate import MIN_BUCKET_SAMPLES
from dhvani.cli_calibrate import main
from dhvani.config import MAX_SPEND_USD
from dhvani.store import BudgetExceeded, Store


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_collect_subcommand_exists():
    with pytest.raises(SystemExit) as exc:
        main(["collect", "--help"])
    assert exc.value.code == 0


def test_escalate_subcommand_exists():
    with pytest.raises(SystemExit) as exc:
        main(["escalate", "--help"])
    assert exc.value.code == 0


# The Tier 0 cache key collect() ACTUALLY writes — Tier0Conformer.variant_key
# for the default Hindi configuration. Seeding hypotheses under "" (which two
# of these tests used to do) matched the C1 bug rather than production: the
# CLI looked up "" too, so the tests passed while every real run skipped
# every segment. Anything that seeds a Tier 0 hypothesis for the CLI to find
# must use this.
TIER0_VARIANT = "lang=hi;model_id=ai4bharat/indic-conformer-600m-multilingual"


def _seed_scored(tmp_path, n=25, risk=0.65, prefix="s"):
    """A scored.json with one populated bucket, so escalate reaches its cost
    gate instead of exiting early on a missing input file."""
    scored = _seed_scored_items(n=n, risk=risk, prefix=prefix)
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))
    return str(path)


def _seed_scored_items(n=25, risk=0.65, prefix="s"):
    return [{"segment_id": f"{prefix}{i:04d}" + "0" * 59, "risk": risk,
             "lang": "hi-IN", "duration_ms": 3000,
             "tier0_variant": TIER0_VARIANT} for i in range(n)]


def test_escalate_without_confirm_refuses_to_spend(tmp_path, capsys):
    """The cost gate: a run that would spend must not do so silently."""
    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", _seed_scored(tmp_path),
               "--out", str(tmp_path / "d.json")])
    out = capsys.readouterr()
    assert rc == 2, "must exit non-zero specifically on the confirm gate"
    assert "--confirm" in (out.out + out.err)


def test_missing_scored_input_is_a_different_failure(tmp_path, capsys):
    """Distinguishes 'no input' from 'refused to spend' — otherwise the
    confirm-gate test would pass without ever reaching the gate."""
    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", str(tmp_path / "absent.json"),
               "--out", str(tmp_path / "d.json")])
    assert rc == 1
    assert "collect" in capsys.readouterr().err


def test_dry_run_writes_no_table(tmp_path):
    path = tmp_path / "d.json"
    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", _seed_scored(tmp_path),
               "--out", str(path), "--dry-run"])
    assert rc == 0
    assert not path.exists()


def test_pyproject_declares_the_calibrate_script():
    import pathlib
    import tomllib
    root = pathlib.Path(__file__).resolve().parent.parent
    with open(root / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["scripts"]["dhvani-calibrate"] == "dhvani.cli_calibrate:main"


# --- Addition 1: budget failure must leave no partial table behind ---
#
# Moved here from Task 5's brief (see tests/test_escalate_phase.py's note)
# because it asserts a property of THIS module's ordering — escalate
# before write_table — not of escalate_selected alone.

def test_budget_failure_leaves_no_table_behind(tmp_path):
    """A breached ceiling must leave no partial table, because the router
    would trust one."""
    db = str(tmp_path / "t.db")
    scored = _seed_scored_items(n=25)
    scored_path = tmp_path / "scored.json"
    scored_path.write_text(json.dumps(scored))

    with Store(db) as store:
        # Seed references + tier0 hypotheses so escalate_selected actually
        # reaches the point of reserving spend, instead of skipping every
        # row for lack of ground truth.
        for item in scored:
            store.put_reference(item["segment_id"], "ref text", item["lang"])
            store.put_hypothesis(item["segment_id"], "tier0", "hyp text",
                                 {}, 0.0, TIER0_VARIANT)
        # Leave zero headroom: any paid call must breach the ceiling.
        store.reserve_spend("tier1", MAX_SPEND_USD)

    out = tmp_path / "d.json"
    with pytest.raises(BudgetExceeded):
        main(["escalate", "--db", db,
              "--scored-in", str(scored_path),
              "--out", str(out), "--confirm"])
    assert not out.exists()


# --- Addition 2: dropped (below-floor) buckets are marked in the histogram ---

def test_histogram_marks_buckets_below_the_floor(tmp_path, capsys):
    thin = _seed_scored_items(n=MIN_BUCKET_SAMPLES - 1, risk=0.35, prefix="thin")
    fat = _seed_scored_items(n=MIN_BUCKET_SAMPLES + 5, risk=0.65, prefix="fatx")
    scored = thin + fat
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(scored))

    rc = main(["escalate", "--db", str(tmp_path / "t.db"),
               "--scored-in", str(path), "--out", str(tmp_path / "d.json"),
               "--dry-run"])
    assert rc == 0

    err = capsys.readouterr().err
    lines = {ln.split()[0]: ln for ln in err.splitlines() if ln.strip().startswith("0.")}
    assert "below floor" in lines["0.3-0.4"], "thin bucket must be marked"
    assert "below floor" not in lines["0.6-0.7"], "fat bucket must not be marked"


# --- Addition 3: skipped rows (no reference / no Tier 0 hypothesis) are reported ---

def test_reports_skipped_segments_to_stderr(tmp_path, capsys):
    """An empty store means every selected segment lacks a reference and a
    Tier 0 hypothesis, so escalate_selected skips all of them. That must be
    visible, not silent — otherwise "3 legitimately lacked Tier 0" and "the
    variant key was wrong so everything was skipped" look identical."""
    db = str(tmp_path / "t.db")
    scored_path = _seed_scored(tmp_path, n=25)

    rc = main(["escalate", "--db", db, "--scored-in", scored_path,
               "--out", str(tmp_path / "d.json"), "--confirm"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "25" in err
    assert "skip" in err.lower()


# --- Addition 4: write_table must receive this run's marginal spend only ---

class _FakeTier1:
    """Stands in for dhvani.backends.tier1_chirp.Tier1Chirp so this test
    never reaches a live Chirp call. Monkeypatched onto the module the CLI
    imports it from; the CLI's deferred `from ... import Tier1Chirp` picks
    up the replacement at call time."""

    name = "tier1"
    variant_key = "fake"

    def cost_per_call(self, segment):
        return 0.01

    def transcribe(self, segment):
        return {"text": "fake tier1 output", "signals": {}}


def test_write_table_gets_marginal_spend_not_cumulative(tmp_path, monkeypatch):
    monkeypatch.setattr("dhvani.backends.tier1_chirp.Tier1Chirp", _FakeTier1)

    db = str(tmp_path / "t.db")
    scored = _seed_scored_items(n=25)
    scored_path = tmp_path / "scored.json"
    scored_path.write_text(json.dumps(scored))

    with Store(db) as store:
        # Unrelated pre-existing spend from some earlier run.
        store.reserve_spend("unrelated-tier", 5.0)
        for item in scored:
            store.put_reference(item["segment_id"], "ref text", item["lang"])
            store.put_hypothesis(item["segment_id"], "tier0", "hyp text",
                                 {}, 0.0, TIER0_VARIANT)

    out = tmp_path / "delta_table.json"
    rc = main(["escalate", "--db", db, "--scored-in", str(scored_path),
               "--out", str(out), "--confirm"])
    assert rc == 0

    with Store(db) as store:
        total_after = store.total_spend()

    payload = json.loads(out.read_text())
    spend = payload["meta"]["spend_usd"]

    assert spend == pytest.approx(total_after - 5.0), (
        "meta.spend_usd must be this run's marginal spend "
        "(store.total_spend() - before), not the cumulative total"
    )
    assert spend != pytest.approx(total_after), (
        "spend_usd must not equal the store's full cumulative history"
    )
    assert spend < 1.0, "sanity: this run's spend must be nowhere near the unrelated $5 baseline"


# --- C1: the CLI must look Tier 0 up under the key collect() actually wrote ---

def test_escalate_finds_tier0_under_the_real_variant_key(tmp_path, monkeypatch):
    """Regression for C1. The CLI called escalate_selected() without a
    tier0_variant, so it looked hypotheses up under "" while collect()
    stored them under Tier0Conformer.variant_key. Every segment therefore
    hit the "no Tier 0 output" skip, rows was empty, and the CLI wrote
    {"tier1": {}} and exited 0 — a silently empty calibration.

    Both pre-existing CLI tests seeded Tier 0 under "" and so agreed with
    the bug. This one seeds under the realistic key and demands rows.
    """
    monkeypatch.setattr("dhvani.backends.tier1_chirp.Tier1Chirp", _FakeTier1)

    db = str(tmp_path / "t.db")
    scored = _seed_scored_items(n=25)
    scored_path = tmp_path / "scored.json"
    scored_path.write_text(json.dumps(scored))

    with Store(db) as store:
        for item in scored:
            store.put_reference(item["segment_id"], "alpha beta gamma", item["lang"])
            store.put_hypothesis(item["segment_id"], "tier0", "alpha beta WRONG",
                                 {}, 0.0, TIER0_VARIANT)

    out = tmp_path / "delta_table.json"
    rc = main(["escalate", "--db", db, "--scored-in", str(scored_path),
               "--out", str(out), "--confirm"])
    assert rc == 0

    payload = json.loads(out.read_text())
    assert payload["tier1"], "no bucket measured: every segment was skipped"
    assert "0.6-0.7" in payload["tier1"]
    assert payload["meta"]["bucket_n"]["0.6-0.7"] == 25
