"""Tests for the run_all.py orchestrator.

Two layers, deliberately separate:

  wiring        the guard, fail-fast behaviour and stage ordering, tested with
                fakes and no data at all. Fast.
  end-to-end    the real pipeline run against a SYNTHETIC fixture of a few
                thousand users. This proves the stages actually compose --
                that each one produces what the next one reads -- without
                depending on the 21.5M-row dataset being present.

The fixture is sized so that cells clear MIN_EVENTS_PER_CELL; smaller would
silently skip every model fit and the test would pass while proving nothing.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import duckdb
import pytest

import run_all

TX_HEADER = (
    "msno,payment_method_id,payment_plan_days,plan_list_price,"
    "actual_amount_paid,is_auto_renew,transaction_date,"
    "membership_expire_date,is_cancel"
)
MEM_HEADER = "msno,city,bd,gender,registered_via,registration_init_time"

WINDOW_END = date(2017, 2, 28)


def make_fixture(raw_dir, n_users: int = 3000, seed: int = 0) -> None:
    """Write synthetic transactions/members CSVs with the real schema.

    Shaped to exercise every branch the pipeline cares about: left-truncated
    and incident subjects, churners and survivors, resubscribers with a gap,
    and users with no members row (the excluded population).
    """
    rng = random.Random(seed)
    tx = [TX_HEADER]
    mem = [MEM_HEADER]

    for i in range(n_users):
        msno = f"u{i:06d}"
        start = date(2015, rng.randint(1, 3), rng.randint(1, 28))

        # Half left-truncated (registered well before the window), half incident.
        reg = start - timedelta(days=rng.randint(200, 1200)) if i % 2 else start

        # Bimodal lifetimes: most churn during the window, a substantial
        # minority renew right through it and end up censored. Without
        # survivors the censoring rate is ~0 and `censoring_rate` rejects the
        # fixture -- as it should, since that is not a survival dataset.
        renewals = rng.randint(2, 15) if rng.random() < 0.6 else rng.randint(26, 32)
        cursor = start
        for _ in range(renewals):
            expiry = cursor + timedelta(days=30)
            if expiry > WINDOW_END:
                break
            tx.append(
                f"{msno},41,30,149,149,1,{cursor:%Y%m%d},{expiry:%Y%m%d},0"
            )
            cursor = expiry

        # One in five comes back after a gap wider than the grace window, so the
        # reactivation table is not empty.
        if i % 5 == 0:
            back = cursor + timedelta(days=rng.randint(45, 120))
            if back + timedelta(days=30) <= WINDOW_END:
                tx.append(
                    f"{msno},41,30,149,149,1,{back:%Y%m%d},"
                    f"{back + timedelta(days=30):%Y%m%d},0"
                )

        # One in ten has no members row -> the excluded population.
        if i % 10:
            mem.append(f"{msno},1,25,male,7,{reg:%Y%m%d}")

    (raw_dir / "transactions.csv").write_text("\n".join(tx) + "\n")
    (raw_dir / "members_v3.csv").write_text("\n".join(mem) + "\n")


@pytest.fixture
def synthetic_pipeline(tmp_path, monkeypatch):
    """Point every module at a temp fixture instead of the real data.

    Each module binds its paths from src.config at import time, so patching
    src.config alone would not reach them -- every bound name is patched
    individually. Verbose, but it fails loudly if a module grows a new path
    rather than silently writing into the real data directory.
    """
    from src.backtest import evaluate
    from src.cohorts import excluded_spells, revenue, spells
    from src.ingest import stage
    from src.models import km, ltv
    from src.validate import checks

    raw = tmp_path / "raw"
    raw.mkdir()
    make_fixture(raw)

    db = tmp_path / "staged.duckdb"
    marts = tmp_path / "marts.duckdb"

    monkeypatch.setattr(stage, "RAW_TRANSACTIONS", raw / "transactions.csv")
    monkeypatch.setattr(stage, "RAW_MEMBERS", raw / "members_v3.csv")
    monkeypatch.setattr(stage, "DB", db)
    monkeypatch.setattr(checks, "VALIDATION_LOG", tmp_path / "validation_log.jsonl")
    for mod in (spells, excluded_spells, revenue):
        monkeypatch.setattr(mod, "DB", db)
        monkeypatch.setattr(mod, "MARTS_DB", marts)
    for mod in (km, ltv, evaluate):
        monkeypatch.setattr(mod, "MARTS_DB", marts)

    monkeypatch.setattr(run_all, "RAW_TRANSACTIONS", raw / "transactions.csv")
    monkeypatch.setattr(run_all, "RAW_MEMBERS", raw / "members_v3.csv")
    monkeypatch.setattr(run_all, "MARTS_DB", marts)
    return {"raw": raw, "db": db, "marts": marts}


# --- wiring ---------------------------------------------------------------


def test_missing_transactions_exits_and_names_the_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all, "RAW_TRANSACTIONS", tmp_path / "transactions.csv")
    monkeypatch.setattr(run_all, "RAW_MEMBERS", tmp_path / "members_v3.csv")
    with pytest.raises(SystemExit) as e:
        run_all.check_raw_data()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "transactions.csv" in out
    assert "members_v3.csv" in out
    assert "kkbox" in out.lower()


def test_guard_names_only_the_file_that_is_missing(tmp_path, monkeypatch, capsys):
    present = tmp_path / "members_v3.csv"
    present.write_text(MEM_HEADER + "\n")
    monkeypatch.setattr(run_all, "RAW_TRANSACTIONS", tmp_path / "transactions.csv")
    monkeypatch.setattr(run_all, "RAW_MEMBERS", present)
    with pytest.raises(SystemExit):
        run_all.check_raw_data()
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert out.count("MISSING") == 1


def test_guard_passes_when_both_files_exist(tmp_path, monkeypatch):
    for name, header in [("transactions.csv", TX_HEADER), ("members_v3.csv", MEM_HEADER)]:
        (tmp_path / name).write_text(header + "\n")
    monkeypatch.setattr(run_all, "RAW_TRANSACTIONS", tmp_path / "transactions.csv")
    monkeypatch.setattr(run_all, "RAW_MEMBERS", tmp_path / "members_v3.csv")
    run_all.check_raw_data()  # must not raise


def test_run_stage_returns_the_stage_result():
    assert run_all.run_stage(1, "noop", lambda: "value") == "value"


def test_run_stage_aborts_the_run_when_a_stage_raises(capsys):
    def boom():
        raise RuntimeError("staging blew up")

    with pytest.raises(SystemExit) as e:
        run_all.run_stage(2, "ingest", boom)
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "STAGE 2 FAILED" in out
    assert "Stopping" in out


def test_a_failing_stage_prevents_later_stages_from_running(monkeypatch):
    """Fail-fast is the point: a broken mart must not be read downstream."""
    ran = []
    monkeypatch.setattr(run_all, "check_raw_data", lambda: ran.append("check"))

    def explode():
        ran.append("ingest")
        raise RuntimeError("no")

    import src.ingest.stage as stage_mod

    monkeypatch.setattr(stage_mod, "main", explode)
    monkeypatch.setattr(
        "src.cohorts.spells.main", lambda: ran.append("spells")
    )
    with pytest.raises(SystemExit):
        run_all.main()
    assert ran == ["check", "ingest"]
    assert "spells" not in ran


def test_spot_check_returns_false_when_figures_drift(synthetic_pipeline, capsys):
    """Synthetic numbers cannot match the README, so drift must be reported.

    The value of this check is that it fails loudly rather than quietly
    agreeing -- a spot-check that cannot fail proves nothing.
    """
    from src.ingest import stage

    stage.main()
    run_all.build_marts()

    class FakeKmf:
        median_survival_time_ = 999.0

    assert run_all.spot_check(FakeKmf(), {365: {"mae": 0.5}}) is False
    out = capsys.readouterr().out
    assert "DRIFT" in out


# --- end to end on synthetic data -----------------------------------------


def test_full_pipeline_composes_on_synthetic_data(synthetic_pipeline):
    """Every stage runs, and each produces what the next one reads."""
    from src.backtest import evaluate
    from src.ingest import stage
    from src.models import km, ltv

    stage.main()
    con = duckdb.connect(str(synthetic_pipeline["db"]), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM members").fetchone()[0] > 0
    con.close()

    run_all.build_marts()
    con = duckdb.connect(str(synthetic_pipeline["marts"]), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"spells", "spells_excluded", "spell_revenue", "reactivation"} <= tables
    subjects = con.execute("SELECT COUNT(*) FROM spells").fetchone()[0]
    events = con.execute("SELECT SUM(event::INT) FROM spells").fetchone()[0]
    con.close()
    assert subjects > 0
    assert events > 0, "fixture produced no events; model stages would be vacuous"

    pooled = km.main()
    assert pooled is not None, "pooled KM did not fit -- fixture too small"
    assert pooled.median_survival_time_ > 0

    ltv.main()

    results = evaluate.main()
    assert set(results) == {365, 180}


def test_reconciliation_holds_on_synthetic_data(synthetic_pipeline):
    """The staged->marts row accounting must balance on any input, not just ours."""
    from src.ingest import stage

    stage.main()
    run_all.build_marts()

    con = duckdb.connect(str(synthetic_pipeline["marts"]), read_only=True)
    con.execute(f"ATTACH '{synthetic_pipeline['db']}' AS staged (READ_ONLY)")
    users = con.execute("SELECT COUNT(DISTINCT msno) FROM staged.transactions").fetchone()[0]
    subjects = con.execute("SELECT COUNT(*) FROM spells").fetchone()[0]
    quarantined = con.execute("SELECT COUNT(*) FROM spells_quarantine").fetchone()[0]
    con.close()
    # Subjects + quarantined can never exceed the users they were built from.
    assert subjects + quarantined <= users
