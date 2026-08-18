"""Run the whole pipeline end to end, from two raw CSVs to backtested LTV.

    python run_all.py

A fresh clone needs only `transactions.csv` and `members_v3.csv` in `data/raw/`
(see the message printed when they are missing). Everything else -- staging,
marts, models, backtest -- is rebuilt from scratch by this one command.

Runtime on the full 21.5M-row dataset is about 1 minute 45 seconds on an
M-series MacBook. Measured breakdown: ingestion 25s, marts 50s, Kaplan-Meier
4s, LTV 14s, backtest 8s. The marts stage dominates because it segments 21.5M
transactions into spells three times over -- once each for the main table, the
excluded population and the revenue tables.

This module ORCHESTRATES ONLY. Every calculation lives in `src/`; nothing here
reimplements pipeline logic. It exists so the README's reproduction steps are a
single command that cannot drift from what the code actually does.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from src.config import MARTS_DB, RAW_MEMBERS, RAW_TRANSACTIONS

# Headline figures as published in README.md. The final stage re-derives each
# one from what the pipeline just produced and compares. This turns the README
# into something the pipeline can falsify: if a number here stops matching, one
# of the two is wrong and the run says so rather than quietly disagreeing.
README_SUBJECTS = 1_873_529
README_POOLED_MEDIAN_DAYS = 295
README_BACKTEST_MAE_365 = 0.0781

KAGGLE_SOURCE = (
    "KKBox's Churn Prediction Challenge on Kaggle:\n"
    "      https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data\n"
    "    Download transactions.csv.7z and members_v3.csv.7z from the\n"
    "    ORIGINAL competition files -- NOT churn_comp_refresh.7z, whose\n"
    "    transactions_v2.csv is a March-2017 snapshot with no usable\n"
    "    survival signal (see docs/step1_audit.md)."
)


def check_raw_data() -> None:
    """Stop before doing any work if either raw input is missing.

    Deliberately not a warning: a partial pipeline would produce marts that
    look complete and are not, which is exactly the class of silent failure
    the rest of this project is built to prevent.
    """
    missing = [p for p in (RAW_TRANSACTIONS, RAW_MEMBERS) if not p.exists()]
    if not missing:
        return
    print("Cannot start: required raw data is missing.\n")
    for path in missing:
        print(f"  MISSING  {path}")
    print(f"\n  Get these from the {KAGGLE_SOURCE}")
    print(f"\n  Extract both into {RAW_TRANSACTIONS.parent}/ and re-run.")
    sys.exit(1)


def run_stage(number: int, name: str, fn):
    """Run one stage, timed, and abort the whole run if it raises.

    Fail-fast is the point: continuing past a broken stage would leave a
    half-built mart that later stages would happily read.
    """
    banner = f" STAGE {number}: {name} "
    print(f"\n{'=' * 78}\n{banner:=^78}\n{'=' * 78}", flush=True)
    started = time.time()
    try:
        result = fn()
    except Exception:  # noqa: BLE001 - any failure must abort the run
        print(f"\n{'!' * 78}")
        print(f"STAGE {number} FAILED: {name}")
        print(f"{'!' * 78}\n")
        traceback.print_exc()
        print(
            f"\nStopping. Nothing after stage {number} has run, so no downstream "
            "output\nis based on this failure."
        )
        sys.exit(1)
    print(f"\n  [stage {number} complete in {time.time() - started:.1f}s]", flush=True)
    return result


def build_marts() -> None:
    """staged -> marts. Three tables, built in dependency order."""
    from src.cohorts import excluded_spells, revenue, spells

    spells.main()
    excluded_spells.main()
    revenue.main()


def spot_check(pooled_kmf, backtest_results) -> bool:
    """Re-derive three headline README figures from this run and compare.

    Returns True if every check matches. A mismatch is not necessarily a bug --
    it means the code and the README have diverged, and one of them needs
    updating before either is trusted.
    """
    import duckdb

    con = duckdb.connect(str(MARTS_DB), read_only=True)
    subjects = con.execute("SELECT COUNT(*) FROM spells").fetchone()[0]
    con.close()

    median = float(pooled_kmf.median_survival_time_)
    mae = backtest_results[365]["mae"]

    checks = [
        ("total subjects", subjects, README_SUBJECTS, 0),
        ("pooled median survival (days)", median, README_POOLED_MEDIAN_DAYS, 0),
        ("backtest MAE, 365d", mae, README_BACKTEST_MAE_365, 5e-5),
    ]

    print(f"\n  {'figure':<32} {'this run':>12} {'README':>12}   status")
    ok = True
    for label, got, expected, tol in checks:
        matches = abs(got - expected) <= tol
        ok &= matches
        got_s = f"{got:,.4f}" if isinstance(got, float) and tol else f"{got:,.0f}"
        exp_s = f"{expected:,.4f}" if isinstance(got, float) and tol else f"{expected:,.0f}"
        print(f"  {label:<32} {got_s:>12} {exp_s:>12}   {'MATCH' if matches else 'DRIFT'}")
    return ok


def summarise(pooled_kmf, backtest_results) -> None:
    root = Path(__file__).resolve().parent
    print("\nOutputs on disk:")
    for label, path in [
        ("staged transactions + members", root / "data/staged/subscriptions.duckdb"),
        ("spell table, revenue, reactivation", MARTS_DB),
        ("validation history", root / "data/validation_log.jsonl"),
        ("step-1 data audit", root / "docs/step1_audit.md"),
        ("decisions and their reasoning", root / "docs/decisions.md"),
    ]:
        mark = "  " if path.exists() else "  (missing) "
        print(f"{mark}{label:<36} {path.relative_to(root)}")

    print(
        "\n  Survival curves, LTV figures and backtest tables are printed to stdout\n"
        "  by stages 4-6 above; they are derived from marts.duckdb and are not\n"
        "  separately persisted."
    )

    print("\nSpot-check against README.md:")
    if spot_check(pooled_kmf, backtest_results):
        print("\n  All three headline figures match README.md.")
    else:
        print(
            "\n  MISMATCH: the pipeline and README.md disagree. One of them is\n"
            "  out of date -- do not quote either until that is resolved."
        )
        sys.exit(1)


def main() -> None:
    overall_started = time.time()
    print("Subscription retention & LTV -- full pipeline")
    print(f"  transactions : {RAW_TRANSACTIONS}")
    print(f"  members      : {RAW_MEMBERS}")

    run_stage(1, "check raw data", check_raw_data)

    from src.backtest import evaluate
    from src.ingest import stage
    from src.models import km, ltv

    run_stage(2, "ingest raw -> DuckDB staging", stage.main)
    run_stage(3, "build marts (spells, excluded, revenue)", build_marts)
    pooled_kmf = run_stage(4, "Kaplan-Meier survival", km.main)
    run_stage(5, "lifetime value", ltv.main)
    backtest_results = run_stage(6, "backtest", evaluate.main)

    print(f"\n{'=' * 78}\n{' SUMMARY ':=^78}\n{'=' * 78}")
    summarise(pooled_kmf, backtest_results)
    print(f"\nFull pipeline completed in {time.time() - overall_started:.1f}s.")


if __name__ == "__main__":
    main()
