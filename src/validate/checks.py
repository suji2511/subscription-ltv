"""Validation rules run at every pipeline stage boundary.

Design intent: a check never fixes data. It observes, records, and (if severity
is FAIL) halts the pipeline. Repair belongs in the transform step, where it is
visible and testable.

Results append to data/validation_log.jsonl so the failure history is itself a
dataset — it is what lets the README report what the layer actually caught.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

import polars as pl

from src.config import (
    EPOCH_SENTINEL,
    OBSERVATION_CUTOFF,
    VALIDATION_LOG,
    WINDOW_OPEN,
)


class Severity(str, Enum):
    WARN = "warn"   # logged, pipeline continues
    FAIL = "fail"   # pipeline halts


@dataclass
class CheckResult:
    name: str
    stage: str
    passed: bool
    severity: Severity
    observed: dict
    checked_at: str


class ValidationError(RuntimeError):
    """Raised when a FAIL-severity check does not pass."""


def run_checks(stage: str, checks: list[Callable[[], CheckResult]]) -> list[CheckResult]:
    """Run every check, log all results, then raise if any FAIL check failed.

    Runs all checks before raising so a single pass surfaces every problem
    rather than one at a time.
    """
    results = [c() for c in checks]
    _log(results)
    failures = [r for r in results if not r.passed and r.severity is Severity.FAIL]
    if failures:
        names = ", ".join(f.name for f in failures)
        raise ValidationError(f"Stage '{stage}' failed validation: {names}")
    return results


def _log(results: list[CheckResult]) -> None:
    VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATION_LOG.open("a") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def make_result(name: str, stage: str, passed: bool, severity: Severity, **observed) -> CheckResult:
    return CheckResult(
        name=name,
        stage=stage,
        passed=passed,
        severity=severity,
        observed=observed,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


# --- Implemented rules ----------------------------------------------------


def row_count_reconciliation(
    stage: str,
    raw_rows: int,
    staged_rows: int,
    logged_drops: dict[str, int] | None = None,
) -> CheckResult:
    """Every row that left the previous stage must be accounted for.

    staged_rows + sum(logged_drops) must equal raw_rows exactly. Anything else
    is unexplained: rows vanished without a reason being recorded, or the join
    fanned out and invented rows. Both halt the pipeline.

    Assumption: `logged_drops` is a complete record of intentional removals,
    keyed by reason. What could violate it: a transform that filters rows
    without incrementing a counter — which is precisely the failure this rule
    exists to catch, so the counters must be incremented at the point of the
    filter, never reconstructed afterwards.
    """
    if raw_rows < 0 or staged_rows < 0:
        raise ValueError(
            f"Row counts cannot be negative (raw={raw_rows}, staged={staged_rows})"
        )

    drops = logged_drops or {}
    logged_drops_total = sum(drops.values())
    difference = raw_rows - staged_rows
    unexplained = difference - logged_drops_total

    return make_result(
        name="row_count_reconciliation",
        stage=stage,
        passed=unexplained == 0,
        severity=Severity.FAIL,
        raw_rows=raw_rows,
        staged_rows=staged_rows,
        difference=difference,
        logged_drops_total=logged_drops_total,
        logged_drops=drops,
        unexplained=unexplained,
    )


DATE_SANITY_COLUMNS = ["transaction_date", "membership_expire_date", "is_cancel"]


def date_sanity(df: pl.DataFrame, stage: str) -> CheckResult:
    """Dates must be internally consistent and inside the observation window.

    Three things fail the stage:
      * an expiry before its transaction date on a row that is NOT a
        cancellation
      * the 1970-01-01 epoch sentinel surviving into a staged frame
      * a transaction date outside [WINDOW_OPEN, OBSERVATION_CUTOFF]

    Assumption, and the reason this rule is conditional rather than blanket:
    a cancellation legitimately backdates the expiry, so expiry < transaction
    is correct for those rows. In the real data 147,200 of 153,660 such rows
    are cancellations; a blanket `expiry >= transaction` rule would halt the
    pipeline on 147k correct rows. What could violate the assumption: an
    is_cancel flag that means something other than "this transaction ended the
    membership early". Cancellation violations are counted and reported so the
    exemption stays visible rather than becoming a blind spot.

    Null expiries are expected — they are nulled epoch sentinels — and are
    counted but not failed.
    """
    missing = [c for c in DATE_SANITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"date_sanity requires columns, missing: {missing}")

    tx = pl.col("transaction_date")
    expiry = pl.col("membership_expire_date")

    is_sentinel = (expiry == EPOCH_SENTINEL).fill_null(False)
    # Sentinels are excluded here so the two failure modes do not double-count
    # the same row; a sentinel is a missing date, not a backwards one.
    expiry_before_tx = (expiry.is_not_null() & ~is_sentinel & (expiry < tx)).fill_null(False)
    out_of_window = ((tx < WINDOW_OPEN) | (tx > OBSERVATION_CUTOFF)).fill_null(False)
    # Cast so the rule works whether is_cancel arrives as 0/1 from the raw CSV
    # or as a BOOLEAN after staging.
    cancelled = pl.col("is_cancel").cast(pl.Int8) == 1

    observed = df.select(
        rows_checked=pl.len(),
        null_expiry=expiry.is_null().sum(),
        epoch_sentinel_expiry=is_sentinel.sum(),
        expiry_before_transaction_cancelled=(
            expiry_before_tx & cancelled
        ).sum(),
        expiry_before_transaction_not_cancelled=(
            expiry_before_tx & ~cancelled
        ).sum(),
        transaction_date_out_of_window=out_of_window.sum(),
    ).row(0, named=True)
    observed = {k: int(v) for k, v in observed.items()}

    passed = (
        observed["epoch_sentinel_expiry"] == 0
        and observed["expiry_before_transaction_not_cancelled"] == 0
        and observed["transaction_date_out_of_window"] == 0
    )

    return make_result(
        name="date_sanity",
        stage=stage,
        passed=passed,
        severity=Severity.FAIL,
        **observed,
    )


def censoring_rate(
    df: pl.DataFrame,
    stage: str,
    min_rate: float = 0.01,
    max_rate: float = 0.99,
) -> CheckResult:
    """Share of subjects censored, failed at either degenerate extreme.

    This is a plausibility check on the event labelling, not on retention. A
    censoring rate at either end of the range means the labelling logic did not
    fire rather than that the business is unusual:

      rate ~ 0.0  every subject churned; the censoring branch never ran
      rate ~ 1.0  almost nothing churned; there are no observed lifetimes

    The upper bound is the one that earns its place. transactions_v2 produced
    99.95% censoring with 630 events in 1.2M subjects, and looked like superb
    retention rather than an unusable file. This check turns that into a halt.

    Assumption: an `event` column where truthy means "churn observed". What
    could violate it: an inverted indicator, which would trip the opposite
    bound and still be caught — the check is symmetric for exactly that reason.
    """
    if "event" not in df.columns:
        raise ValueError("censoring_rate requires an 'event' column")
    if df.height == 0:
        raise ValueError("censoring_rate cannot run on an empty frame")

    n_subjects = df.height
    n_events = int(df.select(pl.col("event").cast(pl.Int8).sum()).item())
    n_censored = n_subjects - n_events
    rate = n_censored / n_subjects

    return make_result(
        name="censoring_rate",
        stage=stage,
        passed=min_rate <= rate <= max_rate,
        severity=Severity.FAIL,
        n_subjects=n_subjects,
        n_events=n_events,
        n_censored=n_censored,
        censoring_rate=rate,
        min_rate=min_rate,
        max_rate=max_rate,
    )


def duplicate_transactions(df: pl.DataFrame, stage: str) -> CheckResult:
    """Exact duplicate rows fail; same-day activity by one user does not.

    The distinction matters because the two look alike and mean opposite
    things. Two rows identical in every column are a double-read or a bad join,
    and staging them would double-count revenue. Two rows sharing only
    (msno, transaction_date) are ordinary business — a renewal and a
    cancellation land on the same date for 27,942 user-days in the real data,
    and both are real events.

    Counting is of surplus copies, not groups: three identical rows are two
    duplicates, because two rows have to go for the data to be right.

    Assumption: every column participates in identity. What could violate it:
    an ingestion-added column that varies per row (a load timestamp, a
    surrogate key) would make true duplicates look distinct. There is no such
    column in the staged schema, and adding one would need this revisited.
    """
    exact_duplicates = df.height - df.n_unique()

    same_day_user_rows = 0
    if {"msno", "transaction_date"} <= set(df.columns):
        per_user_day = df.group_by(["msno", "transaction_date"]).len()
        same_day_user_rows = int(
            per_user_day.filter(pl.col("len") > 1).select(pl.col("len").sum()).item() or 0
        )

    return make_result(
        name="duplicate_transactions",
        stage=stage,
        passed=exact_duplicates == 0,
        severity=Severity.FAIL,
        rows_checked=df.height,
        exact_duplicates=exact_duplicates,
        same_day_user_rows=same_day_user_rows,
    )


# --- Rules still to implement ---------------------------------------------
# Write the test in tests/test_checks.py BEFORE implementing each of these.
#
# TODO: price_reconciliation      -- plan_list_price vs actual_amount_paid vs plan_days
# TODO: schema_conformance        -- column names, dtypes, no unexpected nulls
