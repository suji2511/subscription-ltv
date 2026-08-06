"""Tests for the validation rules.

Written before the implementation, per CLAUDE.md. Every fixture here is a small
synthetic frame with known-good and known-bad rows, never the real data — a
test that reads 21.5M rows tells you the data changed, not that the code broke.

The rules under test are deliberately picky about two things the step-1 audit
showed are real in this dataset:
  * rows can legitimately have expiry before transaction date IF is_cancel=1
  * 1970-01-01 is a missing-date sentinel, not a real expiry
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.validate.checks import (
    Severity,
    ValidationError,
    censoring_rate,
    date_sanity,
    duplicate_transactions,
    row_count_reconciliation,
    run_checks,
)


@pytest.fixture(autouse=True)
def isolated_validation_log(tmp_path, monkeypatch):
    """Send run_checks' log to a temp file, not data/validation_log.jsonl.

    Without this, running the suite would append synthetic results to the real
    validation history — which CLAUDE.md treats as a dataset in its own right.
    """
    monkeypatch.setattr(
        "src.validate.checks.VALIDATION_LOG", tmp_path / "validation_log.jsonl"
    )


def make_frame(rows: list[dict]) -> pl.DataFrame:
    """Build a transactions-shaped frame from a list of row dicts.

    Defaults give a clean, valid row so each test only states the fields it is
    actually exercising.
    """
    default = {
        "msno": "u1",
        "transaction_date": date(2015, 6, 1),
        "membership_expire_date": date(2015, 7, 1),
        "is_cancel": 0,
    }
    return pl.DataFrame([{**default, **r} for r in rows])


# --- row_count_reconciliation ---------------------------------------------


def test_reconciliation_passes_when_counts_match_exactly():
    result = row_count_reconciliation(stage="raw->staged", raw_rows=100, staged_rows=100)
    assert result.passed
    assert result.observed["difference"] == 0


def test_reconciliation_passes_when_drops_account_for_the_difference():
    """Rows may disappear, but only if something logged why."""
    result = row_count_reconciliation(
        stage="raw->staged",
        raw_rows=100,
        staged_rows=90,
        logged_drops={"duplicate_transactions": 7, "unparseable_date": 3},
    )
    assert result.passed
    assert result.observed["logged_drops_total"] == 10


def test_reconciliation_fails_on_unexplained_loss():
    """The whole point of the rule: silent row loss must halt the pipeline."""
    result = row_count_reconciliation(
        stage="raw->staged", raw_rows=100, staged_rows=90, logged_drops={"dupes": 4}
    )
    assert not result.passed
    assert result.severity is Severity.FAIL
    assert result.observed["unexplained"] == 6


def test_reconciliation_fails_when_rows_appear_from_nowhere():
    """More staged than raw means a join fanned out. Also a failure."""
    result = row_count_reconciliation(stage="raw->staged", raw_rows=100, staged_rows=105)
    assert not result.passed
    assert result.observed["unexplained"] == -5


def test_reconciliation_fails_when_drops_exceed_the_gap():
    """Claiming more drops than rows actually lost means the accounting is wrong."""
    result = row_count_reconciliation(
        stage="raw->staged", raw_rows=100, staged_rows=98, logged_drops={"dupes": 10}
    )
    assert not result.passed


def test_reconciliation_rejects_negative_counts():
    with pytest.raises(ValueError):
        row_count_reconciliation(stage="raw->staged", raw_rows=-1, staged_rows=0)


# --- date_sanity ----------------------------------------------------------


def test_date_sanity_passes_on_clean_rows():
    df = make_frame([{}, {"membership_expire_date": date(2015, 12, 31)}])
    result = date_sanity(df, stage="raw->staged")
    assert result.passed
    assert result.observed["expiry_before_transaction_not_cancelled"] == 0


def test_date_sanity_allows_expiry_before_transaction_when_cancelled():
    """147,200 of 153,660 such rows in the real data are legitimate cancellations.

    A blanket expiry >= transaction rule would fail all of them and halt the
    pipeline on correct data. See docs/decisions.md.
    """
    df = make_frame(
        [
            {
                "transaction_date": date(2015, 6, 1),
                "membership_expire_date": date(2015, 5, 1),
                "is_cancel": 1,
            }
        ]
    )
    result = date_sanity(df, stage="raw->staged")
    assert result.passed
    assert result.observed["expiry_before_transaction_cancelled"] == 1
    assert result.observed["expiry_before_transaction_not_cancelled"] == 0


def test_date_sanity_fails_on_expiry_before_transaction_without_cancel():
    df = make_frame(
        [
            {
                "transaction_date": date(2015, 6, 1),
                "membership_expire_date": date(2015, 5, 1),
                "is_cancel": 0,
            }
        ]
    )
    result = date_sanity(df, stage="raw->staged")
    assert not result.passed
    assert result.severity is Severity.FAIL
    assert result.observed["expiry_before_transaction_not_cancelled"] == 1


def test_date_sanity_accepts_boolean_is_cancel_after_staging():
    """is_cancel is 0/1 in the raw CSV but BOOLEAN once staged."""
    df = pl.DataFrame(
        {
            "msno": ["u1"],
            "transaction_date": [date(2015, 6, 1)],
            "membership_expire_date": [date(2015, 5, 1)],
            "is_cancel": [True],
        }
    )
    result = date_sanity(df, stage="staged")
    assert result.passed
    assert result.observed["expiry_before_transaction_cancelled"] == 1


def test_date_sanity_fails_on_epoch_sentinel_reaching_the_check():
    """1970-01-01 must be nulled at ingestion; seeing one here means it leaked."""
    df = make_frame([{"membership_expire_date": date(1970, 1, 1)}])
    result = date_sanity(df, stage="raw->staged")
    assert not result.passed
    assert result.observed["epoch_sentinel_expiry"] == 1


def test_date_sanity_fails_on_dates_after_the_observation_window():
    df = make_frame([{"transaction_date": date(2019, 1, 1)}])
    result = date_sanity(df, stage="raw->staged")
    assert not result.passed
    assert result.observed["transaction_date_out_of_window"] == 1


def test_date_sanity_fails_on_transaction_before_window_open():
    df = make_frame([{"transaction_date": date(2014, 12, 31)}])
    result = date_sanity(df, stage="raw->staged")
    assert not result.passed
    assert result.observed["transaction_date_out_of_window"] == 1


def test_date_sanity_tolerates_null_expiry():
    """Nulled sentinels are expected and must not be counted as violations."""
    df = pl.DataFrame(
        {
            "msno": ["u1"],
            "transaction_date": [date(2015, 6, 1)],
            "membership_expire_date": [None],
            "is_cancel": [0],
        },
        schema_overrides={"membership_expire_date": pl.Date},
    )
    result = date_sanity(df, stage="raw->staged")
    assert result.passed
    assert result.observed["null_expiry"] == 1


def test_date_sanity_counts_every_violation_not_just_the_first():
    df = make_frame(
        [
            {},  # clean
            {"membership_expire_date": date(2015, 5, 1)},  # expiry < tx, not cancelled
            {"membership_expire_date": date(2015, 5, 2)},  # ditto
            # Out of window only. Its expiry is moved forward too, so this row
            # trips exactly one rule and the counts stay independent.
            {
                "transaction_date": date(2020, 1, 1),
                "membership_expire_date": date(2020, 2, 1),
            },
        ]
    )
    result = date_sanity(df, stage="raw->staged")
    assert not result.passed
    assert result.observed["expiry_before_transaction_not_cancelled"] == 2
    assert result.observed["transaction_date_out_of_window"] == 1
    assert result.observed["rows_checked"] == 4


def test_date_sanity_rejects_a_frame_missing_a_required_column():
    df = pl.DataFrame({"msno": ["u1"], "transaction_date": [date(2015, 6, 1)]})
    with pytest.raises(ValueError, match="membership_expire_date"):
        date_sanity(df, stage="raw->staged")


# --- run_checks integration -----------------------------------------------


def test_run_checks_raises_when_a_fail_check_fails():
    df = make_frame([{"membership_expire_date": date(2015, 5, 1)}])
    with pytest.raises(ValidationError):
        run_checks(
            "raw->staged",
            [
                lambda: row_count_reconciliation("raw->staged", 10, 10),
                lambda: date_sanity(df, stage="raw->staged"),
            ],
        )


def test_run_checks_runs_every_check_before_raising():
    """A single pass should surface every problem, not stop at the first."""
    df = make_frame([{"membership_expire_date": date(2015, 5, 1)}])
    called = []

    def tracked(name, fn):
        def inner():
            called.append(name)
            return fn()

        return inner

    with pytest.raises(ValidationError):
        run_checks(
            "raw->staged",
            [
                tracked("recon", lambda: row_count_reconciliation("raw->staged", 10, 5)),
                tracked("dates", lambda: date_sanity(df, stage="raw->staged")),
            ],
        )
    assert called == ["recon", "dates"]


def test_run_checks_returns_results_when_all_pass():
    df = make_frame([{}])
    results = run_checks(
        "raw->staged",
        [
            lambda: row_count_reconciliation("raw->staged", 1, 1),
            lambda: date_sanity(df, stage="raw->staged"),
        ],
    )
    assert len(results) == 2
    assert all(r.passed for r in results)


# --- censoring_rate -------------------------------------------------------


def make_spells(events: list[int]) -> pl.DataFrame:
    """Spell-table-shaped frame carrying only the event indicator."""
    return pl.DataFrame({"msno": [f"u{i}" for i in range(len(events))], "event": events})


def test_censoring_rate_passes_on_a_plausible_mix():
    result = censoring_rate(make_spells([1, 0, 1, 0]), stage="staged->marts")
    assert result.passed
    assert result.observed["censoring_rate"] == 0.5
    assert result.observed["n_events"] == 2


def test_censoring_rate_fails_when_nothing_is_censored():
    """Everyone churning means the censoring logic never fired."""
    result = censoring_rate(make_spells([1, 1, 1, 1]), stage="staged->marts")
    assert not result.passed
    assert result.observed["censoring_rate"] == 0.0


def test_censoring_rate_fails_when_almost_nothing_churns():
    """The transactions_v2 failure mode: 99.95% censored, 630 events in 1.2M.

    A rate this high means the file has no observed lifetimes, not that
    retention is excellent. This check is what would have caught it.
    """
    events = [0] * 999 + [1]
    result = censoring_rate(make_spells(events), stage="staged->marts")
    assert not result.passed
    assert result.observed["censoring_rate"] == 0.999


def test_censoring_rate_accepts_boolean_events():
    result = censoring_rate(
        pl.DataFrame({"msno": ["a", "b"], "event": [True, False]}), stage="staged->marts"
    )
    assert result.passed
    assert result.observed["n_events"] == 1


def test_censoring_rate_bounds_are_configurable():
    events = [0] * 90 + [1] * 10  # 90% censored
    assert censoring_rate(make_spells(events), stage="s").passed
    assert not censoring_rate(make_spells(events), stage="s", max_rate=0.8).passed


def test_censoring_rate_rejects_an_empty_frame():
    with pytest.raises(ValueError):
        censoring_rate(make_spells([]), stage="staged->marts")


def test_censoring_rate_rejects_a_frame_without_an_event_column():
    with pytest.raises(ValueError, match="event"):
        censoring_rate(pl.DataFrame({"msno": ["a"]}), stage="staged->marts")


# --- duplicate_transactions -----------------------------------------------


def test_duplicates_passes_on_distinct_rows():
    df = make_frame([{"msno": "u1"}, {"msno": "u2"}])
    result = duplicate_transactions(df, stage="raw->staged")
    assert result.passed
    assert result.observed["exact_duplicates"] == 0


def test_duplicates_fails_on_fully_identical_rows():
    df = make_frame([{"msno": "u1"}, {"msno": "u1"}])
    result = duplicate_transactions(df, stage="raw->staged")
    assert not result.passed
    assert result.severity is Severity.FAIL
    assert result.observed["exact_duplicates"] == 1


def test_duplicates_allows_same_day_transactions_that_differ():
    """A renewal and a cancellation on one date are two real events, not a dupe.

    27,942 user-days in the real data carry more than one transaction.
    """
    df = make_frame(
        [
            {"msno": "u1", "transaction_date": date(2015, 6, 1), "is_cancel": 0},
            {"msno": "u1", "transaction_date": date(2015, 6, 1), "is_cancel": 1},
        ]
    )
    result = duplicate_transactions(df, stage="raw->staged")
    assert result.passed
    assert result.observed["exact_duplicates"] == 0
    assert result.observed["same_day_user_rows"] == 2


def test_duplicates_counts_repeats_not_groups():
    """Three identical rows are two surplus copies, not one."""
    df = make_frame([{"msno": "u1"}] * 3)
    result = duplicate_transactions(df, stage="raw->staged")
    assert result.observed["exact_duplicates"] == 2
