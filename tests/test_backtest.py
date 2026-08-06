"""Tests for the backtest module.

Written before the implementation, per CLAUDE.md. Synthetic frames with known
answers throughout -- the point is to prove the arithmetic and the censoring
logic, not to re-measure the dataset.

The sign convention is the thing most worth pinning down: we already expect
these forecasts to be optimistic, so a bias metric whose sign is ambiguous
would be worse than useless.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.evaluate import (
    censor_at_fit_window,
    eligible_cohorts,
    error_metrics,
    truncation_error_relationship,
)


def make_spells(rows: list[dict]) -> pd.DataFrame:
    default = {
        "msno": "u1",
        "cohort_month": "2015-01",
        "plan_type": "02_8-31d",
        "registration_date": date(2015, 1, 1),
        "spell_start_date": date(2015, 1, 1),
        "end_date": date(2016, 1, 1),
        "entry_days": 0,
        "tenure_days": 365,
        "event": True,
    }
    return pd.DataFrame([{**default, **r} for r in rows])


# --- censor_at_fit_window -------------------------------------------------


def test_censoring_leaves_subjects_who_ended_before_the_fit_window():
    df = make_spells([{"end_date": date(2016, 6, 1), "tenure_days": 517}])
    out = censor_at_fit_window(df, date(2016, 12, 31))
    assert bool(out.loc[0, "event"]) is True
    assert out.loc[0, "tenure_days"] == 517


def test_censoring_truncates_subjects_who_ended_after_the_fit_window():
    """A churn that happens after the fit window has NOT been observed yet."""
    df = make_spells([{"end_date": date(2017, 1, 15), "tenure_days": 745}])
    out = censor_at_fit_window(df, date(2016, 12, 31))
    assert bool(out.loc[0, "event"]) is False
    # registration 2015-01-01 -> fit window end 2016-12-31 is 730 days
    assert out.loc[0, "tenure_days"] == 730


def test_censoring_drops_subjects_who_start_after_the_fit_window():
    df = make_spells(
        [
            {"msno": "early", "spell_start_date": date(2016, 1, 1)},
            {"msno": "late", "spell_start_date": date(2017, 2, 1)},
        ]
    )
    out = censor_at_fit_window(df, date(2016, 12, 31))
    assert list(out["msno"]) == ["early"]


def test_censoring_drops_subjects_whose_tenure_goes_non_positive():
    """Registered after the fit window: nothing about them is observable."""
    df = make_spells(
        [{"registration_date": date(2016, 12, 30), "spell_start_date": date(2016, 12, 30)}]
    )
    out = censor_at_fit_window(df, date(2016, 12, 31))
    assert len(out) == 1
    df2 = make_spells(
        [{"registration_date": date(2017, 6, 1), "spell_start_date": date(2016, 1, 1)}]
    )
    assert len(censor_at_fit_window(df2, date(2016, 12, 31))) == 0


def test_censoring_does_not_mutate_the_input():
    df = make_spells([{"end_date": date(2017, 1, 15), "tenure_days": 745}])
    censor_at_fit_window(df, date(2016, 12, 31))
    assert bool(df.loc[0, "event"]) is True
    assert df.loc[0, "tenure_days"] == 745


# --- error_metrics --------------------------------------------------------


def test_error_metrics_on_a_perfect_forecast():
    m = error_metrics([0.5, 0.6], [0.5, 0.6])
    assert m["mae"] == 0.0
    assert m["bias"] == 0.0
    assert m["n"] == 2


def test_bias_is_positive_when_the_forecast_is_optimistic():
    """Sign convention: bias = mean(forecast - realised).

    Positive bias means predicted survival exceeded realised survival, i.e.
    the model was OPTIMISTIC. Both known biases in this project predict a
    positive number here, so the sign has to be unambiguous.
    """
    m = error_metrics([0.7, 0.8], [0.5, 0.6])
    assert m["bias"] == pytest.approx(0.2)
    assert m["mae"] == pytest.approx(0.2)


def test_bias_is_negative_when_the_forecast_is_pessimistic():
    m = error_metrics([0.4, 0.5], [0.5, 0.6])
    assert m["bias"] == pytest.approx(-0.1)


def test_mae_and_bias_differ_when_errors_cancel():
    """MAE alone would hide a model that is wrong in both directions."""
    m = error_metrics([0.6, 0.4], [0.5, 0.5])
    assert m["bias"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.1)


def test_error_metrics_ignores_missing_pairs():
    m = error_metrics([0.5, None, 0.7], [0.4, 0.5, None])
    assert m["n"] == 1
    assert m["bias"] == pytest.approx(0.1)


def test_error_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        error_metrics([0.1, 0.2], [0.1])


def test_error_metrics_on_no_usable_pairs():
    m = error_metrics([None], [None])
    assert m["n"] == 0
    assert m["mae"] is None


# --- eligible_cohorts -----------------------------------------------------


def test_eligible_cohorts_requires_full_follow_up():
    starts = {"2015-01": date(2015, 1, 1), "2016-06": date(2016, 6, 1)}
    got = eligible_cohorts(starts, horizon=365, as_of=date(2017, 1, 29))
    assert got == ["2015-01"]


def test_eligible_cohorts_boundary_is_inclusive():
    starts = {"exact": date(2016, 1, 30)}
    assert eligible_cohorts(starts, horizon=365, as_of=date(2017, 1, 29)) == ["exact"]


def test_eligible_cohorts_shorter_horizon_admits_more():
    starts = {"a": date(2015, 1, 1), "b": date(2016, 6, 1)}
    assert len(eligible_cohorts(starts, 365, date(2017, 1, 29))) == 1
    assert len(eligible_cohorts(starts, 180, date(2017, 1, 29))) == 2


# --- truncation_error_relationship ---------------------------------------


def test_truncation_error_relationship_detects_a_positive_association():
    """More truncation -> larger error is the failure mode we care about."""
    cells = pd.DataFrame(
        {
            "truncated_share": [0.1, 0.3, 0.6, 0.9],
            "abs_error": [0.01, 0.03, 0.06, 0.09],
        }
    )
    r = truncation_error_relationship(cells)
    assert r["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert r["n_cells"] == 4


def test_truncation_error_relationship_returns_none_when_too_few_cells():
    cells = pd.DataFrame({"truncated_share": [0.5], "abs_error": [0.02]})
    assert truncation_error_relationship(cells)["correlation"] is None


def test_truncation_error_relationship_handles_zero_variance():
    cells = pd.DataFrame(
        {"truncated_share": [0.5, 0.5, 0.5], "abs_error": [0.01, 0.02, 0.03]}
    )
    assert truncation_error_relationship(cells)["correlation"] is None
