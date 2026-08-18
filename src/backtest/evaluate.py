"""Backtest: forecast from a fit window, compare against what actually happened.

The design question this answers is not "is the model accurate" but "is it
accurate in the direction we feared". Two documented biases -- delayed entry
with overstated entry times, and the exclusion of 432,592 fast-churning users --
both predict OPTIMISTIC forecasts. A signed bias metric tests that prediction
directly; MAE alone would hide it.

Method:
  1. Censor every subject at FIT_WINDOW_END, discarding knowledge the model
     would not have had. Churn after that date becomes censored, not observed.
  2. Fit one delayed-entry KM curve per stratum on that censored view.
  3. For every (cohort, stratum) cell, forecast S(H) from the stratum curve.
  4. Compare against the cell's realised S(H), fitted on the full data.
  5. Report MAE and signed bias, and test whether error tracks truncation.

FIT_WINDOW_END is local to this module, deliberately. It is a property of the
backtest, not of the data: as a global cutoff it would silently strip follow-up
from every downstream analysis.

Run:  python -m src.backtest.evaluate
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
from lifelines.utils import restricted_mean_survival_time

from src.config import (
    LABEL_CUTOFF,
    MARTS_DB,
    MIN_FOLLOWUP_DAYS,
    SECONDARY_FOLLOWUP_DAYS,
)
from src.models.km import fit

# The fit window. Local by design -- see module docstring.
FIT_WINDOW_END = date(2016, 12, 31)

HORIZONS = [MIN_FOLLOWUP_DAYS, SECONDARY_FOLLOWUP_DAYS]  # 365 primary, 180 secondary


def censor_at_fit_window(df: pd.DataFrame, fit_end: date) -> pd.DataFrame:
    """Restrict a spell table to what was knowable at `fit_end`.

    Three things change. A churn recorded after `fit_end` had not happened yet
    as far as the model is concerned, so the subject becomes censored and their
    tenure is cut back to `fit_end`. Subjects whose spell had not started are
    dropped. Subjects whose tenure would be non-positive are dropped, since a
    subject cannot enter and leave the risk set at the same instant.

    Assumption: administrative censoring at a calendar date is independent of
    the churn process. What could violate it: a promotion or price change timed
    near `fit_end` would make the censoring informative. 2016-12-31 was chosen
    partly because nothing in the transaction volume series marks it out.
    """
    out = df.copy()
    fit_ts = pd.Timestamp(fit_end)

    out = out[pd.to_datetime(out["spell_start_date"]) <= fit_ts]

    # Anyone whose clock had not started by fit_end is unobservable, regardless
    # of what their end_date says. Filtering on end_date alone would leave such
    # a row untouched and carry a stale tenure through into the fit.
    days_to_fit_end = (fit_ts - pd.to_datetime(out["registration_date"])).dt.days
    out = out[days_to_fit_end > 0]
    days_to_fit_end = days_to_fit_end[out.index]

    # Widen before assigning: tenure_days arrives as int32 from DuckDB and the
    # recomputed values are int64, which pandas will refuse to coerce silently
    # in a future version.
    out["tenure_days"] = out["tenure_days"].astype("int64")
    out["event"] = out["event"].astype(bool)

    ended_after = pd.to_datetime(out["end_date"]) > fit_ts
    out.loc[ended_after, "tenure_days"] = days_to_fit_end[ended_after]
    out.loc[ended_after, "event"] = False

    out = out[out["tenure_days"] > out["entry_days"]]
    return out.reset_index(drop=True)


def error_metrics(forecast: list, realised: list) -> dict:
    """MAE and signed bias over paired forecasts.

    bias = mean(forecast - realised). POSITIVE means the model predicted more
    survival than actually occurred, i.e. it was optimistic. Both known biases
    in this project predict a positive value, so this sign is the headline.

    MAE and bias are both reported because they answer different questions: a
    model wrong by +0.1 and -0.1 has zero bias and non-zero MAE, and calling
    that model unbiased would be a mistake worth avoiding.
    """
    if len(forecast) != len(realised):
        raise ValueError(
            f"forecast and realised must be the same length "
            f"({len(forecast)} vs {len(realised)})"
        )
    pairs = [
        (f, r)
        for f, r in zip(forecast, realised)
        if f is not None and r is not None and not pd.isna(f) and not pd.isna(r)
    ]
    if not pairs:
        return {"n": 0, "mae": None, "bias": None}
    errors = [f - r for f, r in pairs]
    return {
        "n": len(pairs),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
    }


def eligible_cohorts(starts: dict, horizon: int, as_of: date) -> list[str]:
    """Cohorts with a full `horizon` of follow-up by `as_of`.

    A cohort is a fair forecast target only if its realised S(H) is actually
    observable; otherwise the forecast is scored against a truncated actual and
    the error metric flatters the model.
    """
    return sorted(
        name
        for name, start in starts.items()
        if (as_of - start).days >= horizon
    )


def truncation_error_relationship(cells: pd.DataFrame) -> dict:
    """Does forecast error grow with left truncation?

    This is the empirical test of the delayed-entry assumption. If entry times
    are trustworthy, error should be unrelated to how truncated a cell is. A
    positive correlation is evidence that delayed entry is doing damage where
    it is applied most heavily.
    """
    usable = cells.dropna(subset=["truncated_share", "abs_error"])
    if len(usable) < 2:
        return {"n_cells": len(usable), "correlation": None}
    if usable["truncated_share"].nunique() < 2 or usable["abs_error"].nunique() < 2:
        return {"n_cells": len(usable), "correlation": None}
    return {
        "n_cells": len(usable),
        "correlation": float(usable["truncated_share"].corr(usable["abs_error"])),
    }


# --- the backtest itself --------------------------------------------------


def load(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT msno, cohort_month, plan_type, registration_date, spell_start_date,
               end_date, entry_days, tenure_days, event
        FROM spells
        """
    ).df()


def cohort_starts(df: pd.DataFrame) -> dict:
    g = df.groupby("cohort_month")["spell_start_date"].min()
    return {k: pd.Timestamp(v).date() for k, v in g.items()}


def run_horizon(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict]:
    """Forecast S(H) and RMST(H) per (cohort, stratum); compare to realised."""
    fit_view = censor_at_fit_window(df, FIT_WINDOW_END)
    cohorts = eligible_cohorts(cohort_starts(df), horizon, LABEL_CUTOFF)

    # Forecast: one stratum-level curve per plan_type, fitted on the fit window.
    models = {}
    for st, part in fit_view.groupby("plan_type", observed=True):
        kmf = fit(part, st)
        if kmf is not None:
            models[st] = kmf

    rows = []
    for (cohort, st), cell in df.groupby(["cohort_month", "plan_type"], observed=True):
        if cohort not in cohorts or st not in models:
            continue
        realised_kmf = fit(cell, f"{cohort}/{st}")
        if realised_kmf is None:
            continue

        f_s = float(models[st].survival_function_at_times([horizon]).iloc[0])
        r_s = float(realised_kmf.survival_function_at_times([horizon]).iloc[0])
        f_rmst = float(restricted_mean_survival_time(models[st], t=horizon))
        r_rmst = float(restricted_mean_survival_time(realised_kmf, t=horizon))

        rows.append(
            {
                "cohort": cohort,
                "plan_type": st,
                "n": len(cell),
                "truncated_share": float((cell["entry_days"] > 0).mean()),
                "forecast_S": f_s,
                "realised_S": r_s,
                "error_S": f_s - r_s,
                "abs_error": abs(f_s - r_s),
                "forecast_RMST": f_rmst,
                "realised_RMST": r_rmst,
                "error_RMST": f_rmst - r_rmst,
            }
        )

    cells = pd.DataFrame(rows)
    overall = error_metrics(
        cells["forecast_S"].tolist(), cells["realised_S"].tolist()
    )
    return cells, overall


def report(cells: pd.DataFrame, overall: dict, horizon: int, tag: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"{tag}: {horizon}-DAY HORIZON  (fit window ends {FIT_WINDOW_END})")
    print("=" * 78)
    print(
        f"  cells {overall['n']}   MAE {overall['mae']:.4f}   "
        f"signed bias {overall['bias']:+.4f}"
    )
    print(
        "  => forecasts are OPTIMISTIC as predicted"
        if overall["bias"] > 0
        else "  => forecasts are PESSIMISTIC, contrary to prediction"
    )

    print(f"\n  by stratum:  {'stratum':<12} {'cells':>6} {'MAE':>9} {'bias':>10} {'RMST bias':>11}")
    for st, part in cells.groupby("plan_type", observed=True):
        m = error_metrics(part["forecast_S"].tolist(), part["realised_S"].tolist())
        rm = error_metrics(
            part["forecast_RMST"].tolist(), part["realised_RMST"].tolist()
        )
        print(
            f"               {st:<12} {m['n']:>6} {m['mae']:>9.4f} "
            f"{m['bias']:>+10.4f} {rm['bias']:>+11.1f}d"
        )

    rel = truncation_error_relationship(cells)
    print(f"\n  truncation vs |error| across {rel['n_cells']} cells:")
    if rel["correlation"] is None:
        print("    too few cells to correlate")
    else:
        c = rel["correlation"]
        print(f"    Pearson r = {c:+.3f}")
        verdict = (
            "error GROWS with truncation -- delayed entry is hurting"
            if c > 0.3
            else "error FALLS with truncation"
            if c < -0.3
            else "no meaningful relationship -- delayed entry is not the driver"
        )
        print(f"    => {verdict}")

    print("\n  truncation quartiles:")
    q = cells.copy()
    q["bucket"] = pd.qcut(q["truncated_share"], 4, duplicates="drop")
    for b, part in q.groupby("bucket", observed=True):
        m = error_metrics(part["forecast_S"].tolist(), part["realised_S"].tolist())
        print(
            f"    {b!s:<22} cells {m['n']:>3}  MAE {m['mae']:.4f}  "
            f"bias {m['bias']:+.4f}"
        )


def main() -> dict:
    con = duckdb.connect(str(MARTS_DB), read_only=True)
    df = load(con)
    con.close()
    print(f"Loaded {len(df):,} subjects; fit window ends {FIT_WINDOW_END}")
    print(f"Censored fit view: {len(censor_at_fit_window(df, FIT_WINDOW_END)):,} subjects")

    results = {}
    for horizon, tag in [(MIN_FOLLOWUP_DAYS, "PRIMARY"), (SECONDARY_FOLLOWUP_DAYS, "SECONDARY")]:
        cells, overall = run_horizon(df, horizon)
        results[horizon] = overall
        report(cells, overall, horizon, tag)
        print(f"\n  worst 5 cells by |error| ({horizon}d):")
        worst = cells.nlargest(5, "abs_error")[
            ["cohort", "plan_type", "n", "truncated_share", "forecast_S", "realised_S", "error_S"]
        ]
        print(worst.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
