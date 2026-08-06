"""Lifetime value: survival-weighted revenue, plus an empirical reactivation term.

LTV(H) = revenue_per_subscribed_day  x  RMST(H)  +  reactivation_term

RMST(H) is the restricted mean survival time -- the area under the Kaplan-Meier
curve out to H -- which is the expected number of days a subject remains
subscribed within the first H days. Multiplying by revenue per subscribed day
converts expected days into expected revenue.

Reported as a RANGE, never a point. Two independently measured uncertainties
bound it, and both were quantified before this module was written:

  exclusion bracket  432,592 users with no members row churn about twice as
                     fast. Excluding them gives the upper survival curve;
                     including them at entry=0 gives the lower. At 365 days
                     that is 6.1 percentage points of survival.
  via-3 sensitivity  entry times for registered_via = 3 are the least
                     trustworthy in the sample; removing that group moves
                     survival by up to 2.93pp at 730 days.

Assumption behind ARPD x RMST: revenue accrues uniformly over the tenure the
survival curve measures. What could violate it: the curve's clock starts at
registration while revenue starts at the first transaction, so for the 54.9% of
subjects who are left-truncated the two clocks are offset. Revenue per day is
therefore computed over OBSERVED spell days, which keeps numerator and
denominator on the same basis, but the product still assumes that rate applies
across the whole RMST window. This is the weakest link in the LTV chain and is
why the range matters more than the midpoint.

Run:  python -m src.models.ltv
"""

from __future__ import annotations

import duckdb
import pandas as pd
from lifelines.utils import restricted_mean_survival_time

from src.config import MARTS_DB, OBSERVATION_CUTOFF
from src.models.km import fit

# LTV is quoted at the horizon the backtest will be able to check.
HORIZONS = [365, 730]


def load(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spells = con.execute(
        """
        SELECT s.plan_type, s.registered_via, s.entry_days, s.tenure_days, s.event
        FROM spells s
        """
    ).df()
    both = con.execute(
        """
        SELECT plan_type, entry_days, tenure_days, event FROM spells
        UNION ALL
        SELECT plan_type, entry_days, tenure_days, event FROM spells_excluded
        """
    ).df()
    rev = con.execute(
        """
        SELECT r.plan_type, s.entry_days,
               r.first_spell_rev, r.observed_days, r.imputed_rev
        FROM spell_revenue r JOIN spells s USING (msno)
        """
    ).df()
    return spells, both, rev


def rmst(df: pd.DataFrame, horizon: int, label: str) -> float | None:
    """Expected subscribed days within `horizon`, from the delayed-entry curve."""
    kmf = fit(df, label)
    if kmf is None:
        return None
    return float(restricted_mean_survival_time(kmf, t=horizon))


def revenue_per_day(
    rev: pd.DataFrame, stratum: str, incident_only: bool = False
) -> tuple[float, float]:
    """Revenue per observed subscribed day, and the same excluding imputed rows.

    `incident_only` restricts to entry_days == 0, where the revenue clock and
    the survival clock coincide by construction. That removes the uniform-
    accrual extrapolation entirely, which is what makes the two figures
    comparable evidence rather than a caveat.
    """
    part = rev[rev["plan_type"] == stratum]
    if incident_only:
        part = part[part["entry_days"] == 0]
    days = part["observed_days"].sum()
    if days == 0:
        return 0.0, 0.0
    total = part["first_spell_rev"].sum()
    return total / days, (total - part["imputed_rev"].sum()) / days


def reactivation_term(
    con: duckdb.DuckDBPyConnection,
    stratum: str,
    window: int,
    full_followup: bool = True,
) -> dict:
    """Empirical reactivation value: P(return within window) x mean later revenue.

    Arithmetic on observed quantities, per Decision 1 -- not a second survival
    model. The rate is conditioned on an observed first-spell churn, because a
    censored subject has not churned and cannot be asked whether they returned.
    """
    # Only churners with a FULL `window` of post-churn follow-up can be observed
    # returning. Including the rest censors the rate downward -- at 365 days the
    # uncorrected rate is 33.92% against a corrected 40.85%, understating LTV.
    # This is the one known bias that runs OPPOSITE to delayed entry and the
    # exclusion bracket, both of which overstate.
    followup = (
        f"AND first_end_date <= DATE '{OBSERVATION_CUTOFF}' - INTERVAL {window} DAY"
        if full_followup
        else ""
    )
    r = con.execute(
        f"""
        SELECT COUNT(*),
               AVG((reactivated AND days_to_return <= {window})::INT),
               AVG(later_spell_rev) FILTER (WHERE reactivated AND days_to_return <= {window})
        FROM reactivation WHERE plan_type = ? {followup}
        """,
        [stratum],
    ).fetchone()
    n, rate, mean_rev = r
    rate = rate or 0.0
    mean_rev = mean_rev or 0.0
    return {
        "churned_subjects": n,
        "reactivation_rate": rate,
        "mean_revenue_per_reactivation": mean_rev,
        "value": rate * mean_rev,
    }


def ltv_table(
    con: duckdb.DuckDBPyConnection, horizon: int, incident_only: bool = False
) -> pd.DataFrame:
    spells, both, rev = load(con)
    if incident_only:
        spells = spells[spells["entry_days"] == 0]
        both = both[both["entry_days"] == 0]
    strata = sorted(s for s in spells["plan_type"].unique() if s != "unknown")

    rows = {}
    for st in strata:
        main = spells[spells["plan_type"] == st]
        rpd, rpd_no_imp = revenue_per_day(rev, st, incident_only=incident_only)

        # Upper bound: main table only. Lower bound: plus the excluded users.
        r_upper = rmst(main, horizon, st)
        r_lower = rmst(both[both["plan_type"] == st], horizon, st)
        # via-3 sensitivity, a second independent source of uncertainty.
        r_via3 = rmst(main[main["registered_via"] != 3], horizon, st)
        if r_upper is None:
            continue

        candidates = [r for r in (r_upper, r_lower, r_via3) if r is not None]
        lo_rmst, hi_rmst = min(candidates), max(candidates)

        react = reactivation_term(con, st, horizon)
        rows[st] = {
            "subjects": f"{len(main):,}",
            "rev/day": f"{rpd:.3f}",
            "RMST lo": f"{lo_rmst:,.0f}",
            "RMST hi": f"{hi_rmst:,.0f}",
            "LTV lo": f"{rpd * lo_rmst + react['value']:,.0f}",
            "LTV hi": f"{rpd * hi_rmst + react['value']:,.0f}",
            "spell1 (mid)": f"{rpd * (lo_rmst + hi_rmst) / 2:,.0f}",
            "reactiv.": f"{react['value']:,.0f}",
            "react %": f"{react['value'] / (rpd * (lo_rmst + hi_rmst) / 2 + react['value']):.1%}",
            "LTV mid": f"{rpd * (lo_rmst + hi_rmst) / 2 + react['value']:,.0f}",
            "LTV mid ex-imputed": f"{rpd_no_imp * (lo_rmst + hi_rmst) / 2 + react['value']:,.0f}",
        }
    return pd.DataFrame(rows).T


def reactivation_detail(con: duckdb.DuckDBPyConnection) -> None:
    print("\n=== REACTIVATION COMPONENT (empirical, not modelled) ===")
    print("  Conditioned on an OBSERVED first-spell churn.\n")
    print(f"  {'stratum':<12} {'churned':>10} {'rate@365d':>11} {'mean rev':>10} {'value':>9}")
    for st in ["01_<=7d", "02_8-31d", "03_32-120d", "04_121d+"]:
        d = reactivation_term(con, st, 365)
        u = reactivation_term(con, st, 365, full_followup=False)
        print(
            f"  {st:<12} {d['churned_subjects']:>10,} "
            f"{d['reactivation_rate']:>10.2%} "
            f"{d['mean_revenue_per_reactivation']:>10,.0f} {d['value']:>9,.0f}"
            f"   (uncorrected rate {u['reactivation_rate']:.2%})"
        )
    for w in [90, 180, 365]:
        r = con.execute(
            f"""
            SELECT AVG((reactivated AND days_to_return <= {w})::INT),
                   AVG(later_spell_rev) FILTER (WHERE reactivated AND days_to_return <= {w}),
                   MEDIAN(days_to_return) FILTER (WHERE reactivated AND days_to_return <= {w})
            FROM reactivation
            WHERE first_end_date <= DATE '{OBSERVATION_CUTOFF}' - INTERVAL {w} DAY
            """
        ).fetchone()
        print(
            f"\n  all strata (full follow-up), within {w:>3}d: rate {r[0]:.2%}, "
            f"mean revenue {r[1]:,.0f}, median days to return {r[2]:,.0f}"
        )


def clock_sensitivity(con: duckdb.DuckDBPyConnection, horizon: int) -> None:
    """Full population vs incident-only, where the two clocks coincide.

    If the two agree, the uniform-accrual extrapolation behind rev/day x RMST
    is benign and can be stated as evidence rather than hedged as a caveat.
    """
    full = ltv_table(con, horizon)
    inc = ltv_table(con, horizon, incident_only=True)
    print(f"\n=== CLOCK SENSITIVITY AT {horizon}d: full population vs incident-only ===")
    print("  Incident subjects (entry_days == 0) have revenue and survival on the")
    print("  SAME clock, so they carry no uniform-accrual assumption.\n")
    print(f"  {'stratum':<12} {'LTV mid full':>13} {'LTV mid incid':>14} {'diff':>9} {'rel':>8}")
    for st in full.index:
        if st not in inc.index:
            continue
        a = float(full.loc[st, "LTV mid"].replace(",", ""))
        b = float(inc.loc[st, "LTV mid"].replace(",", ""))
        print(f"  {st:<12} {a:>13,.0f} {b:>14,.0f} {b - a:>+9,.0f} {(b - a) / a:>+7.1%}")


def main() -> None:
    con = duckdb.connect(str(MARTS_DB), read_only=True)
    for h in HORIZONS:
        print(f"\n=== HEADLINE: LTV AT {h} DAYS, INCIDENT SUBJECTS (entry_days == 0) ===")
        print("  Incident-only is the headline because the clock-sensitivity check")
        print("  below shows the full-population figure diverges materially --")
        print("  the uniform-accrual extrapolation is NOT benign.")
        print("  Range spans the exclusion bracket and the via-3 sensitivity.")
        print("  LTV lo/hi/mid INCLUDE reactivation; spell1 column excludes it.\n")
        print(ltv_table(con, h, incident_only=True).to_string())
        print(f"\n  --- sensitivity: full population at {h}d ---")
        print(ltv_table(con, h).to_string())
    for h in HORIZONS:
        clock_sensitivity(con, h)
    reactivation_detail(con)
    con.close()


if __name__ == "__main__":
    main()
