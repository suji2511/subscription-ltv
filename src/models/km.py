"""Kaplan-Meier survival estimation with delayed entry.

Phase 4 is deliberately Kaplan-Meier only. No Cox: the point of fitting the
non-parametric estimator first is to find out whether proportional hazards is
defensible before assuming it, and crossing survival curves are the clearest
evidence that it is not.

Every fit here passes `entry=entry_days`. Left truncation is severe -- 54.9% of
subjects accrued tenure before the observation window opened -- so omitting
`entry` would treat long-tenured survivors as new signups and bias every curve
upward.

Every survival estimate is reported with its number at risk. Under delayed
entry the risk set GROWS with t rather than shrinking, so a survival figure
without its denominator is uninterpretable: S(30) here rests on 46% of the
sample, S(730) on 71%.

Run:  python -m src.models.km
"""

from __future__ import annotations

import duckdb
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from src.config import MARTS_DB, MIN_EVENTS_PER_CELL

HORIZONS = [30, 60, 90, 180, 365, 545, 730]


def load_spells(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load the subject-level spell table (pandas: lifelines requires it)."""
    return con.execute(
        """
        SELECT msno, cohort_month, plan_type, registered_via,
               entry_days, tenure_days, event
        FROM spells
        """
    ).df()


def load_with_excluded(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Main subjects plus the no-members-row subjects at entry = 0.

    Mixes two clocks -- registration for the main table, first transaction for
    the excluded -- so this is a bracketing device, not an estimate.
    """
    return con.execute(
        """
        SELECT plan_type, entry_days, tenure_days, event FROM spells
        UNION ALL
        SELECT plan_type, entry_days, tenure_days, event FROM spells_excluded
        """
    ).df()


def fit(df: pd.DataFrame, label: str) -> KaplanMeierFitter | None:
    """Fit one delayed-entry KM curve, or None if the cell has too few events.

    Cells are screened on EVENTS, not subjects: Greenwood's variance sums over
    event times only, so a cell with many censored subjects and few deaths
    estimates nothing.

    Assumption: subjects are independent. Guaranteed by the first-spell-only
    rule -- one subject per user, so no within-user clustering exists to
    invalidate the Greenwood intervals.
    """
    if int(df["event"].sum()) < MIN_EVENTS_PER_CELL:
        return None
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(
        durations=df["tenure_days"],
        event_observed=df["event"],
        entry=df["entry_days"],
    )
    return kmf


def at_risk(kmf: KaplanMeierFitter, t: int) -> int:
    """Number of subjects under observation at time t."""
    tbl = kmf.event_table
    idx = tbl.index[tbl.index <= t]
    return int(tbl.loc[idx[-1], "at_risk"]) if len(idx) else 0


def compact(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def describe(kmf: KaplanMeierFitter, n: int, events: int) -> dict:
    """Median survival with CI, and S(t) paired with its number at risk."""
    ci = median_survival_times(kmf.confidence_interval_)
    row = {
        "n": f"{n:,}",
        "events": f"{events:,}",
        "median": fmt_days(kmf.median_survival_time_),
        "95% CI": f"[{fmt_days(ci.iloc[0, 0])}, {fmt_days(ci.iloc[0, 1])}]",
    }
    for h in HORIZONS:
        s = float(kmf.survival_function_at_times([h]).iloc[0])
        row[f"S({h})"] = f"{s:.3f} @{compact(at_risk(kmf, h))}"
    return row


def fmt_days(v) -> str:
    if v is None or pd.isna(v) or v == float("inf"):
        return "inf"
    return f"{v:,.0f}"


def fit_by(df: pd.DataFrame, column: str, title: str) -> None:
    """Fit one curve per level of `column` and tabulate."""
    rows = {}
    for level, part in df.groupby(column, observed=True):
        kmf = fit(part, label=str(level))
        if kmf is not None:
            rows[str(level)] = describe(kmf, len(part), int(part["event"].sum()))
    print(f"\n=== {title} ===")
    print("  (S(t) shown as  value @number-at-risk)")
    print(pd.DataFrame(rows).T.to_string())


def truncation_profile(df: pd.DataFrame) -> None:
    """Left-truncation exposure per stratum -- the context for any crossing."""
    print("\n=== TRUNCATION PROFILE BY STRATUM ===")
    rows = {}
    for level, part in df.groupby("plan_type", observed=True):
        trunc = part["entry_days"] > 0
        rows[str(level)] = {
            "n": f"{len(part):,}",
            "truncated": f"{trunc.mean():.1%}",
            "median entry": f"{part['entry_days'].median():,.0f}d",
            "median entry | truncated": f"{part.loc[trunc, 'entry_days'].median():,.0f}d"
            if trunc.any()
            else "-",
            "incident (entry=0)": f"{(~trunc).sum():,}",
        }
    print(pd.DataFrame(rows).T.to_string())


def crossing(a: KaplanMeierFitter, b: KaplanMeierFitter) -> tuple[bool, str]:
    """Do two curves cross on the horizon grid?"""
    diff = [
        float(a.survival_function_at_times([h]).iloc[0])
        - float(b.survival_function_at_times([h]).iloc[0])
        for h in HORIZONS
    ]
    signs = {d > 0 for d in diff if abs(d) > 1e-9}
    detail = " ".join(f"{h}d:{d:+.3f}" for h, d in zip(HORIZONS, diff))
    return len(signs) > 1, detail


def check_crossings(df: pd.DataFrame, title: str, column: str = "plan_type") -> bool:
    """Report whether stratum curves cross.

    Crossing curves are the clearest evidence against proportional hazards:
    under PH the hazard ratio is constant, so curves are ordered identically at
    every t and cannot cross. A variable whose curves cross belongs in
    `strata=`, not on the right-hand side of a Cox model.
    """
    curves = {}
    for level, part in df.groupby(column, observed=True):
        kmf = fit(part, label=str(level))
        if kmf is not None:
            curves[str(level)] = kmf

    print(f"\n=== CROSSING CHECK: {title} ===")
    levels = sorted(curves)
    any_cross = False
    for i, a in enumerate(levels):
        for b in levels[i + 1 :]:
            crossed, detail = crossing(curves[a], curves[b])
            any_cross |= crossed
            print(f"  {'CROSS ' if crossed else 'ok    '} {a} - {b}:  {detail}")
    print(
        "  => PH UNSAFE for this variable" if any_cross else "  => no crossings found"
    )
    return any_cross


def incident_only_check(df: pd.DataFrame) -> None:
    """Refit the two crossing strata on incident subjects only (entry == 0).

    The 60-90d crossing between <=7d and 8-31d appeared on curves whose early
    portions rest on very different subpopulations: the strata differ sharply
    in truncation. Restricting to entry == 0 removes delayed entry entirely, so
    if the crossing survives it is a property of the products; if it vanishes
    it was an artefact of who was under observation early on.
    """
    pair = df[df["plan_type"].isin(["01_<=7d", "02_8-31d"])]
    incident = pair[pair["entry_days"] == 0]

    print("\n=== INCIDENT-ONLY REFIT (entry_days == 0) ===")
    print(f"  full pair:     {len(pair):,} subjects")
    print(f"  incident only: {len(incident):,} subjects ({len(incident) / len(pair):.1%})")

    rows = {}
    for level, part in incident.groupby("plan_type", observed=True):
        kmf = fit(part, label=str(level))
        if kmf is not None:
            rows[str(level)] = describe(kmf, len(part), int(part["event"].sum()))
    print(pd.DataFrame(rows).T.to_string())

    crossed_full, detail_full = crossing(
        fit(pair[pair["plan_type"] == "01_<=7d"], "a"),
        fit(pair[pair["plan_type"] == "02_8-31d"], "b"),
    )
    crossed_inc, detail_inc = crossing(
        fit(incident[incident["plan_type"] == "01_<=7d"], "a"),
        fit(incident[incident["plan_type"] == "02_8-31d"], "b"),
    )
    print(f"\n  all subjects  (<=7d - 8-31d): {detail_full}")
    print(f"    crossing: {crossed_full}")
    print(f"  incident only (<=7d - 8-31d): {detail_inc}")
    print(f"    crossing: {crossed_inc}")
    if crossed_full and not crossed_inc:
        print("\n  => the crossing is a LEFT-TRUNCATION ARTEFACT, not a product effect")
    elif crossed_inc:
        print("\n  => the crossing PERSISTS on incident subjects: a real product effect")


def exclusion_bracket(main: pd.DataFrame, both: pd.DataFrame) -> None:
    """Bracket the true curve using the no-members-row subjects.

    The main curve excludes 432,592 users who churn about twice as fast, so it
    is an upper bound. Adding them back as incident subjects overstates their
    churn if they are themselves truncated, so that curve is closer to a lower
    bound. The truth lies between.
    """
    print("\n=== EXCLUSION BRACKET ===")
    upper = fit(main, "upper (excludes no-members)")
    lower = fit(both, "lower (includes at entry=0)")
    rows = {
        "UPPER: main table only": describe(
            upper, len(main), int(main["event"].sum())
        ),
        "LOWER: + no-members at entry=0": describe(
            lower, len(both), int(both["event"].sum())
        ),
    }
    print(pd.DataFrame(rows).T.to_string())
    print(f"\n  {'t':>6} {'upper':>8} {'lower':>8} {'width pp':>10}")
    for h in HORIZONS:
        u = float(upper.survival_function_at_times([h]).iloc[0])
        low = float(lower.survival_function_at_times([h]).iloc[0])
        print(f"  {h:>5}d {u:>8.4f} {low:>8.4f} {(u - low) * 100:>9.2f}")


def via3_sensitivity(df: pd.DataFrame) -> None:
    """Delayed-entry KM with and without registered_via = 3.

    via 3 carries the worst registration-to-first-transaction lag (median 45
    days, p90 459), so its entry times are the least trustworthy in the sample.
    """
    print("\n=== SENSITIVITY: registered_via = 3 ===")
    full = fit(df, "all subjects")
    rest = df[df["registered_via"] != 3]
    without = fit(rest, "excluding via 3")
    n3 = int((df["registered_via"] == 3).sum())
    print(f"  via 3 subjects: {n3:,} ({n3 / len(df):.1%})")
    rows = {
        "all subjects": describe(full, len(df), int(df["event"].sum())),
        "excluding via 3": describe(without, len(rest), int(rest["event"].sum())),
    }
    print(pd.DataFrame(rows).T.to_string())

    # Significance and materiality are different questions at n = 1.87M: the
    # 95% band is ~0.001 wide, so almost any subgroup removal is "significant".
    # The magnitude is what decides whether LTV conclusions change.
    material_pp = 1.0
    band = full.confidence_interval_
    print(f"\n  {'t':>6} {'all':>8} {'excl':>8} {'diff pp':>9} {'rel':>8}  {'CI':>8}")
    max_pp = 0.0
    for h in HORIZONS:
        idx = band.index[band.index <= h]
        if len(idx) == 0:
            continue
        lo, hi = band.loc[idx[-1]].values
        s_all = float(full.survival_function_at_times([h]).iloc[0])
        s_ex = float(without.survival_function_at_times([h]).iloc[0])
        pp = (s_ex - s_all) * 100
        max_pp = max(max_pp, abs(pp))
        ci = "outside" if (s_ex < lo or s_ex > hi) else "inside"
        print(
            f"  {h:>5}d {s_all:>8.4f} {s_ex:>8.4f} {pp:>+9.2f} "
            f"{pp / (s_all * 100):>+7.1%}  {ci:>8}"
        )
    print(f"\n  largest shift {max_pp:.2f}pp vs materiality threshold {material_pp:.1f}pp")
    print(
        "  => MATERIAL at long horizons; LTV must be a range"
        if max_pp >= material_pp
        else "  => detectable but immaterial"
    )


def main() -> None:
    con = duckdb.connect(str(MARTS_DB), read_only=True)
    df = load_spells(con)
    both = load_with_excluded(con)
    con.close()
    print(f"Loaded {len(df):,} subjects, {int(df['event'].sum()):,} events")

    pooled = fit(df, "pooled")
    print("\n=== POOLED CURVE — A MIXTURE, NOT A POPULATION CURVE ===")
    print("  Pools <=7d promos with 121d+ annual plans; describes no real product.")
    print(
        pd.DataFrame(
            {"pooled (MIXTURE)": describe(pooled, len(df), int(df["event"].sum()))}
        ).T.to_string()
    )

    fit_by(df, "plan_type", "BY PLAN TYPE (primary stratum)")
    fit_by(df, "cohort_month", "BY COHORT MONTH")
    truncation_profile(df)
    check_crossings(df, "plan_type")
    incident_only_check(df)
    exclusion_bracket(df, both)
    via3_sensitivity(df)


if __name__ == "__main__":
    main()
