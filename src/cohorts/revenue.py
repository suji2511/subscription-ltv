"""staged -> marts: revenue per subject, and the reactivation component.

Two tables:

  spell_revenue   one row per subject in `spells`: revenue earned during the
                  first spell, and the observed days it was earned over.
  reactivation    one row per subject whose first spell ended in an observed
                  churn: whether they came back, how long they took, and what
                  the later spells were worth.

The reactivation table is what Decision 1 promised. Having chosen first-spell-
only for the survival model, the returning behaviour of 19.81% of users is not
thrown away -- it is measured here as an empirical rate and an average revenue,
which is arithmetic on observed quantities rather than a second correlated
survival model.

Run:  python -m src.cohorts.revenue
"""

from __future__ import annotations

import duckdb

from src.cohorts.spells import SPELL_ORDER
from src.config import CHURN_GRACE_DAYS, DB, LABEL_CUTOFF, MARTS_DB

# Reactivation is reported at several windows because "did they come back"
# has no single right horizon; the rate is meaningless without one attached.
REACTIVATION_WINDOWS = [90, 180, 365]

SPELL_REVENUE_DDL = """
CREATE OR REPLACE TABLE spell_revenue (
    msno              VARCHAR NOT NULL,
    plan_type         VARCHAR NOT NULL,
    first_spell_rev   INTEGER NOT NULL,
    observed_days     INTEGER NOT NULL,  -- spell_start -> end, the days revenue accrued over
    imputed_rev       INTEGER NOT NULL,  -- revenue from plan_days_imputed rows
    event             BOOLEAN NOT NULL
)
"""

REACTIVATION_DDL = """
CREATE OR REPLACE TABLE reactivation (
    msno              VARCHAR NOT NULL,
    plan_type         VARCHAR NOT NULL,
    first_end_date    DATE NOT NULL,
    reactivated       BOOLEAN NOT NULL,
    days_to_return    INTEGER,
    later_spell_rev   INTEGER NOT NULL,
    n_later_spells    INTEGER NOT NULL
)
"""


def numbered_sql() -> str:
    """Transactions with their spell number, using the same segmentation as spells.py."""
    return f"""
    WITH ordered AS (
        SELECT t.*,
            MAX(CASE WHEN NOT is_cancel THEN membership_expire_date END) OVER (
                PARTITION BY msno ORDER BY {SPELL_ORDER}
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS coverage_so_far
        FROM staged.transactions t
    ),
    flagged AS (
        SELECT *, CASE
            WHEN coverage_so_far IS NULL THEN 1
            WHEN DATE_DIFF('day', coverage_so_far, transaction_date)
                 > {CHURN_GRACE_DAYS} THEN 1 ELSE 0 END AS opens_spell
        FROM ordered
    )
    SELECT *, SUM(opens_spell) OVER (
        PARTITION BY msno ORDER BY {SPELL_ORDER}
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS spell_no
    FROM flagged
    """


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"CREATE OR REPLACE TEMP TABLE numbered AS {numbered_sql()}")
    con.execute(SPELL_REVENUE_DDL)
    con.execute(REACTIVATION_DDL)

    # Revenue during the first spell. Cancellations carry no revenue, so they
    # are excluded from the sum rather than netted off -- actual_amount_paid on
    # a cancel row is not a refund, it is the amount of the original purchase.
    con.execute(
        """
        INSERT INTO spell_revenue
        SELECT
            s.msno, s.plan_type,
            COALESCE(SUM(n.actual_amount_paid) FILTER (WHERE NOT n.is_cancel), 0),
            DATE_DIFF('day', s.spell_start_date, s.end_date),
            COALESCE(SUM(n.actual_amount_paid)
                     FILTER (WHERE NOT n.is_cancel AND n.plan_days_imputed), 0),
            ANY_VALUE(s.event)
        FROM spells s
        JOIN numbered n ON n.msno = s.msno AND n.spell_no = 1
        GROUP BY s.msno, s.plan_type, s.spell_start_date, s.end_date
        """
    )

    # Reactivation, defined only for subjects whose first spell ended in an
    # OBSERVED churn. A censored subject has not churned, so "did they come
    # back" is not a question that can be asked of them, and including them
    # would deflate the rate by its denominator.
    con.execute(
        f"""
        INSERT INTO reactivation
        WITH churned AS (
            SELECT msno, plan_type, end_date FROM spells WHERE event
        ),
        later AS (
            SELECT msno,
                   MIN(transaction_date) AS first_return_date,
                   SUM(actual_amount_paid) FILTER (WHERE NOT is_cancel) AS rev,
                   COUNT(DISTINCT spell_no) AS n_spells
            FROM numbered WHERE spell_no > 1
            GROUP BY msno
        )
        SELECT
            c.msno, c.plan_type, c.end_date,
            l.first_return_date IS NOT NULL AS reactivated,
            DATE_DIFF('day', c.end_date, l.first_return_date) AS days_to_return,
            COALESCE(l.rev, 0), COALESCE(l.n_spells, 0)
        FROM churned c LEFT JOIN later l USING (msno)
        WHERE c.end_date <= DATE '{LABEL_CUTOFF}'
        """
    )


def report(con: duckdb.DuckDBPyConnection) -> None:
    n, rev = con.execute(
        "SELECT COUNT(*), SUM(first_spell_rev) FROM spell_revenue"
    ).fetchone()
    imp = con.execute("SELECT SUM(imputed_rev) FROM spell_revenue").fetchone()[0]
    print(f"spell_revenue : {n:,} subjects, total revenue {rev:,}")
    print(f"  revenue from imputed rows: {imp:,} ({imp / rev:.2%} of total)")
    print("  (every revenue figure is reproducible with imputed rows excluded)")

    print("\n  revenue per subscribed day, by stratum:")
    print(f"  {'stratum':<12} {'subjects':>10} {'revenue':>14} {'obs days':>14} {'rev/day':>9}")
    for r in con.execute(
        """
        SELECT plan_type, COUNT(*), SUM(first_spell_rev), SUM(observed_days),
               SUM(first_spell_rev) * 1.0 / NULLIF(SUM(observed_days), 0)
        FROM spell_revenue GROUP BY 1 ORDER BY 1
        """
    ).fetchall():
        # rev/day is NULL where a stratum has zero observed days (the single
        # 'unknown' subject, whose spell start and end coincide).
        rpd = f"{r[4]:>9.4f}" if r[4] is not None else f"{'n/a':>9}"
        print(f"  {r[0]:<12} {r[1]:>10,} {r[2]:>14,} {r[3]:>14,} {rpd}")

    n_ch = con.execute("SELECT COUNT(*) FROM reactivation").fetchone()[0]
    print(f"\nreactivation  : {n_ch:,} subjects with an observed first-spell churn")
    for w in REACTIVATION_WINDOWS:
        r = con.execute(
            f"""
            SELECT AVG((reactivated AND days_to_return <= {w})::INT),
                   AVG(later_spell_rev) FILTER (WHERE reactivated AND days_to_return <= {w})
            FROM reactivation
            """
        ).fetchone()
        print(f"  within {w:>3}d: rate {r[0]:>7.2%}   mean later-spell revenue {r[1]:>8,.0f}")


def main() -> None:
    con = duckdb.connect(str(MARTS_DB))
    con.execute(f"ATTACH IF NOT EXISTS '{DB}' AS staged (READ_ONLY)")
    build(con)
    report(con)
    con.close()


if __name__ == "__main__":
    main()
