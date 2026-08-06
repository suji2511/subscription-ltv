"""Spell records for the 432,592 subjects excluded for having no members row.

These users cannot enter the main spell table: with no `registration_init_time`
there is no clock to measure tenure on and no way to compute a delayed-entry
time. But they are not missing at random -- they churn roughly twice as fast
than retained subjects (see docs/decisions.md) -- so excluding them biases every
survival curve upward.

This module rebuilds them on the only clock available, their first transaction,
with entry = 0. That makes them incident subjects by construction. The result
is not a drop-in replacement for the main table: it mixes two clocks, and the
combined curve is a bracketing device rather than an estimate. Its purpose is
to put a floor under the survival curve so the true curve can be bounded from
both sides instead of only from above.

Run:  python -m src.cohorts.excluded_spells
"""

from __future__ import annotations

import duckdb

from src.cohorts.spells import SPELL_ORDER
from src.config import CHURN_GRACE_DAYS, DB, LABEL_CUTOFF, MARTS_DB

EXCLUDED_DDL = """
CREATE OR REPLACE TABLE spells_excluded (
    msno              VARCHAR NOT NULL,
    cohort_month      VARCHAR NOT NULL,
    spell_start_date  DATE NOT NULL,
    entry_days        INTEGER NOT NULL,   -- always 0: incident by construction
    tenure_days       INTEGER NOT NULL,
    event             BOOLEAN NOT NULL,
    end_date          DATE NOT NULL,
    ended_by_cancel   BOOLEAN NOT NULL,
    n_transactions    INTEGER NOT NULL,
    first_plan_days   INTEGER,
    plan_type         VARCHAR NOT NULL
)
"""


def build_sql() -> str:
    """Same spell segmentation as the main table, clocked from first transaction.

    Assumption: the first transaction is the start of the subscription for
    these users. What could violate it: if they too are left-truncated -- i.e.
    subscribing before 2015-01-01 -- their tenure is understated and this curve
    is too pessimistic. That is the correct direction for a lower bound, so the
    assumption is conservative for the purpose it serves, but it does mean the
    bracket is wider than the truth on that side.
    """
    return f"""
    WITH excluded_users AS (
        SELECT DISTINCT msno FROM staged.transactions
        EXCEPT
        SELECT msno FROM staged.members
    ),
    tx AS (
        SELECT t.* FROM staged.transactions t
        JOIN excluded_users USING (msno)
    ),
    ordered AS (
        SELECT tx.*,
            MAX(CASE WHEN NOT is_cancel THEN membership_expire_date END) OVER (
                PARTITION BY msno ORDER BY {SPELL_ORDER}
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS coverage_so_far
        FROM tx
    ),
    flagged AS (
        SELECT *, CASE
            WHEN coverage_so_far IS NULL THEN 1
            WHEN DATE_DIFF('day', coverage_so_far, transaction_date)
                 > {CHURN_GRACE_DAYS} THEN 1 ELSE 0 END AS opens_spell
        FROM ordered
    ),
    numbered AS (
        SELECT *, SUM(opens_spell) OVER (
            PARTITION BY msno ORDER BY {SPELL_ORDER}
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS spell_no
        FROM flagged
    ),
    agg AS (
        SELECT
            msno,
            MIN(transaction_date) AS spell_start_date,
            COUNT(*) AS n_transactions,
            MAX_BY(membership_expire_date,
                   {{'d': transaction_date, 'e': membership_expire_date}})
                FILTER (WHERE NOT is_cancel
                        AND membership_expire_date IS NOT NULL)
                AS last_noncancel_expiry,
            MAX(transaction_date) FILTER (WHERE NOT is_cancel) AS last_noncancel_date,
            MAX(transaction_date) FILTER (WHERE is_cancel) AS last_cancel_date,
            MIN_BY(plan_days,
                   {{'d': transaction_date,
                     'e': COALESCE(membership_expire_date, DATE '9999-12-31'),
                     'c': is_cancel}}) AS first_plan_days
        FROM numbered WHERE spell_no = 1
        GROUP BY msno
    ),
    resolved AS (
        SELECT a.*,
            COALESCE(a.last_cancel_date > a.last_noncancel_date, FALSE) AS ended_by_cancel,
            CASE WHEN COALESCE(a.last_cancel_date > a.last_noncancel_date, FALSE)
                 THEN a.last_cancel_date ELSE a.last_noncancel_expiry END AS raw_end_date
        FROM agg a
    )
    SELECT
        msno,
        STRFTIME(spell_start_date, '%Y-%m') AS cohort_month,
        spell_start_date,
        0 AS entry_days,
        DATE_DIFF('day', spell_start_date,
            CASE WHEN (CASE WHEN ended_by_cancel
                            THEN raw_end_date <= DATE '{LABEL_CUTOFF}'
                            ELSE raw_end_date + INTERVAL {CHURN_GRACE_DAYS} DAY
                                 <= DATE '{LABEL_CUTOFF}' END)
                 THEN raw_end_date
                 ELSE LEAST(raw_end_date, DATE '{LABEL_CUTOFF}') END
        ) AS tenure_days,
        CASE WHEN ended_by_cancel THEN raw_end_date <= DATE '{LABEL_CUTOFF}'
             ELSE raw_end_date + INTERVAL {CHURN_GRACE_DAYS} DAY
                  <= DATE '{LABEL_CUTOFF}' END AS event,
        CASE WHEN (CASE WHEN ended_by_cancel
                        THEN raw_end_date <= DATE '{LABEL_CUTOFF}'
                        ELSE raw_end_date + INTERVAL {CHURN_GRACE_DAYS} DAY
                             <= DATE '{LABEL_CUTOFF}' END)
             THEN raw_end_date ELSE LEAST(raw_end_date, DATE '{LABEL_CUTOFF}') END AS end_date,
        ended_by_cancel,
        n_transactions,
        first_plan_days,
        CASE
            WHEN first_plan_days IS NULL THEN 'unknown'
            WHEN first_plan_days <= 7   THEN '01_<=7d'
            WHEN first_plan_days <= 31  THEN '02_8-31d'
            WHEN first_plan_days <= 120 THEN '03_32-120d'
            ELSE '04_121d+'
        END AS plan_type
    FROM resolved
    WHERE raw_end_date IS NOT NULL
      AND spell_start_date <= DATE '{LABEL_CUTOFF}'
    """


def main() -> None:
    con = duckdb.connect(str(MARTS_DB))
    con.execute(f"ATTACH IF NOT EXISTS '{DB}' AS staged (READ_ONLY)")
    con.execute(EXCLUDED_DDL)
    con.execute(f"INSERT INTO spells_excluded {build_sql()}")

    n, ev = con.execute(
        "SELECT COUNT(*), SUM(event::INT) FROM spells_excluded"
    ).fetchone()
    neg = con.execute(
        "SELECT COUNT(*) FROM spells_excluded WHERE tenure_days < 0"
    ).fetchone()[0]
    print(f"spells_excluded: {n:,} subjects, {ev:,} events, censoring {1 - ev / n:.2%}")
    print(f"  negative tenure (should be 0): {neg}")
    for r in con.execute(
        "SELECT plan_type, COUNT(*), SUM(event::INT) FROM spells_excluded GROUP BY 1 ORDER BY 1"
    ).fetchall():
        print(f"  {r[0]:<12} {r[1]:>9,} subjects  {r[2]:>9,} events")
    con.close()


if __name__ == "__main__":
    main()
