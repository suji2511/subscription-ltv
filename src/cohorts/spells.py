"""staged -> marts: build the subject-level spell table.

One row per subject. Per docs/decisions.md, a subject is a user's FIRST
observed subscription spell, so the estimand is time to first observed churn
and every row is independent of every other.

The clock runs from `registration_init_time`, not from the first transaction,
because left truncation is handled by delayed entry: subjects enter the risk
set at the tenure already accrued when the observation window opened rather
than at t=0.

Run:  python -m src.cohorts.spells
"""

from __future__ import annotations

import duckdb

from src.config import (
    CHURN_GRACE_DAYS,
    DB,
    LABEL_CUTOFF,
    MARTS_DB,
    WINDOW_OPEN,
)
from src.validate.checks import (
    censoring_rate,
    duplicate_transactions,
    row_count_reconciliation,
    run_checks,
)

STAGE = "staged->marts"

# A TOTAL order over a user's transactions, used by every window function here.
#
# Ordering on (transaction_date, membership_expire_date) alone is not total:
# 27,942 user-days carry more than one transaction, and rows tying on both
# dates but differing in is_cancel got an arbitrary relative order inside the
# ROWS frame. That changed which expiry counted as coverage, which moved spell
# boundaries, which changed the subject count between runs of the same code on
# the same data. Listing every remaining column makes the order total up to
# exact duplicates -- and exact duplicates are interchangeable by definition,
# which `duplicate_transactions` separately verifies do not exist.
SPELL_ORDER = """
    transaction_date, membership_expire_date, is_cancel, payment_method_id,
    plan_days, list_price, actual_amount_paid, is_auto_renew, plan_days_imputed
"""

SPELLS_DDL = """
CREATE OR REPLACE TABLE spells (
    msno              VARCHAR NOT NULL,
    cohort_month      VARCHAR NOT NULL,
    spell_start_date  DATE NOT NULL,
    registration_date DATE NOT NULL,
    entry_days        INTEGER NOT NULL,   -- delayed entry (lifelines `entry=`)
    tenure_days       INTEGER NOT NULL,   -- duration (lifelines `durations=`)
    event             BOOLEAN NOT NULL,   -- churn observed
    end_date          DATE NOT NULL,
    ended_by_cancel   BOOLEAN NOT NULL,
    n_transactions    INTEGER NOT NULL,
    first_plan_days   INTEGER,
    plan_type         VARCHAR NOT NULL,
    registered_via    SMALLINT,
    city              SMALLINT,
    bd                INTEGER,
    gender            VARCHAR
)
"""

SPELLS_QUARANTINE_DDL = """
CREATE OR REPLACE TABLE spells_quarantine (
    msno              VARCHAR NOT NULL,
    cohort_month      VARCHAR,
    spell_start_date  DATE,
    registration_date DATE,
    entry_days        INTEGER,
    tenure_days       INTEGER,
    event             BOOLEAN,
    end_date          DATE,
    ended_by_cancel   BOOLEAN,
    n_transactions    INTEGER,
    quarantine_reason VARCHAR NOT NULL
)
"""


def build_spell_sql() -> str:
    """The spell-table query, as one readable pipeline of named steps.

    Each CTE does one thing:

      ordered       attach to each transaction the running maximum expiry of
                    every PRECEDING non-cancel transaction by the same user.
                    Cancel rows are excluded from that running max because
                    their expiry is not trustworthy (see the backdated
                    cancellation decision) -- a cancel backdated to 2005 must
                    not be allowed to influence spell boundaries.
      flagged       a transaction opens a new spell when it lands more than
                    CHURN_GRACE_DAYS after the coverage accrued so far. The
                    first transaction always opens one.
      numbered      cumulative sum of that flag gives a spell number per user.
      first_spell   keep spell 1 only, per the first-spell-only rule.
      agg           collapse to one row per subject.

    Assumption: transactions form a chronological sequence per user, ordered by
    SPELL_ORDER. What could violate it: two transactions on one date whose true
    business order differs from that ordering. 27,942 user-days carry more than
    one transaction, so this is not hypothetical. The total order makes the
    result *deterministic*, which is not the same as *correct* — if a renewal
    and a cancellation land on one date, nothing in the data says which came
    first, and the spell boundary depends on that. `duplicate_transactions`
    reports the size of the exposure at every build.
    """
    return f"""
    WITH ordered AS (
        SELECT
            t.*,
            MAX(CASE WHEN NOT is_cancel THEN membership_expire_date END) OVER (
                PARTITION BY msno
                ORDER BY {SPELL_ORDER}
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS coverage_so_far
        FROM staged.transactions t
    ),
    flagged AS (
        SELECT *,
            CASE
                WHEN coverage_so_far IS NULL THEN 1
                WHEN DATE_DIFF('day', coverage_so_far, transaction_date)
                     > {CHURN_GRACE_DAYS} THEN 1
                ELSE 0
            END AS opens_spell
        FROM ordered
    ),
    numbered AS (
        SELECT *,
            SUM(opens_spell) OVER (
                PARTITION BY msno
                ORDER BY {SPELL_ORDER}
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS spell_no
        FROM flagged
    ),
    first_spell AS (
        SELECT * FROM numbered WHERE spell_no = 1
    ),
    agg AS (
        SELECT
            msno,
            MIN(transaction_date) AS spell_start_date,
            COUNT(*) AS n_transactions,
            -- Spell end is the expiry of the LAST non-cancel transaction. The
            -- cancel row's own expiry is never read.
            --
            -- Ordered on (date, expiry) rather than date alone: a bare
            -- MAX_BY(expiry, transaction_date) breaks ties arbitrarily, and
            -- 27,942 user-days carry more than one transaction. That made the
            -- result differ between evaluations of this query, which is how a
            -- row ended up in both the spell table and the quarantine.
            -- The ordering key must never contain a NULL. membership_expire_date
            -- is nullable (1,776 epoch sentinels were nulled at ingestion), and
            -- a NULL inside the struct makes the comparison ambiguous: the
            -- aggregate then resolved differently between runs, flipping one
            -- subject in and out of the `raw_end_date IS NOT NULL` filter about
            -- one run in five. Filtering nulls out is also the correct
            -- semantics -- an expiry we do not know cannot be the spell end.
            MAX_BY(
                membership_expire_date,
                {{'d': transaction_date, 'e': membership_expire_date}}
            ) FILTER (
                WHERE NOT is_cancel AND membership_expire_date IS NOT NULL
            ) AS last_noncancel_expiry,
            MAX(transaction_date) FILTER (WHERE NOT is_cancel) AS last_noncancel_date,
            MAX(transaction_date) FILTER (WHERE is_cancel) AS last_cancel_date,
            -- Plan length bought at the START of the spell. This is the
            -- primary stratification variable, not a covariate: a 7-day promo
            -- and a 410-day annual plan are different products with different
            -- hazard shapes, so a pooled curve over them is a mixture.
            MIN_BY(
                plan_days,
                {{'d': transaction_date,
                  'e': COALESCE(membership_expire_date, DATE '9999-12-31'),
                  'c': is_cancel}}
            ) AS first_plan_days
        FROM first_spell
        GROUP BY msno
    ),
    resolved AS (
        SELECT
            a.*,
            -- A spell ends by cancellation only when the cancel comes after
            -- every non-cancel transaction in it.
            COALESCE(a.last_cancel_date > a.last_noncancel_date, FALSE)
                AS ended_by_cancel,
            CASE
                WHEN COALESCE(a.last_cancel_date > a.last_noncancel_date, FALSE)
                    THEN a.last_cancel_date          -- event time = cancel date
                ELSE a.last_noncancel_expiry         -- spell end = last expiry
            END AS raw_end_date
        FROM agg a
    ),
    labelled AS (
        SELECT
            r.*,
            m.registration_init_time AS registration_date,
            m.registered_via, m.city, m.bd, m.gender,
            -- Churn is only observable once the full grace window has been
            -- seen. A cancellation is its own evidence and needs no grace.
            CASE
                WHEN r.ended_by_cancel
                    THEN r.raw_end_date <= DATE '{LABEL_CUTOFF}'
                ELSE r.raw_end_date + INTERVAL {CHURN_GRACE_DAYS} DAY
                     <= DATE '{LABEL_CUTOFF}'
            END AS event,
            -- Censored subjects stop being observed at LABEL_CUTOFF, never
            -- later: beyond it no outcome is knowable.
            LEAST(r.raw_end_date, DATE '{LABEL_CUTOFF}') AS censor_date
        FROM resolved r
        JOIN staged.members m USING (msno)
    )
    SELECT
        msno,
        STRFTIME(spell_start_date, '%Y-%m') AS cohort_month,
        spell_start_date,
        registration_date,
        -- Delayed entry: tenure already accrued when observation began. Zero
        -- for anyone who registered after the window opened.
        GREATEST(DATE_DIFF('day', registration_date, DATE '{WINDOW_OPEN}'), 0)
            AS entry_days,
        DATE_DIFF(
            'day', registration_date,
            CASE WHEN event THEN raw_end_date ELSE censor_date END
        ) AS tenure_days,
        event,
        CASE WHEN event THEN raw_end_date ELSE censor_date END AS end_date,
        ended_by_cancel,
        n_transactions,
        first_plan_days,
        -- 'unknown' is a real bucket, not a dumping ground: plan_days is NULL
        -- only where the zero-day imputation could not recover a duration.
        CASE
            WHEN first_plan_days IS NULL THEN 'unknown'
            WHEN first_plan_days <= 7   THEN '01_<=7d'
            WHEN first_plan_days <= 31  THEN '02_8-31d'
            WHEN first_plan_days <= 120 THEN '03_32-120d'
            ELSE '04_121d+'
        END AS plan_type,
        registered_via, city, bd, gender
    FROM labelled
    WHERE raw_end_date IS NOT NULL
    """


def build(con: duckdb.DuckDBPyConnection) -> dict:
    """Build the spell table, quarantining anything the guards catch."""
    con.execute(SPELLS_DDL)
    con.execute(SPELLS_QUARANTINE_DDL)
    # Materialised, not a view. The query is read three times below, and a view
    # would re-execute each time; any non-determinism then puts a row in two
    # places at once. Computing it once removes that class of bug entirely
    # rather than relying on the query being perfectly deterministic.
    con.execute(f"CREATE OR REPLACE TABLE candidates AS {build_spell_sql()}")

    # Three guards, checked in this order because the first explains most of
    # what the others would otherwise catch and mislabel.
    #
    # starts_after_label_cutoff: a spell beginning after LABEL_CUTOFF has zero
    #   observable follow-up -- no outcome is knowable for it under the
    #   two-cutoff rule. Censoring such a subject at LABEL_CUTOFF would place
    #   their censor date BEFORE their spell started, which is incoherent, and
    #   for the subset whose registration is recent it also drives tenure
    #   negative. These are outside the observable window, not corrupt.
    # negative_tenure: the ruling's hard guard. The computed end precedes
    #   registration for a reason other than the clamp above -- no repair can
    #   make that meaningful.
    # entry_after_exit: the same class one step later. lifelines requires a
    #   subject to enter the risk set strictly before leaving it, so such a row
    #   would raise inside the fitter or be dropped without comment.
    #
    # None are clamped. Each is quarantined with its reason and reported.
    excluded = f"""
        spell_start_date > DATE '{LABEL_CUTOFF}'
        OR tenure_days < 0
        OR entry_days >= tenure_days
    """
    con.execute(
        f"""
        INSERT INTO spells_quarantine
        SELECT msno, cohort_month, spell_start_date, registration_date,
               entry_days, tenure_days, event, end_date, ended_by_cancel,
               n_transactions,
               CASE
                   WHEN spell_start_date > DATE '{LABEL_CUTOFF}'
                       THEN 'starts_after_label_cutoff'
                   WHEN tenure_days < 0 THEN 'negative_tenure'
                   ELSE 'entry_after_exit'
               END
        FROM candidates
        WHERE {excluded}
        """
    )
    con.execute(
        f"""
        INSERT INTO spells
        SELECT msno, cohort_month, spell_start_date, registration_date,
               entry_days, tenure_days, event, end_date, ended_by_cancel,
               n_transactions, first_plan_days, plan_type,
               registered_via, city, bd, gender
        FROM candidates
        WHERE NOT ({excluded})
        """
    )

    counts = con.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM candidates),
            (SELECT COUNT(*) FROM spells),
            (SELECT COUNT(*) FROM spells_quarantine WHERE quarantine_reason='negative_tenure'),
            (SELECT COUNT(*) FROM spells_quarantine WHERE quarantine_reason='entry_after_exit'),
            (SELECT COUNT(DISTINCT msno) FROM staged.transactions),
            (SELECT COUNT(*) FROM spells_quarantine
             WHERE quarantine_reason='starts_after_label_cutoff')
        """
    ).fetchone()
    # The mart holds deliverables, not scaffolding.
    con.execute("DROP TABLE candidates")

    return {
        "candidates": counts[0],
        "subjects": counts[1],
        "negative_tenure": counts[2],
        "entry_after_exit": counts[3],
        "staged_users": counts[4],
        "starts_after_label_cutoff": counts[5],
    }


def validate(con: duckdb.DuckDBPyConnection, counts: dict) -> None:
    """Run the staged->marts checks. Any FAIL halts before anything is used."""
    spells = con.execute("SELECT msno, event FROM spells").pl()
    unmatched = counts["staged_users"] - counts["candidates"]

    run_checks(
        STAGE,
        [
            lambda: censoring_rate(spells, stage=STAGE),
            lambda: duplicate_transactions(spells, stage=STAGE),
            # One subject per user, so the subject count must account for every
            # staged user exactly once: those with no members row (no
            # registration, so no computable entry time) and those the guards
            # caught.
            lambda: row_count_reconciliation(
                stage=STAGE,
                raw_rows=counts["staged_users"],
                staged_rows=counts["subjects"],
                logged_drops={
                    "no_members_row_or_null_spell_end": unmatched,
                    "starts_after_label_cutoff": counts["starts_after_label_cutoff"],
                    "negative_tenure": counts["negative_tenure"],
                    "entry_after_exit": counts["entry_after_exit"],
                },
            ),
        ],
    )


def report(con: duckdb.DuckDBPyConnection, counts: dict) -> None:
    r = con.execute(
        """
        SELECT COUNT(*), SUM(event::INT), AVG(event::INT),
               MEDIAN(tenure_days), MEDIAN(entry_days),
               SUM(CASE WHEN entry_days > 0 THEN 1 ELSE 0 END),
               SUM(ended_by_cancel::INT)
        FROM spells
        """
    ).fetchone()
    n, events, event_rate, med_ten, med_entry, truncated, cancels = r

    print("\n=== SPELL TABLE ===")
    print(f"  subjects            : {n:,}")
    print(f"  events (churn)      : {events:,}")
    print(f"  censored            : {n - events:,}")
    print(f"  censoring rate      : {1 - event_rate:.2%}")
    print(f"  ended by cancel     : {cancels:,}")
    print(f"  left-truncated      : {truncated:,} ({truncated / n:.2%}) entry_days > 0")
    print(f"  median tenure_days  : {med_ten:,.0f}")
    print(f"  median entry_days   : {med_entry:,.0f}")

    print("\n=== GUARDS ===")
    print(f"  starts_after_label_cutoff   : {counts['starts_after_label_cutoff']:,}")
    print(f"  negative_tenure quarantined : {counts['negative_tenure']:,}")
    print(f"  entry_after_exit quarantined: {counts['entry_after_exit']:,}")
    print(f"  no members row / null end   : {counts['staged_users'] - counts['candidates']:,}")

    print("\n=== PLAN TYPE (primary stratum) ===")
    print(f"  {'stratum':<12} {'subjects':>10} {'events':>10} {'censoring':>10}")
    for row in con.execute(
        """
        SELECT plan_type, COUNT(*), SUM(event::INT), 1 - AVG(event::INT)
        FROM spells GROUP BY 1 ORDER BY 1
        """
    ).fetchall():
        print(f"  {row[0]:<12} {row[1]:>10,} {row[2]:>10,} {row[3]:>9.1%}")

    print("\n=== COHORTS ===")
    print(f"  {'cohort':<9} {'subjects':>10} {'events':>10} {'censoring':>10}")
    for row in con.execute(
        """
        SELECT cohort_month, COUNT(*), SUM(event::INT), 1 - AVG(event::INT)
        FROM spells GROUP BY 1 ORDER BY 1
        """
    ).fetchall():
        print(f"  {row[0]:<9} {row[1]:>10,} {row[2]:>10,} {row[3]:>9.1%}")


def main() -> None:
    MARTS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(MARTS_DB))
    con.execute(f"ATTACH '{DB}' AS staged (READ_ONLY)")

    print(f"Building spell table in {MARTS_DB}")
    counts = build(con)
    validate(con, counts)
    report(con, counts)
    con.close()
    print("\nValidation passed. Spell table built.")


if __name__ == "__main__":
    main()
