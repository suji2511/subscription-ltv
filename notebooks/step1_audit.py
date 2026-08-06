"""Step-1 exploratory audit of the raw KKBox CSVs.

EXPLORATION ONLY. Per CLAUDE.md rule 5, nothing in notebooks/ is a deliverable.
This script exists to answer the five questions in docs/step1_audit.md so the
open decisions can be settled with evidence rather than convention. Anything
here that turns out to matter gets promoted into src/ with a test.

Why DuckDB rather than polars here: the full transactions.csv is ~21.5M rows.
DuckDB scans the CSV out of core, so nothing is ever materialised in memory,
and the questions are all naturally expressed as aggregations and window
functions that read more clearly in SQL than in chained dataframe calls.
polars remains the ingestion tool in src/ingest/ per the stack decision; this
is a read-only look at raw files, not the pipeline.

Nothing here writes to data/raw/. Run:  python notebooks/step1_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CHURN_GRACE_DAYS, RAW, ROOT

AUDIT_DOC = ROOT / "docs" / "step1_audit.md"

# The competition ships two overlapping pairs of files and they are NOT
# interchangeable:
#
#   transactions.csv    ~21.5M rows, 2015-01 -> 2017-02, real longitudinal
#                       history. This is the one the project needs.
#   transactions_v2.csv ~1.4M rows, the March-2017 "refresh" slice. 88% of its
#                       users appear exactly once and the median observed span
#                       is 0 days, so it carries no lifetimes to model.
#
# Prefer the full files when present and fall back to the refresh, rather than
# hardcoding either. Whichever is used gets printed and written into the audit
# doc, so a set of numbers can never be misread as coming from the other file.
TRANSACTION_CANDIDATES = ["transactions.csv", "transactions_v2.csv"]
MEMBER_CANDIDATES = ["members.csv", "members_v3.csv"]

TRANSACTIONS_CSV = RAW / TRANSACTION_CANDIDATES[-1]
MEMBERS_CSV = RAW / MEMBER_CANDIDATES[-1]

# The columns this audit depends on. If the file does not have exactly these,
# stop rather than silently analysing the wrong thing.
EXPECTED_TRANSACTION_COLUMNS = [
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
]
EXPECTED_MEMBER_COLUMNS = [
    "msno",
    "city",
    "bd",
    "gender",
    "registered_via",
    "registration_init_time",
]


# --- setup ----------------------------------------------------------------


def resolve_inputs() -> None:
    """Pick the best available transactions/members pair and say which it is.

    Fails loudly if neither candidate is present, and warns loudly if only the
    refresh slice is available, because every downstream number would then be
    describing a snapshot rather than a subscription history.
    """
    global TRANSACTIONS_CSV, MEMBERS_CSV
    TRANSACTIONS_CSV = _first_present(TRANSACTION_CANDIDATES)
    MEMBERS_CSV = _first_present(MEMBER_CANDIDATES)

    print(f"transactions : {TRANSACTIONS_CSV.name}")
    print(f"members      : {MEMBERS_CSV.name}")

    if TRANSACTIONS_CSV.name == "transactions_v2.csv":
        print(
            "\n  WARNING: using the March-2017 refresh slice, not the full history.\n"
            "           Q2 (cohorts), Q3 (censoring) and Q5 (resubscribers) will be\n"
            "           degenerate by construction. Download transactions.csv from\n"
            "           the original competition files before trusting these."
        )


def _first_present(candidates: list[str]) -> Path:
    for name in candidates:
        path = RAW / name
        if path.exists():
            return path
    tried = ", ".join(candidates)
    raise SystemExit(f"No raw input found in {RAW}. Tried: {tried}")


def build_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create typed views over the raw CSVs.

    KKBox stores dates as YYYYMMDD integers. We convert once here so every
    query below works in real dates and no arithmetic is ever done on the
    integer form (20170101 - 20161231 = 8870, which is nonsense).

    Assumption: every date parses. If any row has an unparseable date this
    raises rather than yielding NULL, which is what we want at audit time --
    a silent NULL here would understate the observation window.
    """
    con.execute(
        f"""
        CREATE VIEW tx_raw AS
        SELECT * FROM read_csv_auto('{TRANSACTIONS_CSV}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE VIEW members_raw AS
        SELECT * FROM read_csv_auto('{MEMBERS_CSV}', header=true)
        """
    )
    assert_columns(con, "tx_raw", EXPECTED_TRANSACTION_COLUMNS)
    assert_columns(con, "members_raw", EXPECTED_MEMBER_COLUMNS)

    con.execute(
        """
        CREATE VIEW tx AS
        SELECT
            msno,
            payment_method_id,
            payment_plan_days,
            plan_list_price,
            actual_amount_paid,
            is_auto_renew,
            is_cancel,
            strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE
                AS transaction_date,
            strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE
                AS membership_expire_date
        FROM tx_raw
        """
    )
    con.execute(
        """
        CREATE VIEW members AS
        SELECT
            msno,
            city,
            bd,
            gender,
            registered_via,
            strptime(CAST(registration_init_time AS VARCHAR), '%Y%m%d')::DATE
                AS registration_init_time
        FROM members_raw
        """
    )


def assert_columns(con: duckdb.DuckDBPyConnection, view: str, expected: list[str]) -> None:
    """Stop if the file's columns are not the ones this audit was written for."""
    found = [row[0] for row in con.execute(f"DESCRIBE {view}").fetchall()]
    if found != expected:
        raise SystemExit(
            f"Unexpected columns in {view}.\n"
            f"  expected: {expected}\n"
            f"  found:    {found}"
        )


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --- Q1: observation window -----------------------------------------------


def q1_observation_window(con: duckdb.DuckDBPyConnection) -> dict:
    """Min/max of both date columns, plus the row and user counts they cover.

    The two windows are reported separately on purpose. Transaction dates say
    when we observed activity; expiry dates extend past the end of the file and
    are what determine who is still active at the cutoff. If max(expiry) is far
    beyond max(transaction_date), the file is a snapshot slice rather than a
    full history, which changes what a "signup cohort" means in Q2.
    """
    row = con.execute(
        """
        SELECT
            MIN(transaction_date)      AS min_tx,
            MAX(transaction_date)      AS max_tx,
            MIN(membership_expire_date) AS min_expiry,
            MAX(membership_expire_date) AS max_expiry,
            COUNT(*)                   AS n_rows,
            COUNT(DISTINCT msno)       AS n_users
        FROM tx
        """
    ).fetchone()
    out = dict(
        zip(
            ["min_tx", "max_tx", "min_expiry", "max_expiry", "n_rows", "n_users"],
            row,
        )
    )

    reg = con.execute(
        """
        SELECT MIN(registration_init_time), MAX(registration_init_time), COUNT(*)
        FROM members
        """
    ).fetchone()
    out["min_registration"], out["max_registration"], out["n_members"] = reg

    rule("Q1  OBSERVATION WINDOW")
    print(f"  transactions rows      : {out['n_rows']:,}")
    print(f"  distinct users in tx   : {out['n_users']:,}")
    print(f"  members rows           : {out['n_members']:,}")
    print(f"  transaction_date       : {out['min_tx']}  ->  {out['max_tx']}")
    print(f"  membership_expire_date : {out['min_expiry']}  ->  {out['max_expiry']}")
    print(f"  registration_init_time : {out['min_registration']}  ->  {out['max_registration']}")

    tx_span_days = (out["max_tx"] - out["min_tx"]).days
    print(f"\n  transaction window spans {tx_span_days} days")
    if tx_span_days < 400:
        print(
            "  NOTE: window is under ~13 months. 'First transaction in this file'\n"
            "        is then a left-truncated view of signup, not a true signup\n"
            "        date. See Q2b before treating it as a cohort."
        )
    return out


# --- Q2: monthly signup cohorts -------------------------------------------


def q2_signup_cohorts(con: duckdb.DuckDBPyConnection) -> dict:
    """Cohort sizes by month of each user's first transaction in this file.

    Assumption this makes: a user's earliest transaction in transactions_v2 is
    their signup. What could violate it: if transactions_v2 is a windowed slice
    of a longer history, then long-tenured users first appear in the slice's
    opening month and pile up into an artificial mega-cohort. Q2b tests exactly
    that by comparing against registration_init_time from members_v3.
    """
    con.execute(
        """
        CREATE VIEW first_tx AS
        SELECT msno, MIN(transaction_date) AS first_transaction_date
        FROM tx
        GROUP BY msno
        """
    )
    cohorts = con.execute(
        """
        SELECT
            strftime(first_transaction_date, '%Y-%m') AS cohort_month,
            COUNT(*) AS n_users
        FROM first_tx
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()

    smallest = min(cohorts, key=lambda r: r[1])
    largest = max(cohorts, key=lambda r: r[1])
    total = sum(r[1] for r in cohorts)

    rule("Q2  MONTHLY SIGNUP COHORTS (by first transaction in file)")
    print(f"  {'cohort':<10} {'users':>12}   share")
    for month, n in cohorts:
        flag = ""
        if (month, n) == smallest:
            flag = "  <-- SMALLEST"
        if (month, n) == largest:
            flag = "  <-- largest"
        print(f"  {month:<10} {n:>12,}   {n / total:6.2%}{flag}")
    print(f"  {'TOTAL':<10} {total:>12,}")
    print(f"\n  number of monthly cohorts : {len(cohorts)}")
    print(f"  smallest cohort           : {smallest[0]} with {smallest[1]:,} users")
    print(f"  largest cohort            : {largest[0]} with {largest[1]:,} users")

    q2b = _q2b_registration_cross_check(con)

    return {
        "cohorts": cohorts,
        "n_cohorts": len(cohorts),
        "smallest_month": smallest[0],
        "smallest_n": smallest[1],
        "largest_month": largest[0],
        "largest_n": largest[1],
        "total_users": total,
        **q2b,
    }


def _q2b_registration_cross_check(con: duckdb.DuckDBPyConnection) -> dict:
    """Compare first-transaction month against registration_init_time.

    If most users registered long before their first transaction in this file,
    the file is a slice and 'first transaction' is not signup. That single fact
    decides whether cohorts can be built from transactions alone or must be
    anchored on registration_init_time.
    """
    row = con.execute(
        """
        SELECT
            COUNT(*) AS n_matched,
            SUM(CASE WHEN m.registration_init_time < f.first_transaction_date
                     THEN 1 ELSE 0 END) AS n_registered_earlier,
            MEDIAN(DATE_DIFF('day', m.registration_init_time,
                             f.first_transaction_date)) AS median_gap_days
        FROM first_tx f
        JOIN members m USING (msno)
        """
    ).fetchone()
    n_matched, n_earlier, median_gap = row
    n_earlier = n_earlier or 0

    unmatched = con.execute(
        "SELECT COUNT(*) FROM first_tx f LEFT JOIN members m USING (msno) "
        "WHERE m.msno IS NULL"
    ).fetchone()[0]

    print("\n  Q2b cross-check against members_v3.registration_init_time")
    print(f"    users matched to a member row      : {n_matched:,}")
    print(f"    users with NO member row           : {unmatched:,}")
    if n_matched:
        print(
            f"    registered before first tx in file : {n_earlier:,} "
            f"({n_earlier / n_matched:.2%})"
        )
        print(f"    median days registration -> first tx: {median_gap}")
        if n_earlier / n_matched > 0.5:
            print(
                "    NOTE: most users predate their first transaction here.\n"
                "          Treat 'first transaction' cohorts as a slice artefact\n"
                "          and consider anchoring cohorts on registration instead."
            )
    return {
        "n_matched_members": n_matched,
        "n_unmatched_members": unmatched,
        "n_registered_earlier": n_earlier,
        "median_reg_to_tx_days": median_gap,
    }


# --- Q3: censoring rate ---------------------------------------------------


def q3_censoring_rate(con: duckdb.DuckDBPyConnection) -> dict:
    """Share of users still active at the last date in the data.

    One row per user: their latest transaction and the expiry it set. Users are
    split three ways rather than two, because a binary active/churned split
    would quietly misclassify the third group:

      active       expiry  >  max date            -> right-censored
      undetermined expiry <= max date, but within CHURN_GRACE_DAYS of it
                                                  -> not enough follow-up to
                                                     tell yet; also censored,
                                                     but for a different reason
      churned      expiry + grace < max date, no later transaction

    Assumption: the last transaction's expiry is the user's current state. What
    could violate it: is_cancel rows, which can set an expiry earlier than a
    prior row's. Counted separately below so the effect is visible.
    """
    con.execute(
        """
        CREATE VIEW last_tx AS
        SELECT msno, transaction_date, membership_expire_date, is_cancel
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY msno
                       ORDER BY transaction_date DESC, membership_expire_date DESC
                   ) AS rn
            FROM tx
        )
        WHERE rn = 1
        """
    )
    max_date = con.execute("SELECT MAX(transaction_date) FROM tx").fetchone()[0]

    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_users,
            SUM(CASE WHEN membership_expire_date > DATE '{max_date}'
                     THEN 1 ELSE 0 END) AS n_active,
            SUM(CASE WHEN membership_expire_date <= DATE '{max_date}'
                      AND membership_expire_date
                          > DATE '{max_date}' - INTERVAL {CHURN_GRACE_DAYS} DAY
                     THEN 1 ELSE 0 END) AS n_undetermined,
            SUM(CASE WHEN membership_expire_date
                          <= DATE '{max_date}' - INTERVAL {CHURN_GRACE_DAYS} DAY
                     THEN 1 ELSE 0 END) AS n_churned,
            SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END) AS n_last_is_cancel
        FROM last_tx
        """
    ).fetchone()
    keys = ["n_users", "n_active", "n_undetermined", "n_churned", "n_last_is_cancel"]
    out = dict(zip(keys, row))
    out["max_date"] = max_date
    n = out["n_users"]

    out["censoring_rate"] = (out["n_active"] + out["n_undetermined"]) / n if n else 0.0

    rule(f"Q3  CENSORING AT LAST DATE IN DATA ({max_date}, grace={CHURN_GRACE_DAYS}d)")
    print(f"  users                                : {n:,}")
    print(f"  active   (expiry > cutoff)           : {out['n_active']:,} ({out['n_active'] / n:.2%})")
    print(
        f"  undetermined (expired within grace)  : {out['n_undetermined']:,} "
        f"({out['n_undetermined'] / n:.2%})"
    )
    print(f"  churned  (expired + grace elapsed)   : {out['n_churned']:,} ({out['n_churned'] / n:.2%})")
    print(f"\n  CENSORING RATE (active+undetermined) : {out['censoring_rate']:.2%}")
    print(f"  (last transaction was is_cancel=1     : {out['n_last_is_cancel']:,})")

    if out["censoring_rate"] > 0.9:
        print(
            "\n  NOTE: censoring above 90%. Kaplan-Meier will be estimable but the\n"
            "        tail will rest on very few observed events. Check that the\n"
            "        backtest horizon stays inside the range with real events."
        )
    return out


# --- Q4: price reconciliation ---------------------------------------------


def q4_price_reconciliation(con: duckdb.DuckDBPyConnection) -> dict:
    """Does actual_amount_paid line up with plan_list_price given plan days?

    The check is deliberately simple: equality of list price and amount paid.
    Deviations are then split by direction, because they mean different things:
    underpayment suggests discounts/promotions, overpayment suggests the two
    fields are not on the same basis at all. Zero-price rows are counted
    separately since a free plan is not a mismatch, it is a different product.

    Rejected alternative: reconciling against a derived daily rate
    (list_price / payment_plan_days) and allowing a tolerance. That bakes in an
    assumption that price is linear in duration, which is exactly what the plan
    grid printed below is meant to test rather than assume.
    """
    row = con.execute(
        """
        SELECT
            COUNT(*) AS n_rows,
            SUM(CASE WHEN actual_amount_paid = plan_list_price THEN 1 ELSE 0 END) AS n_match,
            SUM(CASE WHEN actual_amount_paid < plan_list_price THEN 1 ELSE 0 END) AS n_under,
            SUM(CASE WHEN actual_amount_paid > plan_list_price THEN 1 ELSE 0 END) AS n_over,
            SUM(CASE WHEN plan_list_price = 0 AND actual_amount_paid = 0 THEN 1 ELSE 0 END) AS n_free,
            SUM(CASE WHEN plan_list_price < 0 OR actual_amount_paid < 0 THEN 1 ELSE 0 END) AS n_negative,
            SUM(CASE WHEN payment_plan_days <= 0 THEN 1 ELSE 0 END) AS n_nonpositive_days
        FROM tx
        """
    ).fetchone()
    keys = [
        "n_rows", "n_match", "n_under", "n_over",
        "n_free", "n_negative", "n_nonpositive_days",
    ]
    out = dict(zip(keys, row))
    n = out["n_rows"]
    out["mismatch_rate"] = (out["n_under"] + out["n_over"]) / n if n else 0.0

    rule("Q4  PRICE RECONCILIATION")
    print(f"  rows                          : {n:,}")
    print(f"  paid == list                  : {out['n_match']:,} ({out['n_match'] / n:.2%})")
    print(f"  paid <  list (underpaid)      : {out['n_under']:,} ({out['n_under'] / n:.2%})")
    print(f"  paid >  list (overpaid)       : {out['n_over']:,} ({out['n_over'] / n:.2%})")
    print(f"\n  MISMATCH RATE                 : {out['mismatch_rate']:.2%}")
    print(f"  zero-price rows (list=paid=0) : {out['n_free']:,} ({out['n_free'] / n:.2%})")
    print(f"  negative amounts              : {out['n_negative']:,}")
    print(f"  payment_plan_days <= 0        : {out['n_nonpositive_days']:,}")

    print("\n  Most common (plan_days, list_price) combinations:")
    grid = con.execute(
        """
        SELECT payment_plan_days, plan_list_price, COUNT(*) AS n,
               ROUND(plan_list_price / NULLIF(payment_plan_days, 0), 3) AS price_per_day
        FROM tx
        GROUP BY 1, 2
        ORDER BY n DESC
        LIMIT 12
        """
    ).fetchall()
    print(f"    {'days':>6} {'list':>8} {'rows':>12} {'price/day':>11}")
    for days, price, cnt, ppd in grid:
        print(f"    {days:>6} {price:>8} {cnt:>12,} {ppd if ppd is not None else '-':>11}")

    print("\n  Sample of mismatches:")
    sample = con.execute(
        """
        SELECT payment_method_id, payment_plan_days, plan_list_price,
               actual_amount_paid, is_cancel, transaction_date
        FROM tx
        WHERE actual_amount_paid <> plan_list_price
        LIMIT 15
        """
    ).fetchall()
    print(f"    {'method':>7} {'days':>6} {'list':>8} {'paid':>8} {'cancel':>7}  date")
    for method, days, price, paid, cancel, date in sample:
        print(f"    {method:>7} {days:>6} {price:>8} {paid:>8} {cancel:>7}  {date}")
    out["sample_mismatches"] = sample
    out["plan_grid"] = grid
    return out


# --- Q5: resubscriber prevalence ------------------------------------------


def q5_resubscribers(con: duckdb.DuckDBPyConnection) -> dict:
    """Users who let a membership lapse past the grace window, then came back.

    Definition used: order a user's transactions by date; a return is any
    transaction whose date is more than CHURN_GRACE_DAYS after the expiry set by
    their previous transaction. That is deliberately the same 30-day rule the
    dataset uses to define churn, so 'resubscriber' means 'churned by the
    project's own definition, then transacted again'.

    Assumption: transactions are a clean chronological sequence per user. What
    could violate it: same-day rows (a renewal and a cancellation on one date)
    where the ordering is arbitrary. Tie-break is on expiry date to make the
    result deterministic, and same-day pairs are counted below so the exposure
    to that choice is visible.
    """
    con.execute(
        f"""
        CREATE VIEW gaps AS
        SELECT
            msno,
            transaction_date,
            prev_expiry,
            DATE_DIFF('day', prev_expiry, transaction_date) AS gap_days
        FROM (
            SELECT
                msno,
                transaction_date,
                LAG(membership_expire_date) OVER (
                    PARTITION BY msno
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expiry
            FROM tx
        )
        WHERE prev_expiry IS NOT NULL
          AND DATE_DIFF('day', prev_expiry, transaction_date) > {CHURN_GRACE_DAYS}
        """
    )
    total_users = con.execute("SELECT COUNT(DISTINCT msno) FROM tx").fetchone()[0]
    multi_users = con.execute(
        "SELECT COUNT(*) FROM (SELECT msno FROM tx GROUP BY msno HAVING COUNT(*) > 1)"
    ).fetchone()[0]

    row = con.execute(
        """
        SELECT COUNT(DISTINCT msno) AS n_resub, COUNT(*) AS n_return_events,
               MEDIAN(gap_days) AS median_gap, MAX(gap_days) AS max_gap
        FROM gaps
        """
    ).fetchone()
    n_resub, n_events, median_gap, max_gap = row

    same_day = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT msno, transaction_date
            FROM tx GROUP BY 1, 2 HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    rule(f"Q5  RESUBSCRIBER PREVALENCE (gap > {CHURN_GRACE_DAYS}d after expiry)")
    print(f"  distinct users                  : {total_users:,}")
    print(f"  users with >1 transaction       : {multi_users:,} ({multi_users / total_users:.2%})")
    print(f"  users with >=1 return after gap : {n_resub:,} ({n_resub / total_users:.2%} of all users)")
    if multi_users:
        print(f"                                    ({n_resub / multi_users:.2%} of multi-tx users)")
    print(f"  total return events             : {n_events:,}")
    print(f"  median gap (days)               : {median_gap}")
    print(f"  max gap (days)                  : {max_gap}")
    print(f"\n  (user-days with >1 transaction  : {same_day:,} -- ordering tie-breaks)")

    print("\n  Gap length distribution:")
    buckets = con.execute(
        f"""
        SELECT
            CASE
                WHEN gap_days <= 60  THEN '{CHURN_GRACE_DAYS}-60d'
                WHEN gap_days <= 90  THEN '61-90d'
                WHEN gap_days <= 180 THEN '91-180d'
                WHEN gap_days <= 365 THEN '181-365d'
                ELSE '365d+'
            END AS bucket,
            COUNT(*) AS n
        FROM gaps
        GROUP BY 1
        ORDER BY MIN(gap_days)
        """
    ).fetchall()
    for bucket, cnt in buckets:
        print(f"    {bucket:<12} {cnt:>12,}")

    return {
        "total_users": total_users,
        "multi_users": multi_users,
        "n_resubscribers": n_resub,
        "resub_share_all": n_resub / total_users if total_users else 0.0,
        "resub_share_multi": n_resub / multi_users if multi_users else 0.0,
        "n_return_events": n_events,
        "median_gap_days": median_gap,
        "max_gap_days": max_gap,
        "same_day_user_days": same_day,
    }


# --- Q6: longitudinal coverage --------------------------------------------


def q6_longitudinal_coverage(con: duckdb.DuckDBPyConnection) -> dict:
    """Does this file contain observed lifetimes, or just a snapshot?

    Added after Q1-Q5 came back with 99.95% censoring and a single resubscriber.
    Those two results only make sense if the file is not a longitudinal history,
    so this quantifies that directly rather than leaving it as an inference.

    A survival model needs subjects observed over time. The diagnostic is the
    per-user span from first to last transaction: if the median span is near
    zero, most users are seen once and there is no lifetime to model, no matter
    how many rows the file has.
    """
    rule("Q6  LONGITUDINAL COVERAGE (is this a history or a snapshot?)")

    by_month = con.execute(
        """
        SELECT strftime(transaction_date, '%Y-%m') AS month, COUNT(*) AS n
        FROM tx GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    total_rows = sum(n for _, n in by_month)
    last_month, last_n = by_month[-1]
    final_two = sum(n for _, n in by_month[-2:])

    print("  Transactions by month (all rows, not just first-per-user):")
    for month, n in by_month:
        bar = "#" * max(1, round(60 * n / last_n))
        print(f"    {month}  {n:>10,}  {bar}")
    print(f"\n  share of all rows in {last_month}          : {last_n / total_rows:.2%}")
    print(f"  share of all rows in final two months : {final_two / total_rows:.2%}")

    counts = con.execute(
        """
        SELECT
            CASE WHEN c = 1 THEN '1' WHEN c = 2 THEN '2' WHEN c <= 5 THEN '3-5'
                 WHEN c <= 10 THEN '6-10' ELSE '11+' END AS bucket,
            COUNT(*) AS n_users, MIN(c) AS ord
        FROM (SELECT msno, COUNT(*) AS c FROM tx GROUP BY 1)
        GROUP BY 1 ORDER BY ord
        """
    ).fetchall()
    n_users = sum(r[1] for r in counts)
    single = next((r[1] for r in counts if r[0] == "1"), 0)

    print("\n  Transactions per user:")
    for bucket, n, _ in counts:
        print(f"    {bucket:>6} tx : {n:>10,}  ({n / n_users:6.2%})")
    print(f"\n  users seen exactly once : {single:,} ({single / n_users:.2%})")

    span = con.execute(
        """
        SELECT MEDIAN(span), AVG(span), MAX(span) FROM
        (SELECT msno, DATE_DIFF('day', MIN(transaction_date),
                                MAX(transaction_date)) AS span
         FROM tx GROUP BY 1)
        """
    ).fetchone()
    median_span, mean_span, max_span = span
    print("\n  Observed span per user, first -> last transaction (days):")
    print(f"    median {median_span} | mean {mean_span:.2f} | max {max_span}")

    sanity = con.execute(
        """
        SELECT
            SUM(CASE WHEN membership_expire_date < transaction_date THEN 1 ELSE 0 END),
            SUM(CASE WHEN membership_expire_date = transaction_date THEN 1 ELSE 0 END),
            SUM(CASE WHEN membership_expire_date > DATE '2018-01-01' THEN 1 ELSE 0 END)
        FROM tx
        """
    ).fetchone()
    n_expiry_before_tx, n_expiry_eq_tx, n_expiry_far_future = sanity
    print("\n  Date sanity (these are what date_sanity must catch at raw->staged):")
    print(f"    expiry BEFORE transaction date : {n_expiry_before_tx:,}")
    print(f"    expiry EQUAL to transaction    : {n_expiry_eq_tx:,}")
    print(f"    expiry after 2018-01-01        : {n_expiry_far_future:,}")

    return {
        "by_month": by_month,
        "total_rows": total_rows,
        "last_month": last_month,
        "share_last_month": last_n / total_rows,
        "share_final_two": final_two / total_rows,
        "counts": counts,
        "n_users": n_users,
        "single_tx_users": single,
        "single_tx_share": single / n_users,
        "median_span": median_span,
        "mean_span": mean_span,
        "max_span": max_span,
        "n_expiry_before_tx": n_expiry_before_tx,
        "n_expiry_eq_tx": n_expiry_eq_tx,
        "n_expiry_far_future": n_expiry_far_future,
    }


# --- write-up -------------------------------------------------------------


def write_audit_doc(q1: dict, q2: dict, q3: dict, q4: dict, q5: dict, q6: dict) -> None:
    """Write the numbers into docs/step1_audit.md, leaving the verdict blank.

    The verdict is the user's call, not the script's. This function must never
    fill that line in.
    """
    cohort_lines = "\n".join(
        f"| {month} | {n:,} | {n / q2['total_users']:.2%} |" for month, n in q2["cohorts"]
    )
    mismatch_lines = "\n".join(
        f"| {m} | {d} | {p} | {paid} | {c} | {dt} |"
        for m, d, p, paid, c, dt in q4["sample_mismatches"]
    )
    month_lines = "\n".join(f"| {month} | {n:,} |" for month, n in q6["by_month"])

    doc = f"""# Step 1: data audit

Answer all five before writing pipeline code. Paste the actual numbers.

Generated by `notebooks/step1_audit.py` from the raw CSVs. Every number below
is reproducible by re-running that script.

Source files: `{TRANSACTIONS_CSV.name}` + `{MEMBERS_CSV.name}`

1. Observation window (min / max transaction date):
   **{q1['min_tx']} -> {q1['max_tx']}** ({(q1['max_tx'] - q1['min_tx']).days} days)
   - membership_expire_date: {q1['min_expiry']} -> {q1['max_expiry']}
   - registration_init_time (members_v3): {q1['min_registration']} -> {q1['max_registration']}
   - {q1['n_rows']:,} transaction rows, {q1['n_users']:,} distinct users, {q1['n_members']:,} member rows

2. Number of monthly signup cohorts, and size of the smallest:
   **{q2['n_cohorts']} cohorts; smallest is {q2['smallest_month']} with {q2['smallest_n']:,} users**
   - largest: {q2['largest_month']} with {q2['largest_n']:,} users
   - total users assigned to a cohort: {q2['total_users']:,}

| cohort month | users | share |
|---|---:|---:|
{cohort_lines}

   Cross-check against registration_init_time:
   - matched to a member row: {q2['n_matched_members']:,}; unmatched: {q2['n_unmatched_members']:,}
   - registered before their first transaction in this file: {q2['n_registered_earlier']:,}
     ({q2['n_registered_earlier'] / q2['n_matched_members']:.2%} of matched)
   - median days from registration to first transaction: {q2['median_reg_to_tx_days']}

3. Censoring rate — share of subscriptions still active at cutoff:
   **{q3['censoring_rate']:.2%}** at cutoff {q3['max_date']} (grace {CHURN_GRACE_DAYS}d)
   - active (expiry after cutoff): {q3['n_active']:,} ({q3['n_active'] / q3['n_users']:.2%})
   - undetermined (expired inside grace window): {q3['n_undetermined']:,} ({q3['n_undetermined'] / q3['n_users']:.2%})
   - churned (expired, grace elapsed): {q3['n_churned']:,} ({q3['n_churned'] / q3['n_users']:.2%})
   - last transaction was a cancellation: {q3['n_last_is_cancel']:,}

4. Does plan price x plan_days reconcile with amount paid? Mismatch rate:
   **{q4['mismatch_rate']:.2%}** ({q4['n_under'] + q4['n_over']:,} of {q4['n_rows']:,} rows)
   - paid == list: {q4['n_match']:,} ({q4['n_match'] / q4['n_rows']:.2%})
   - underpaid: {q4['n_under']:,} | overpaid: {q4['n_over']:,}
   - zero-price rows: {q4['n_free']:,} | negative amounts: {q4['n_negative']:,}
   - payment_plan_days <= 0: {q4['n_nonpositive_days']:,}

   Sample mismatches:

| payment_method_id | plan_days | list_price | paid | is_cancel | transaction_date |
|---|---:|---:|---:|---:|---|
{mismatch_lines}

5. Resubscriber prevalence — users with a gap then a return:
   **{q5['resub_share_all']:.2%} of all users** ({q5['n_resubscribers']:,} of {q5['total_users']:,})
   - as a share of users with more than one transaction: {q5['resub_share_multi']:.2%}
     ({q5['multi_users']:,} users have >1 transaction)
   - total return events: {q5['n_return_events']:,}
   - median gap: {q5['median_gap_days']} days | max gap: {q5['max_gap_days']} days
   - user-days with more than one transaction (ordering tie-breaks): {q5['same_day_user_days']:,}

6. Longitudinal coverage (added after Q3/Q5 came back degenerate):
   **{q6['share_last_month']:.2%} of all transaction rows fall in {q6['last_month']} alone**
   ({q6['share_final_two']:.2%} in the final two months)
   - users seen exactly once: {q6['single_tx_users']:,} ({q6['single_tx_share']:.2%})
   - observed span per user, first -> last transaction:
     median **{q6['median_span']} days**, mean {q6['mean_span']:.2f}, max {q6['max_span']}
   - date sanity, for the raw->staged `date_sanity` check to catch:
     expiry before transaction date: {q6['n_expiry_before_tx']:,};
     expiry equal to transaction date: {q6['n_expiry_eq_tx']:,};
     expiry after 2018-01-01: {q6['n_expiry_far_future']:,}

| month | transactions |
|---|---:|
{month_lines}

**Verdict:** proceed with KKBox / switch to Online Retail II
"""
    AUDIT_DOC.write_text(doc)
    print(f"\nWrote {AUDIT_DOC}")


def main() -> None:
    resolve_inputs()
    con = duckdb.connect()
    build_views(con)
    q1 = q1_observation_window(con)
    q2 = q2_signup_cohorts(con)
    q3 = q3_censoring_rate(con)
    q4 = q4_price_reconciliation(con)
    q5 = q5_resubscribers(con)
    q6 = q6_longitudinal_coverage(con)
    write_audit_doc(q1, q2, q3, q4, q5, q6)
    print("\nVerdict line left blank deliberately -- that is your call.")


if __name__ == "__main__":
    main()
