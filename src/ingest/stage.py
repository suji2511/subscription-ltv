"""Raw -> staged ingestion: chunked CSV reads into DuckDB with an enforced schema.

Reads `transactions.csv` (21.5M rows) and `members_v3.csv` in CHUNK_ROWS-sized
batches via polars, applies the two transforms recorded in docs/decisions.md,
and writes typed tables into data/staged/subscriptions.duckdb.

`user_logs` is skipped entirely, per CLAUDE.md.

Two transforms happen here, and both are deliberate repairs rather than checks —
a check observes and halts, it never fixes:

  1. The 1970-01-01 epoch sentinel in membership_expire_date is nulled. It is a
     missing date encoded as a number, and left in place it poisons every min
     and max downstream.
  2. Rows with payment_plan_days = 0 have plan_days recovered from
     (expiry - transaction) and list_price from actual_amount_paid, carrying a
     plan_days_imputed flag so every revenue figure can be recomputed with them
     excluded.

Rows that violate date_sanity's non-cancellation rule are quarantined into a
separate table rather than dropped, so no row is ever lost and
row_count_reconciliation balances exactly.

Run:  python -m src.ingest.stage
"""

from __future__ import annotations

import csv
import time

import duckdb
import polars as pl

from src.config import (
    CHUNK_ROWS,
    DB,
    EPOCH_SENTINEL,
    RAW_MEMBERS,
    RAW_TRANSACTIONS,
)
from src.ingest.schema import (
    MEMBERS_DDL,
    MEMBERS_SCHEMA,
    TRANSACTIONS_DDL,
    TRANSACTIONS_QUARANTINE_DDL,
    TRANSACTIONS_SCHEMA,
    to_date,
)
from src.validate.checks import date_sanity, row_count_reconciliation, run_checks

STAGE = "raw->staged"

TRANSACTION_COLUMNS = [
    "msno",
    "payment_method_id",
    "plan_days",
    "list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
    "plan_days_imputed",
]

MEMBER_COLUMNS = [
    "msno",
    "city",
    "bd",
    "gender",
    "registered_via",
    "registration_init_time",
]

QUARANTINE_REASON = "expiry_before_transaction_not_cancelled"


def assert_header(path, expected: list[str]) -> None:
    """Stop if the CSV's columns are not exactly what the schema declares.

    polars' schema_overrides is keyed by name, so a renamed or reordered column
    would silently go un-typed rather than raising. This closes that gap.
    """
    with path.open() as fh:
        header = next(csv.reader(fh))
    if header != expected:
        raise SystemExit(
            f"Unexpected columns in {path.name}.\n"
            f"  expected: {expected}\n"
            f"  found:    {header}"
        )


def transform_transactions(chunk: pl.DataFrame) -> pl.DataFrame:
    """Type the dates, null the sentinel, impute the zero-day rows."""
    df = chunk.with_columns(
        to_date("transaction_date").alias("transaction_date"),
        to_date("membership_expire_date").alias("membership_expire_date"),
    )

    df = df.with_columns(
        pl.when(pl.col("membership_expire_date") == EPOCH_SENTINEL)
        .then(None)
        .otherwise(pl.col("membership_expire_date"))
        .alias("membership_expire_date")
    )

    # The zero-day population: plan descriptors unpopulated by the source. The
    # duration is recoverable from the dates only when the row actually extends
    # membership; where it does not, plan_days stays NULL rather than being
    # invented as zero or negative.
    imputed = pl.col("payment_plan_days") == 0
    recoverable = pl.col("membership_expire_date") > pl.col("transaction_date")

    return df.with_columns(
        plan_days_imputed=imputed,
        plan_days=pl.when(imputed)
        .then(
            pl.when(recoverable)
            .then(
                (pl.col("membership_expire_date") - pl.col("transaction_date"))
                .dt.total_days()
                .cast(pl.Int32)
            )
            .otherwise(None)
        )
        .otherwise(pl.col("payment_plan_days")),
        list_price=pl.when(imputed)
        .then(pl.col("actual_amount_paid"))
        .otherwise(pl.col("plan_list_price")),
        is_auto_renew=pl.col("is_auto_renew").cast(pl.Boolean),
        is_cancel=pl.col("is_cancel").cast(pl.Boolean),
    )


def split_quarantine(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separate rows whose expiry precedes their transaction without a cancel.

    Cancellations legitimately backdate expiry and stay in the main table.
    """
    bad = (
        pl.col("membership_expire_date").is_not_null()
        & (pl.col("membership_expire_date") < pl.col("transaction_date"))
        & ~pl.col("is_cancel")
    ).fill_null(False)
    return df.filter(~bad), df.filter(bad)


def stage_transactions(con: duckdb.DuckDBPyConnection) -> dict:
    """Read transactions.csv in chunks, transform, validate, and insert."""
    assert_header(RAW_TRANSACTIONS, list(TRANSACTIONS_SCHEMA))
    con.execute(TRANSACTIONS_DDL)
    con.execute(TRANSACTIONS_QUARANTINE_DDL)

    reader = pl.read_csv_batched(
        RAW_TRANSACTIONS,
        schema_overrides=TRANSACTIONS_SCHEMA,
        batch_size=CHUNK_ROWS,
        ignore_errors=False,  # a value that will not parse must raise
    )

    raw_rows = 0
    quarantined = 0
    sentinels = 0
    started = time.time()

    while (batches := reader.next_batches(1)) is not None:
        for chunk in batches:
            raw_rows += chunk.height
            sentinels += chunk.filter(
                pl.col("membership_expire_date") == 19700101
            ).height

            df = transform_transactions(chunk)
            clean, bad = split_quarantine(df)
            quarantined += bad.height

            # Validate the chunk that is actually being staged. Quarantined
            # rows are excluded because they have already been accounted for.
            run_checks(STAGE, [lambda c=clean: date_sanity(c, stage=STAGE)])

            insert = clean.select(TRANSACTION_COLUMNS)
            con.register("chunk", insert)
            con.execute("INSERT INTO transactions SELECT * FROM chunk")

            if bad.height:
                q = bad.select(TRANSACTION_COLUMNS).with_columns(
                    quarantine_reason=pl.lit(QUARANTINE_REASON)
                )
                con.register("bad_chunk", q)
                con.execute("INSERT INTO transactions_quarantine SELECT * FROM bad_chunk")

            print(
                f"  transactions: {raw_rows:>11,} rows "
                f"({time.time() - started:5.1f}s)",
                end="\r",
            )

    duplicates = deduplicate(con)

    staged = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(
        f"\n  staged {staged:,} | quarantined {quarantined:,} | "
        f"sentinels nulled {sentinels:,} | exact duplicates dropped {duplicates:,}"
    )
    return {
        "raw_rows": raw_rows,
        "staged_rows": staged,
        "quarantined": quarantined,
        "sentinels_nulled": sentinels,
        "duplicates": duplicates,
    }


def deduplicate(con: duckdb.DuckDBPyConnection) -> int:
    """Drop exact duplicate rows, returning the number removed.

    Repair, not validation -- a check observes and halts, it never fixes. The
    count is returned so reconciliation can account for it; nothing is dropped
    silently.

    Two rows identical in every column are a double-read, not two events. They
    do real damage in three ways: they double-count revenue, and they make the
    spell segmentation non-deterministic, because a duplicate can fall either
    side of a spell boundary depending on which copy the window function sees
    first. Deduplicating here is what makes the pipeline reproducible.

    This runs after all chunks are loaded, deliberately: duplicates can span a
    chunk boundary, so a per-chunk check would miss them.
    """
    before = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    con.execute("CREATE OR REPLACE TABLE tx_dedup AS SELECT DISTINCT * FROM transactions")
    after = con.execute("SELECT COUNT(*) FROM tx_dedup").fetchone()[0]
    con.execute("DROP TABLE transactions")
    con.execute("ALTER TABLE tx_dedup RENAME TO transactions")
    return before - after


def stage_members(con: duckdb.DuckDBPyConnection) -> dict:
    """Read members_v3.csv in chunks, type the dates, and insert."""
    assert_header(RAW_MEMBERS, list(MEMBERS_SCHEMA))
    con.execute(MEMBERS_DDL)

    reader = pl.read_csv_batched(
        RAW_MEMBERS,
        schema_overrides=MEMBERS_SCHEMA,
        batch_size=CHUNK_ROWS,
        ignore_errors=False,
    )

    raw_rows = 0
    started = time.time()

    while (batches := reader.next_batches(1)) is not None:
        for chunk in batches:
            raw_rows += chunk.height
            df = chunk.with_columns(
                to_date("registration_init_time").alias("registration_init_time")
            ).select(MEMBER_COLUMNS)
            con.register("mchunk", df)
            con.execute("INSERT INTO members SELECT * FROM mchunk")
            print(
                f"  members:      {raw_rows:>11,} rows "
                f"({time.time() - started:5.1f}s)",
                end="\r",
            )

    staged = con.execute("SELECT COUNT(*) FROM members").fetchone()[0]
    print(f"\n  staged {staged:,}")
    return {"raw_rows": raw_rows, "staged_rows": staged}


def main() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))

    print(f"Staging into {DB}\n")
    tx = stage_transactions(con)
    mem = stage_members(con)

    # Reconciliation runs last, on totals. Quarantined rows are the only
    # accounted-for difference; anything else halts the stage.
    print("\nReconciling...")
    run_checks(
        STAGE,
        [
            lambda: row_count_reconciliation(
                stage=STAGE,
                raw_rows=tx["raw_rows"],
                staged_rows=tx["staged_rows"],
                logged_drops={
                    QUARANTINE_REASON: tx["quarantined"],
                    "exact_duplicate_rows": tx["duplicates"],
                },
            ),
            lambda: row_count_reconciliation(
                stage=STAGE,
                raw_rows=mem["raw_rows"],
                staged_rows=mem["staged_rows"],
            ),
        ],
    )
    con.close()
    print("Reconciliation passed. Staging complete.")


if __name__ == "__main__":
    main()
