"""Explicit schemas for the raw CSVs, and the DDL for their staged tables.

Nothing here infers a dtype. polars is given the full schema up front, so an
unexpected column or an unparseable value raises instead of quietly becoming a
null — silent coercion is the failure mode this module exists to prevent.

KKBox stores all dates as YYYYMMDD integers. They are read as integers and
converted to real dates in one explicit, strict step (`to_date`), never by
inference.
"""

from __future__ import annotations

import polars as pl

# --- raw CSV schemas ------------------------------------------------------
# Order matters: polars matches these positionally against the header, so a
# reordered or renamed source column fails loudly rather than shifting values
# into the wrong field.

TRANSACTIONS_SCHEMA: dict[str, pl.DataType] = {
    "msno": pl.String,
    "payment_method_id": pl.Int16,
    "payment_plan_days": pl.Int32,
    "plan_list_price": pl.Int32,
    "actual_amount_paid": pl.Int32,
    "is_auto_renew": pl.Int8,
    "transaction_date": pl.Int32,          # YYYYMMDD
    "membership_expire_date": pl.Int32,    # YYYYMMDD
    "is_cancel": pl.Int8,
}

MEMBERS_SCHEMA: dict[str, pl.DataType] = {
    "msno": pl.String,
    "city": pl.Int16,
    # `bd` is self-reported age and is known to contain implausible values
    # (negative, and in the thousands). Staged as-is with a wide type; cleaning
    # it is a modelling decision, not an ingestion one.
    "bd": pl.Int32,
    "gender": pl.String,  # genuinely nullable in the source
    "registered_via": pl.Int16,
    "registration_init_time": pl.Int32,    # YYYYMMDD
}


def to_date(column: str) -> pl.Expr:
    """Convert a YYYYMMDD integer column to a Date, strictly.

    Strict by design: a value that does not parse raises rather than becoming
    null. A null here would be indistinguishable from the epoch sentinel that
    ingestion deliberately nulls, and the two mean different things — "the
    source said nothing" versus "the source said something impossible".
    """
    return pl.col(column).cast(pl.String).str.strptime(pl.Date, "%Y%m%d")


# --- staged table DDL -----------------------------------------------------
# Written out rather than inferred from the frame so the staged contract is
# visible in one place and reviewable.

TRANSACTIONS_DDL = """
CREATE OR REPLACE TABLE transactions (
    msno                   VARCHAR NOT NULL,
    payment_method_id      SMALLINT NOT NULL,
    plan_days              INTEGER,          -- NULL when unrecoverable
    list_price             INTEGER NOT NULL,
    actual_amount_paid     INTEGER NOT NULL,
    is_auto_renew          BOOLEAN NOT NULL,
    transaction_date       DATE NOT NULL,
    membership_expire_date DATE,             -- NULL where epoch sentinel
    is_cancel              BOOLEAN NOT NULL,
    plan_days_imputed      BOOLEAN NOT NULL
)
"""

# Rows that fail date_sanity's non-cancellation rule are quarantined here
# rather than deleted, so no row is ever lost and the reconciliation balances.
TRANSACTIONS_QUARANTINE_DDL = """
CREATE OR REPLACE TABLE transactions_quarantine (
    msno                   VARCHAR NOT NULL,
    payment_method_id      SMALLINT NOT NULL,
    plan_days              INTEGER,
    list_price             INTEGER NOT NULL,
    actual_amount_paid     INTEGER NOT NULL,
    is_auto_renew          BOOLEAN NOT NULL,
    transaction_date       DATE NOT NULL,
    membership_expire_date DATE,
    is_cancel              BOOLEAN NOT NULL,
    plan_days_imputed      BOOLEAN NOT NULL,
    quarantine_reason      VARCHAR NOT NULL
)
"""

MEMBERS_DDL = """
CREATE OR REPLACE TABLE members (
    msno                   VARCHAR NOT NULL,
    city                   SMALLINT NOT NULL,
    bd                     INTEGER NOT NULL,
    gender                 VARCHAR,
    registered_via         SMALLINT NOT NULL,
    registration_init_time DATE NOT NULL
)
"""
