"""Central configuration.

Every tunable that affects results lives here, not scattered through the code.
If a number in the README changes, it should be traceable to a value in this file
or to data.
"""

from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
STAGED = ROOT / "data" / "staged"
MARTS = ROOT / "data" / "marts"
DB = STAGED / "subscriptions.duckdb"
MARTS_DB = MARTS / "marts.duckdb"
VALIDATION_LOG = ROOT / "data" / "validation_log.jsonl"

RAW_TRANSACTIONS = RAW / "transactions.csv"
RAW_MEMBERS = RAW / "members_v3.csv"

# --- Resolved decisions ---------------------------------------------------
# Each of these was an open decision in CLAUDE.md. They are settled in
# docs/decisions.md, with the alternatives that were rejected and why. Change a
# value here only alongside an entry there.

CHURN_GRACE_DAYS = 30  # dataset's own definition: no renewal within 30d of expiry

# Last date any transaction is visible in transactions.csv.
OBSERVATION_CUTOFF = date(2017, 2, 28)

# Last date at which a churn EVENT can be determined. Anyone whose membership
# expires after this has not had the full grace window observed, so their
# outcome is unknowable and they can only be censored. Derived rather than
# typed in: if CHURN_GRACE_DAYS changes, this must move with it.
LABEL_CUTOFF = OBSERVATION_CUTOFF - timedelta(days=CHURN_GRACE_DAYS)

# One subject per user: their first observed subscription spell. Reactivation
# is handled as an empirical LTV component, not a second survival model.
RESUBSCRIBER_RULE = "first_spell_only"

# Primary backtest-eligibility gate. A cohort is a fair forecast target only if
# it has been observed at least as long as the horizon being forecast.
MIN_FOLLOWUP_DAYS = 365

# Secondary, shorter-horizon backtest reported alongside the primary.
SECONDARY_FOLLOWUP_DAYS = 180

# Pre-filter for stratified cuts (e.g. payment method x city). Not the backtest
# gate -- MIN_FOLLOWUP_DAYS is that.
MIN_COHORT_SIZE = 1_000

# The binding guard for stratified cuts. Kaplan-Meier precision is driven by the
# number of EVENTS, not the number of subjects: a cell with 50,000 censored
# subjects and 3 deaths estimates nothing, while 100 events gives a usable
# curve regardless of how many subjects sit behind them. Greenwood's variance
# sums 1/(n_i * (n_i - d_i)) over event times only, so cells are screened on
# events and MIN_COHORT_SIZE merely removes obviously-empty cells first.
MIN_EVENTS_PER_CELL = 100

# Start of the observation window. Tenure accrued before this date is the
# delayed-entry time for left-truncated subjects (lifelines `entry=`).
WINDOW_OPEN = date(2015, 1, 1)

# NOTE: the backtest fit-window end (2016-12-31) is deliberately NOT here. It
# is a property of the backtest, not of the data, and lives in src/backtest/.
# As a global constant it would silently strip 59 days of follow-up from every
# cohort in every downstream analysis.

# --- Data quality ---------------------------------------------------------

# Unix epoch, used as a "missing date" sentinel in membership_expire_date.
# Nulled at ingestion so no downstream consumer has to know it exists.
EPOCH_SENTINEL = date(1970, 1, 1)

CHUNK_ROWS = 1_000_000  # ingestion chunk size
