# Decisions log

One entry per non-obvious choice. Format: decision, alternatives rejected, why,
and what would make me revisit it. This file is the backbone of the README and
the source of interview answers.

## Template

### YYYY-MM-DD — <decision>
**Chose:**
**Rejected:**
**Why:**
**Would revisit if:**

---

### 2026-08-04 — Dataset: KKBox, full transaction history

**Chose:** `transactions.csv` (21,547,746 rows) + `members_v3.csv` (6,769,473
rows), covering 2015-01-01 to 2017-02-28.

**Rejected:** `transactions_v2.csv`, and switching to Online Retail II.

**Why:** the step-1 audit was first run against `transactions_v2.csv`, which is
the competition's March-2017 refresh slice, and it came back degenerate:
74.76% of all rows fell in a single month, 88.12% of users appeared exactly
once, the median observed span per user was **0 days**, censoring was **99.95%**
with only **630** observed churn events, and exactly **one** user in 1.2M showed
a lapse-and-return. No modelling choice recovers from that — there are no
observed lifetimes in the file.

The same audit against the full `transactions.csv` gives 45.69% censoring with
**1,283,798** observed churn events, a median observed span of 186 days, and
19.81% resubscribers. Same dataset, same code, completely different viability.
Online Retail II was rejected because it is a non-contractual retail setting
with no subscriptions, expiry dates, or censoring — it would have forced
BG/NBD/Gamma-Gamma and invalidated the stack and phase plan in CLAUDE.md.

**Would revisit if:** we need `user_logs` engagement covariates for the Cox
model (deferred out of v1), or if the 2015 price artefact below turns out to
contaminate revenue figures beyond repair.

Numbers: [step1_audit.md](step1_audit.md), reproducible via
`notebooks/step1_audit.py`.

---

### 2026-08-04 — Resubscribers: first spell only

**Chose:** one subject per user — their **first** observed subscription spell.
The estimand is **time to first observed churn**. Reactivation is handled
separately as an empirical component feeding LTV: reactivation rate within N
days, and revenue per reactivation. Not a second survival model.

**Rejected:**
- *New subject with a fresh clock* (the precedent in CLAUDE.md). Would take
  subjects from 2,363,626 to 2,955,137 (+25.0%), but 468,250 users (19.81%)
  would contribute 2+ spells (382,332 contribute 2; 58,202 contribute 3;
  25,924 contribute 4-5; 1,792 contribute 6+, max 8).
- *Continued lifetime* — treat gaps as non-payment within one unbroken life.
- *New subject + prior-history covariates.*

**Why:** the fresh-clock options break the independence assumption that both
Kaplan-Meier's Greenwood variance and Cox's partial likelihood rest on. Mean
cluster size would be 1.25 overall and 2.26 among returners. The point
estimates would stay consistent — clustering does not bias the KM curve — but
confidence intervals would be too narrow and Cox p-values anti-conservative.
Correcting for that means a user-level cluster bootstrap for KM (lifelines has
no cluster-robust KM variance) and `cluster_col=` for Cox. That is real work
for a secondary result.

First-spell-only buys exact independence by construction, at the cost of
591,511 discarded spells (25% of the information). "Time to first churn" is a
well-defined estimand that needs no caveat, and the reactivation behaviour is
not thrown away — it is modelled explicitly where it is easier to defend, as
an empirical rate rather than a second correlated survival process.

*Continued lifetime* was rejected outright: with 19.81% of users lapsing and
returning, treating a six-month absence as continuous membership would
systematically understate churn and silently convert subscription time into
calendar time.

**Would revisit if:** time permits a robustness appendix — refit as all-spells
with `CoxPHFitter(..., cluster_col="msno")` and a user-level cluster bootstrap
for KM, and report naive vs clustered SEs side by side. The ratio is the design
effect and is worth showing. This is an appendix, not the headline.

---

### 2026-08-04 — Two cutoffs: observation and labelling are different dates

**Chose:** both in `src/config.py`.
- `OBSERVATION_CUTOFF = 2017-02-28` — last date any transaction is visible.
- `LABEL_CUTOFF = 2017-01-29` — `OBSERVATION_CUTOFF` minus the 30-day grace
  window; the last date at which a churn *event* can be determined.

All observation hard-stops at `LABEL_CUTOFF`. The 2016-12-31 backtest split is
**not** a global cutoff — it is a fit-window parameter local to `src/backtest/`.

**Rejected:**
- *A single cutoff at 2017-02-28, censoring undetermined users at their expiry
  date.* Correct if implemented perfectly, and it retains 30 more days of event
  information.
- *A global cutoff at 2016-12-31* holding Jan-Feb 2017 as a pure holdout.

**Why:** the dataset's churn definition requires observing 30 days past expiry
with no renewal. For anyone expiring after 2017-01-29 that evidence cannot
exist. **The final 30 days can only ever produce censored observations, never
events** — 95,709 users (4.05%) sit in that band. Keeping the two dates
separate and named makes it structurally impossible to label an event we could
not have observed.

The single-cutoff option was rejected not because it is wrong but because it is
*fragile*: it requires censoring each undetermined user at their own expiry
date rather than at the cutoff, and censoring at the cutoff — the easy mistake —
claims follow-up we do not have. Two explicit constants are self-documenting;
the cost is 30 days of genuine events, uniformly across cohorts.

Making 2016-12-31 a global cutoff was rejected because a backtest split is a
property of the backtest, not of the data. As a global constant it would
silently strip 59 days of follow-up from every cohort in every downstream
analysis, including ones that have nothing to do with backtesting.

**Would revisit if:** the churn grace window changes from 30 days, in which
case `LABEL_CUTOFF` must be recomputed from `CHURN_GRACE_DAYS` rather than
re-entered by hand.

---

### 2026-08-04 — Backtest eligibility: follow-up, not cohort size

**Chose:** `MIN_FOLLOWUP_DAYS = 365` as the primary backtest-eligibility gate.
Report the 180-day backtest alongside the 365-day one as a secondary result.
`MIN_COHORT_SIZE` is retained only as a secondary guard for stratified cuts.

**Rejected:** `MIN_COHORT_SIZE` as the primary gate (its original framing in
CLAUDE.md), and 180-day or 730-day horizons as the primary.

**Why:** size is not the binding constraint. The smallest monthly cohort is
2017-02 at 32,634 users — ample for a Kaplan-Meier curve. **All 26 cohorts are
estimable.** What actually varies is available follow-up, from 759 days
(2015-01) down to 0 (2017-02). A cohort is a fair forecast target only if it
has been observed at least as long as the horizon being forecast; otherwise the
forecast is compared against a truncated actual and the error metric flatters
the model.

| horizon | eligible cohorts | users | share |
|---|---|---|---|
| 180d | 2015-01 -> 2016-08 (20) | 1,954,949 | 82.7% |
| **365d** | **2015-01 -> 2016-02 (14)** | **1,637,715** | **69.3%** |
| 730d | 2015-01 -> 2015-02 (2) | 599,942 | 25.4% |

365 days is the natural LTV unit and the only horizon that captures the 410-day
annual plans present in the price grid. 730 days leaves 2 cohorts — too few to
separate forecast error from cohort idiosyncrasy. 180 days is reported as a
secondary result because it retains 82.7% of users and gives a second,
shorter-horizon read on the same model.

Size survives as a secondary guard because it does bind on *stratified*
analysis: 32,634 users split by payment method x city produces thin cells even
though the marginal cohort is large.

**Would revisit if:** the backtest fit-window moves earlier than 2016-12-31,
which shifts every row of that table; or if stratified cuts turn out to need a
higher `MIN_COHORT_SIZE` than currently set.

---

### 2026-08-04 — Left truncation: delayed entry via lifelines `entry=`

**Chose:** delayed entry. Subjects enter the risk set at the tenure already
accrued at window open, computed from `registration_init_time`, rather than at
t=0. Applies to `KaplanMeierFitter` and `CoxPHFitter` via `entry=`.

The **432,623 users (18.3%) with no `members_v3` row are excluded from the
truncated population** — no registration date means no computable entry time.
This exclusion is logged with its count and reason at the cohort-build
boundary, per the no-silent-data-loss rule.

**Rejected:**
- *Drop the 2015-01 cohort.* Loses 548,792 users (23.2%) and does not fix the
  problem: 2015-02 through 2015-06 remain 61-85% truncated. Fixing it by
  dropping means dropping most of 2015 — precisely the mature-cohort pool that
  the 365-day backtest gate depends on.
- *Define cohorts on `registration_init_time`, keeping only reg >= 2015-01-01.*
  True incident cohorts and the cleanest semantics, but discards the large
  majority of early cohorts and inherits the same 18.3% unmatched problem.
- *Accept truncation and restate the estimand as a prevalent cohort.* Cheap and
  honest, but the resulting curve is conditional on having survived to window
  open and is not comparable to a signup-cohort curve.

**Why:** left truncation is severe and not confined to 2015-01 — that cohort is
97.5% truncated, but 2015-02 is 72.7%, 2015-03 84.5%, 2015-04 63.5%, 2015-05
61.1%, 2015-06 64.2%. Among truncated 2015-01 users, accrued tenure at window
open is p25 425d, **median 865d**, p75 1457d, p90 2573d. Half had been
subscribers for over two years before the data begins. Ignoring that treats
long-tenured survivors as new signups and biases every curve.

Delayed entry is only safe if `registration_init_time` approximates
subscription start. **Validation (2026-08-04):** among the 875,924 users
registered inside the observation window — the only users whose full history is
visible — the pooled gap from registration to first transaction is median 0
(50.03% same-day) but p90 240 days. That pooled figure is misleading, because
it is dominated by `registered_via=4`, which has a median lag of 87 days and
**zero users in the truncated population**. Re-weighted to the truncated
group's own channel mix (via 9: 44.0%, median lag 2d; via 7: 30.6%, median lag
0d; via 3: 25.4%, median lag 45d), the **weighted median lag is 12.3 days
against median accrued tenure of 865 days — a ~1.4% overstatement of entry
time.** Small enough to proceed.

**Assumption recorded:** delayed entry requires subjects to be **event-free
before entry**. For pre-window history this is **unverifiable** — a user who
churned and resubscribed in 2013 is indistinguishable from one who subscribed
continuously since 2013. Combined with the first-spell-only rule, this means
"first spell" is really "first spell *observed in the window*", which is not
the same thing for truncated users.

**Would revisit if:** either of two caveats bites.
1. **Tail risk.** `registered_via=3` is 25.4% of the truncated population with
   a p90 lag of 459 days. The median is reassuring; the upper quartile is not.
   If survival estimates prove sensitive to entry-time perturbation, refit
   under the prevalent-cohort framing.
2. **Time-stability.** Truncated users' lag cannot be measured; it is inferred
   from post-2015 registrants assuming within-channel behaviour is stable over
   time. `registered_via=4` appearing only after 2015 is direct evidence the
   channel mix *did* shift, which weakens that assumption.

A sensitivity check comparing delayed-entry against prevalent-cohort curves
would settle both and is cheap to run once Phase 4 exists.

---

### 2026-08-04 — Zero-day rows: impute, flag, and keep revenue reproducible

**Chose:** for the 870,124 rows with `payment_plan_days = 0`, impute
`plan_days` from `membership_expire_date - transaction_date` and `list_price`
from `actual_amount_paid`, and carry a boolean **`plan_days_imputed`** column.
**Every revenue figure must be reproducible with imputed rows excluded** — the
flag is what makes that possible, so it propagates through staging into the
marts.

**Rejected:**
- *Treat as a distinct product type.* The profiling says it is not one.
- *Drop the rows.* Would lose renewals for 500,052 users and tear eight holes
  through 2015 — the exact period the mature cohorts live in.

**Why:** profiled before deciding, per instruction. The rows are **not** an
identifiable product:
- **Temporally bounded.** Zero rows in 2015-01/02, 870,118 rows across
  2015-03 -> 2015-10 (8.8%-41.2% of monthly volume), and 6 rows in the
  following 16 months. A real product does not appear, run 8 months, and vanish.
- **Not payment-method-specific.** Spread over 12+ methods (34: 24.6%,
  41: 23.5%, 33: 14.1%, 39, 31, 40, 38, ...), all of which are heavily used
  outside the window — method 41 has 11.3M normal rows against 204k zero-day.
- **They extend membership.** 859,315 of 870,124 (98.8%) set expiry after the
  transaction date, median **+31 days**, clustering at 30-31 days (77%).
- **Real money at standard prices.** `actual_amount_paid` is 149 in 89% of
  rows, then 129, 119, 150 — exactly the standard monthly price points. The
  same users buy 30d @ 149 normally (6.7M rows).

Conclusion: these are ordinary ~30-day renewals where `payment_plan_days` and
`plan_list_price` were left unpopulated by the source system — a field
population defect bounded to eight months of 2015. The information is
**recoverable from fields that are present**, which is why imputation beats
dropping. They also explain most of the 7.96% price mismatch: the 843,898
"overpaid, not cancelled" rows are largely this population (paid 149 against a
list price of 0).

**Would revisit if:** the imputed duration distribution turns out to be
materially wider than 30-31 days on closer inspection, or if a revenue figure
computed with and without imputed rows diverges enough to change a conclusion.

---

### 2026-08-04 — Epoch-zero expiry dates nulled at ingestion

**Chose:** the 1,776 rows with `membership_expire_date = 1970-01-01` are set to
NULL **at ingestion**, not downstream, with the count and reason logged.

**Rejected:** carrying them through staging and filtering later.

**Why:** `1970-01-01` is the Unix epoch — a missing date encoded as `0`, not a
real expiry. Left in place it poisons any min/max: it is why the resubscriber
max gap reads 17,225 days (47 years) while the median is a sane 71. Nulling at
the earliest boundary means no downstream consumer has to know the sentinel
exists. NULL is the honest representation of "we do not know".

**Would revisit if:** other sentinel encodings turn up (e.g. `19700101` in
`transaction_date` or `registration_init_time`), in which case this becomes a
general sentinel rule rather than one hardcoded date.

---

### 2026-08-04 — `date_sanity` expiry rule is conditional on `is_cancel`

**Chose:** the expiry-before-transaction rule fails only for rows where
`is_cancel = 0`. Cancellations are exempt and counted separately as WARN.

**Rejected:** a blanket `membership_expire_date >= transaction_date` rule.

**Why:** 153,660 rows have expiry before the transaction date, but **147,200 of
them (95.8%) are `is_cancel = 1`**, where backdating the expiry is the correct
representation of a cancellation — the membership genuinely ends before the
transaction that recorded it. A blanket rule would fail 147k legitimate rows
and halt the pipeline on correct data. Only the **6,460** non-cancel rows are
genuinely suspect.

**Would revisit if:** the 6,460 non-cancel violations turn out to have a
pattern of their own worth handling explicitly rather than failing.

---

### 2026-08-04 — Backdated cancellations: spell end and event time are different dates

**Chose:** the cancel row's `membership_expire_date` is **never** used as a
spell end. Instead:
- **Spell end** = `membership_expire_date` of the last preceding **non-cancel**
  transaction in the spell.
- **Event time** = the cancel row's `transaction_date`.

A hard guard follows: any subject whose **computed tenure is negative** is
quarantined with a reason and reported, never silently dropped or clamped.

**Rejected:**
- *Using the cancel row's own expiry as the spell end.* This is what Phase 1
  staging left in place.
- *A threshold rule* — e.g. reject backdating beyond N days. Rejected because
  it needs an arbitrary N, and every N leaves a tail of absurd-but-under-N rows
  that pass silently.

**Why:** Phase 1 surfaced **7,232 transactions whose expiry precedes the
observation window**, all of them `is_cancel = 1`. The most extreme is a
transaction on 2016-01-22 setting expiry to **2005-11-26 — backdated 3,709
days**. A cancellation that claims membership ended a decade before the
customer cancelled is a bookkeeping artefact, not a fact about the subscription.
Taken at face value it produces negative tenures and silently corrupts every
survival curve that includes those subjects.

The two-date split is what makes this coherent. The membership genuinely ran to
the last paid-for expiry; the customer's decision to leave happened on the
cancel transaction date. Those are different events and forcing them into one
field is what created the problem. Separating them removes the need for a
threshold entirely: the cancel row's expiry is simply never read, so however
absurd it is, it cannot propagate.

The negative-tenure guard exists because this rule is necessary but not
provably sufficient — it fixes the known 7,232 and any row shaped like them,
and the guard catches whatever shape we have not thought of.

**Would revisit if:** the guard catches a material number of subjects, which
would mean a second distinct corruption pattern exists and needs its own rule
rather than a quarantine.

---

### 2026-08-04 — Stratified cells are screened on events, not subjects

**Chose:** `MIN_EVENTS_PER_CELL = 100` as the binding guard for stratified
cuts. `MIN_COHORT_SIZE = 1_000` is retained as a cheap pre-filter only.

**Rejected:** screening cells on subject count alone.

**Why:** Kaplan-Meier precision is **event-driven, not subject-driven**.
Greenwood's variance sums `1 / (n_i * (n_i - d_i))` over **event times only** —
censored observations contribute nothing to it. A cell with 50,000 subjects and
3 deaths estimates essentially nothing, while 100 events yields a usable curve
regardless of how many subjects sit behind them. Screening on subjects would
wave through exactly the cells whose curves are least trustworthy. With overall
censoring at 45.69%, subject count and event count are not interchangeable, and
they diverge further in any cell stratified on a churn-correlated covariate.

`MIN_COHORT_SIZE` survives only to discard obviously-empty cells before the
more meaningful event count is computed.

**Would revisit if:** stratified cuts turn out to need a higher event floor for
stable tail estimates, which the confidence-band width will show directly.

---

### 2026-08-04 — Left truncation: proceeding with delayed entry, plus sensitivity

**Chose:** delayed entry, confirmed. `registered_via` enters the Cox model as a
covariate. A planned sensitivity analysis runs delayed-entry KM **with and
without `registered_via = 3`**.

**Direction of bias, recorded explicitly:** overstated entry times bias
**survival upward at short tenures — that is, optimistically.** A subject
entered at tenure 800 days when their true subscription tenure is 400 is placed
in a risk set they do not belong to, and having demonstrably survived to the
point of entry, they contribute only survival to those intervals. The error is
therefore not symmetric or self-cancelling: it inflates the early part of the
curve, which is precisely the region that drives LTV. Any LTV figure derived
from these curves should be read as an upper bound until the sensitivity
analysis bounds the size of the effect.

**Why `registered_via` as a covariate:** the channel is not a nuisance
variable, it is the mechanism. The validated registration-to-first-transaction
lag varies by channel from 0 days (via 7, 99.0% within a week) to 45 days
median with a p90 of 459 (via 3). Since entry-time error is channel-determined,
conditioning on channel lets the Cox model absorb the part of the bias that is
systematic rather than leaving it in the baseline hazard.

**Why the via-3 sensitivity specifically:** via 3 is 25.4% of the truncated
population and carries by far the worst lag tail. Refitting without it gives a
direct read on how much of the survival estimate depends on the subgroup whose
entry times are least trustworthy. If the curves separate materially, the
delayed-entry framing is doing more work than the data supports and the
prevalent-cohort framing becomes the honest fallback.

**Would revisit if:** the with/without-via-3 curves diverge beyond their
confidence bands, or if the Cox `registered_via` coefficients imply the channel
effect is implausibly large for a variable that should mostly describe
acquisition rather than retention.

---

### 2026-08-04 — Spell construction must be deterministic: a total row order

**Chose:** every window function in `src/cohorts/spells.py` orders by
`SPELL_ORDER`, a **total** order over all nine transaction columns, and the
candidate set is materialised as a table rather than left as a view.

**Rejected:** ordering on `(transaction_date, membership_expire_date)`, the
obvious choice and the one originally written.

**Why:** that pair is not a total order. 27,942 user-days carry more than one
transaction, and rows tying on both dates but differing in `is_cancel` received
an arbitrary relative order inside the `ROWS BETWEEN UNBOUNDED PRECEDING`
frame. Since the running coverage deliberately ignores cancellations, whichever
row DuckDB happened to place first changed the accrued coverage, which moved
the spell boundary, which changed the first spell, which changed the subject
count. Two runs of identical code over identical data returned 1,872,662 and
1,872,599 subjects.

This surfaced twice before it was understood. First as a reconciliation
failure: `MAX_BY(expiry, transaction_date)` broke ties differently on separate
evaluations of the same view, so 51 subjects were written to both the spell
table and the quarantine, and `unexplained` came back as -51. Materialising the
candidates fixed that symptom. The subject count still drifted between runs,
which is what exposed the underlying cause in the window ordering.

A portfolio project whose headline numbers move between runs cannot be
defended, and the reconciliation check is what made the drift visible rather
than merely present.

**Note on what this does and does not buy:** determinism, not correctness. When
a renewal and a cancellation share a date, nothing in the data establishes
which happened first, and the spell boundary genuinely depends on it. The total
order makes the pipeline reproducible; `duplicate_transactions` reports the
exposure at every build so the residual ambiguity stays visible.

**Would revisit if:** a tie-breaking rule with business meaning emerges — for
instance if `payment_method_id` or a transaction sequence number turns out to
encode intra-day ordering, which would make the order correct as well as total.

---

### 2026-08-04 — Subjects whose spell starts after LABEL_CUTOFF are excluded

**Chose:** a spell beginning after `LABEL_CUTOFF` (2017-01-29) is quarantined
with reason `starts_after_label_cutoff`. **29,350 subjects**, which is the
whole 2017-02 cohort and part of 2017-01.

**Rejected:** censoring them at `LABEL_CUTOFF` alongside everyone else, which
is what the first implementation did.

**Why:** censoring a subject at `LABEL_CUTOFF` when their spell began *after*
it places the censor date before the spell start. The subject is recorded as
having been observed for a period that ends before it begins. For those whose
registration was also recent this drove tenure negative, and the negative-
tenure guard caught them — reporting **18,737** quarantined subjects, of which
**18,707 were this bug rather than corrupt data**. The remainder slipped
through with positive tenure and an incoherent censor date, which is worse,
because nothing flagged them.

Under the two-cutoff rule these subjects have zero observable follow-up: no
outcome is knowable for them at any horizon. Excluding them explicitly, with a
reason that says so, is the honest representation. Once fixed, the genuine
negative-tenure count is **30**.

The general lesson, worth stating because it nearly went unnoticed: a guard
firing in volume is evidence about the code before it is evidence about the
data. 18,737 "corrupt" rows in a dataset this clean was implausible, and taking
the guard's label at face value would have put a fabricated data-quality claim
in the README.

**Would revisit if:** the backtest needs these subjects as forecast targets
rather than as fitting data, in which case they return with a horizon of zero
and are scored on prediction alone.

---

### 2026-08-04 — The 460,348 excluded users are not missing at random

**Decomposition of the exclusion bucket** (2,363,590 staged users ->
1,873,529 subjects):

| reason | count |
|---|---|
| no `members_v3` row (no registration -> no entry time) | **432,592** |
| first spell contains only cancellations (no expiry to end it) | **18,660** |
| every transaction is a cancellation | **9,096** |
| `starts_after_label_cutoff` (quarantined) | 29,350 |
| `entry_after_exit` (quarantined) | 333 |
| `negative_tenure` (quarantined) | 30 |

**Finding: the 432,592 users with no members row differ systematically from
retained subjects on every transaction-only observable available.**

| observable | no members row | retained |
|---|---|---|
| first `actual_amount_paid` (median) | **0** | 149 |
| `is_auto_renew` at first transaction | **86.7%** | 53.6% |
| transactions per user (median) | **2** | 6 |
| `payment_method_id = 41` | **80.4%** | 33.4% |
| over-represented cohorts | 2015-10 → 2015-12 | 2015-01, 2015-06 |

They start free, on one payment method, with auto-renew on, and transact twice.
That is a distinct acquisition channel — most likely a bundled or partner
promotion — whose members never received a `members_v3` record. The missingness
is a property of the channel, not a random omission.

**Direction of bias: survival estimates are biased UPWARD (optimistic).**
Measured, not assumed, by recomputing the first-spell outcome from first
transaction — a clock available for both groups:

| | no members row | retained |
|---|---|---|
| first-spell event rate | **75.5%** | 62.7% |
| median first-spell tenure | **31 days** | 180 days |
| still running at 90d | **34.0%** | 59.0% |
| still running at 365d | **17.9%** | 35.2% |

The excluded group churns roughly twice as fast. Dropping them removes
short-lived subscribers preferentially, so the retained sample over-represents
durable ones and every survival curve sits too high.

**This compounds rather than offsets the delayed-entry bias**, which is also
upward. Two independent exclusions push the same direction, so LTV derived from
these curves should be read as an upper bound, and the README must say so
rather than quoting a point estimate.

**Would revisit if:** an entry time can be constructed for these users without
`registration_init_time` — first transaction date is the obvious candidate, at
the cost of a different clock for 23% of the population. That is a real option
and it is the single largest lever on the bias in this project.

---

### 2026-08-05 — The plan-type crossing is a product effect, not a truncation artefact

**Tested:** refitted `<=7d` and `8-31d` on incident subjects only
(`entry_days == 0`, 796,544 subjects, 46.5% of the pair), removing delayed
entry entirely.

**Result: the crossing PERSISTS.**

| pair | 30d | 60d | 90d | 180d | 365d |
|---|---:|---:|---:|---:|---:|
| all subjects | +0.008 | +0.032 | −0.008 | −0.179 | −0.368 |
| incident only | +0.009 | +0.033 | −0.014 | −0.211 | −0.407 |

Both cross between 60d and 90d, and the incident-only gap is slightly *wider*.
So `<=7d` genuinely has higher early survival than `8-31d` before collapsing —
plausibly because a short plan plus a 30-day grace window puts a floor under
observed tenure — and the ordering then reverses decisively.

This is the opposite of the convenient answer. Had the crossing vanished on
incident subjects it would have been an artefact of delayed entry and
`plan_type` could have gone into a Cox model as an ordinary covariate.
**It did not, so proportional hazards is genuinely violated and `plan_type`
must be used in `strata=`.**

**Truncation profile, for context (the 8-31d row was missing before):**

| stratum | n | truncated | median entry | median entry given truncated | incident |
|---|---:|---:|---:|---:|---:|
| `01_<=7d` | 372,520 | 39.6% | 0d | 560d | 225,000 |
| `02_8-31d` | 1,338,978 | **57.3%** | **236d** | **876d** | 571,544 |
| `03_32-120d` | 19,854 | 31.4% | 0d | 700d | 13,623 |
| `04_121d+` | 142,176 | **75.7%** | **657d** | 962d | 34,557 |

**Would revisit if:** a hazard-based diagnostic (log-log plot, Schoenfeld
residuals once Cox exists) disagrees with the crossing evidence.

---

### 2026-08-05 — The survival curve is bracketed, not point-estimated

**Chose:** report survival and everything derived from it as a **bracket**.

| t | upper (main table) | lower (+ no-members at entry=0) | width |
|---|---:|---:|---:|
| 30d | 0.9155 | 0.7584 | **15.71pp** |
| 90d | 0.7653 | 0.6310 | 13.43pp |
| 365d | 0.4414 | 0.3805 | **6.10pp** |
| 730d | 0.2627 | 0.2234 | 3.93pp |

Median survival: **295 days (upper) vs 209 days (lower)** — the bracket is
41% of the lower estimate, far too wide to be rounded away.

**Why each end is a bound rather than an estimate:**
- **Upper.** The main table excludes 432,592 users who churn about twice as
  fast. Removing fast churners preferentially lifts the curve.
- **Lower.** Adding them back at `entry = 0` treats them as incident. If they
  were themselves subscribing before 2015-01-01 their tenure is understated and
  their churn overstated, pushing the curve down.

The lower curve also mixes two clocks — registration for the main table, first
transaction for the added subjects — which is defensible for bracketing but
would not be defensible as an estimate. That is precisely why it is presented
as a bound and never as a headline number.

**Would revisit if:** an entry time can be constructed for the no-members
users, collapsing the bracket to a single clock. This remains the largest
single lever on accuracy in the project.
