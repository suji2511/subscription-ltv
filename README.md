# Subscription Retention & Lifetime Value Forecasting

Cohort retention, survival-based churn modelling, and backtested LTV forecasting
on 21.5M subscription transactions from the KKBox Churn Prediction dataset.

Every number below traces to code in this repository, and is reproducible by
re-running the module named beside it.

---

## The headline result, and the direction it is wrong in

Median first-spell survival is **295 days** pooled — but that figure describes
no product actually on offer. It is a mixture across four plan types whose
medians span 177 to 576 days.

More importantly: **every survival estimate in this project is an upper bound.**
Two independent biases push the same way, and neither offsets the other.

**1. Delayed entry with imperfect entry times.** 54.9% of subjects are
left-truncated. Entry times come from account registration, which can precede
paid subscription. Re-weighted to the truncated population's own channel mix,
the median registration-to-first-transaction lag is **12.3 days against a median
accrued tenure of 865 days** — a ~1.4% overstatement of entry time. Small at the
median, but `registered_via = 3` (25.4% of the truncated population) carries a
p90 lag of 459 days. Overstating entry means skipping risk time the subject
actually lived through, which inflates survival at short tenures.

**2. A fast-churning excluded population.** 432,592 subjects have no member
record and so cannot be given an entry time. They are not missing at random:
they start free (median first payment **0 vs 149**), are **86.7% auto-renew vs
53.6%**, transact twice against six times, and 80.4% use a single payment
method. Measured on a comparable clock they churn about twice as fast —
**365-day survival 17.9% vs 35.2%**, median first-spell tenure 31 days vs 180.

**A third bias ran the other way and has been corrected.** The reactivation rate
was computed over all churners, including those who could not yet be observed
returning — 52% of them lacked a full 365 days of post-churn follow-up. That
censored the rate downward (**33.92% → 40.85%**) and understated LTV. It is
fixed, so **LTV inherits the survival bias only**, and that bias points upward.

### The bracket

Because the excluded population can be rebuilt on its own clock, the true curve
can be bounded from both sides rather than only from above.

| t | Upper (main analysis) | Lower (excluded population re-included at entry=0) | Width |
|---|---:|---:|---:|
| 30d | 0.9155 | 0.7584 | 15.71pp |
| 365d | 0.4414 | 0.3805 | 6.09pp |
| 730d | 0.2627 | 0.2234 | 3.93pp |

Median survival: **295 days upper, 209 days lower** — a bracket 41% of the lower
estimate. Neither end is an estimate. The lower curve mixes two clocks
(registration for the main table, first transaction for the added subjects),
which is acceptable for bounding and not acceptable as a headline.

The backtest independently confirms the optimism, below.

*Source: `src/models/km.py`.*

---

## Why survival analysis, not classification

The obvious approach is binary churn classification, which is what the original
Kaggle competition asked for. That framing discards the information LTV needs:
*when* churn happens, and the fact that most subscriptions have not ended yet.

36.50% of subjects are censored. A classifier must either drop them or label
them "not churned", and both corrupt the estimate. Survival analysis handles
censoring natively and yields a curve, which is what LTV integrates against.

**Plan type is a stratum, not a covariate — on evidence, not preference.** Under
proportional hazards a constant hazard ratio implies curves ordered identically
at every *t*, so they cannot cross. Two of six stratum pairs cross: `<=7d` vs
`8-31d` between 60 and 90 days, and `8-31d` vs `32-120d` between 180 and 365
days. Refitting on incident subjects only (`entry_days == 0`, 796,544 subjects)
*widens* the first crossing rather than removing it, ruling out a truncation
artefact. Fitting `plan_type` as a Cox covariate would be misspecified no matter
how good its p-values looked.

*Source: `src/models/km.py` — `check_crossings`, `incident_only_check`.*

---

## Data, and the audit that justified it

The dataset ships in two forms, and the difference decides the project.

An initial audit against `transactions_v2.csv` (1.43M rows) returned a
degenerate result: **99.95% censoring with 630 observed events**, 74.76% of rows
in a single month, and a **median observed span per user of zero days**. That
file is the competition's March-2017 scoring refresh, not a subscription
history. No survival model is estimable on it.

The full `transactions.csv` (21.5M rows, 26 months) audits as a usable survival
dataset: **45.69% censoring, 1,283,798 observed events**, cohorts spread across
the window, and **19.81% resubscription**.

Both runs are preserved in `docs/step1_audit.md`, stamped with their source
file, because the two sets of numbers must never be mistaken for each other.

*Source: `notebooks/step1_audit.py`.*

### Censoring: why the audit says 45.69% and the marts say 36.50%

Two numbers describing two different things. The difference is fully accounted
for, step by step:

| step | n | censoring | delta |
|---|---:|---:|---:|
| **A.** Audit definition — per user, state of their *last* transaction, at `OBSERVATION_CUTOFF` (2017-02-28) | 2,363,590 | **45.71%** | — |
| **B.** Switch the unit to *first spell* | 2,318,905 | 34.02% | **−11.69%** |
| **C.** Switch cutoff to `LABEL_CUTOFF` (2017-01-29) | 2,318,905 | 35.91% | +1.89% |
| **D.** Restrict to retained subjects | 1,873,529 | **36.50%** | +0.59% |

(A reproduces the audit's 45.69% to within 0.02%, the gap being 36 users lost to
the staging quarantine.)

**The unit change is the whole story — it accounts for −11.69% of a −9.21% net
move.** The audit asks "is this user subscribed right now?", which censors
anyone currently active. The marts ask "did this user's *first* subscription
end?", and 19.81% of users lapsed and returned. Under the audit's question those
returners look active and are censored; under the first-spell estimand their
first spell demonstrably ended, so they are events. That is not a discrepancy to
reconcile away — it is the first-spell-only decision doing exactly what it was
chosen to do.

The other two components push the opposite way and are small. Moving the cutoff
30 days earlier (C) converts late events into censored observations, as
intended: the final 30 days can only ever produce censored observations.
Restricting the population (D) removes users who churn roughly twice as fast,
which raises the censoring rate slightly — see the selection-bias entry in
`docs/decisions.md`.

---

## Pipeline

```
raw CSV → DuckDB staging → marts (spell table) → models → backtest
```

Validation runs at every stage boundary. Checks observe and halt; they never
repair. Repair happens in the transform, where it is visible and testable.
Results append to `data/validation_log.jsonl`.

**Reconciliation balances exactly**, because bad rows are quarantined or
deduplicated with a logged reason rather than deleted:

```
21,539,475 staged + 4,932 quarantined + 3,339 exact duplicates = 21,547,746 raw
```

**The halt is proven, not asserted.** A 2099 transaction date fault-injected
into a 2,000-row subset produced a `ValidationError` with **0 rows inserted** —
it does not stage a partial table and then complain.

*Source: `src/ingest/stage.py`, `src/validate/checks.py`.*

### Three bugs the pipeline caught in its own code

**1. A guard firing in volume is evidence about the code before it is evidence
about the data.** The tenure guard initially reported 18,737 negative tenures.
**18,707 were a clamping bug** — subjects whose spell began after the label
cutoff had their censor date pushed before their own start date, and those with
older registrations kept a *positive* tenure and passed silently, which is
worse. Genuine negative tenures: **30**. Reported as-is, 18,737 would have been
a fabricated data-quality claim in this README.

**2. The pipeline was non-deterministic.** `ORDER BY (transaction_date,
membership_expire_date)` is not a *total* order; rows tying on both dates but
differing on `is_cancel` were ordered arbitrarily inside the window frame,
shifting spell boundaries between runs. Now ordered on all nine columns.

**3. A passing test proves the check works, not that it runs where the defect
lives.** `duplicate_transactions` was written, tested and passing — but wired
only into staged→marts, where the frame is one row per user and therefore
*structurally cannot see* duplicate transaction rows. **3,339 exact duplicates
survived into staging and double-counted 343,788 in revenue.** They also
reintroduced non-determinism, because a duplicate can fall either side of a
spell boundary depending on which copy the window function sees first.

The claim that rebuilds were byte-identical was **asserted in the Phase 3 report
and was false when made** — rebuilds returned 1,873,529 / 1,873,530 /
1,873,532. The final audit caught it. Deduplication now happens in the ingest
transform with its count fed into reconciliation, and three consecutive rebuilds
are verified identical.

**Caveat worth stating:** a total order buys reproducibility, not correctness.
When a renewal and a cancellation share a date, nothing in the data says which
came first.

---

## Modelling decisions

Full reasoning and revisit conditions in `docs/decisions.md`. Summary:

| Decision | Chose | Rejected because |
|---|---|---|
| Resubscribers | First spell only | "New subject" makes 25% of subjects non-independent, breaking Greenwood CIs and Cox inference. "Continued lifetime" destroys the churn signal outright. |
| Cutoffs | Two explicit: observation 2017-02-28, labelling 2017-01-29 | A single cutoff is correct only if censoring is applied at expiry rather than at the cutoff — correct, but easy to implement wrongly. |
| Backtest eligibility | `MIN_FOLLOWUP_DAYS = 365` | Cohort *size* is not binding (smallest is 32,634). Follow-up is. Precision is event-driven, not subject-driven — hence `MIN_EVENTS_PER_CELL = 100`. |
| Left truncation | Delayed entry via `entry=` | Dropping the truncated cohorts removes the mature-cohort pool the backtest depends on. |
| Backdated cancellations | Spell end from the last non-cancel expiry; event time from the cancel date | 7,232 cancellations backdate expiry before the window opens, the worst by 3,709 days. Any threshold rule leaves a tail of absurd-but-under-threshold rows passing silently. |
| Zero-day rows | Impute, with a flag | 870,124 rows are ordinary ~30-day renewals with unpopulated plan fields, bounded to an eight-month window in 2015 — recoverable from date arithmetic and amount paid. Dropping tears holes in exactly the 2015 cohorts the backtest needs. |

Every imputed row carries `plan_days_imputed`, so any revenue figure is
reproducible with imputation excluded. Excluding them moves LTV midpoints by
**0.5% to 2.5%** — only `121d+` reaches ~2%.

---

## Results

### Survival by plan type

| Stratum | n | Events | Median (d) | 95% CI | S(365) | S(730) |
|---|---:|---:|---:|---|---:|---:|
| `<=7d` | 372,520 | 357,208 | 177 | [176, 178] | 0.180 | 0.053 |
| `8-31d` | 1,338,978 | 720,890 | 447 | [444, 449] | 0.548 | 0.372 |
| `32-120d` | 19,854 | 10,836 | 324 | [317, 334] | 0.467 | 0.283 |
| `121d+` | 142,176 | 100,755 | 576 | [565, 587] | 0.772 | 0.427 |
| *pooled (mixture)* | 1,873,529 | 1,189,690 | *295* | *[294, 296]* | *0.441* | *0.263* |

Under delayed entry the risk set **grows** with *t* — 46.3% of subjects at risk
at t=30, rising to 70.5% at t=730. Early estimates therefore rest on the
least-truncated minority, so number-at-risk is printed alongside every survival
figure in the module output.

*Source: `src/models/km.py`.*

### Reactivation — the finding

**40.85% of churned subjects with full follow-up return within 365 days**
(19.89% within 90d, 38.49% within 180d), median **61 days** to return. Because
the analysis is first-spell-only, this is measured empirically from the excluded
later spells rather than modelled.

| Stratum | Rate @365d | Mean later revenue | Share of LTV |
|---|---:|---:|---:|
| `<=7d` | 8.82% | 975 | 17.6% |
| `8-31d` | 51.87% | 1,562 | **44.4%** |
| `32-120d` | 40.87% | 1,058 | 26.1% |
| `121d+` | 34.80% | 905 | 18.5% |

**44.4% of the value of the largest stratum — 71% of all subjects — lives in
behaviour the survival model deliberately excludes.** Folding reactivation into
a second survival model would have buried this inside a correlated-observations
problem instead of producing a number that can be quoted.

*Source: `src/cohorts/revenue.py`, `src/models/ltv.py`.*

### LTV at 365 days (range, not point)

Reported on **incident subjects** (`entry_days == 0`), where the revenue clock
and the survival clock coincide by construction.

| Stratum | n | LTV range | First spell | Reactivation |
|---|---:|---:|---:|---:|
| `<=7d` | 225,000 | 450 – 528 | 403 | 86 |
| `8-31d` | 571,544 | 1,732 – 1,919 | 1,015 | 811 |
| `32-120d` | 13,623 | 1,642 – 1,671 | 1,224 | 433 |
| `121d+` | 34,557 | 1,694 – 1,706 | 1,385 | 315 |

`LTV(H) = revenue per subscribed day × RMST(H) + reactivation`. Ranges are
bounded by the via-3 entry-time sensitivity and the exclusion bracket.

**730-day LTV is not reported.** The reactivation term requires churners with a
full 730 days of post-churn follow-up, which needs `first_end_date` on or before
2015-02-28; too few qualify, and `32-120d` returns exactly zero. Quoting a
730-day LTV would ship a silently broken component. S(730) appears in the
survival table above, where it is sound.

*Source: `src/models/ltv.py`.*

### Backtest

Fit window ends 2016-12-31, implemented as a fit-window parameter rather than a
global cutoff so no other analysis loses follow-up.

| Horizon | Cells | MAE | Signed bias |
|---|---:|---:|---:|
| 365d (primary) | 47 | 0.0781 | **+0.0374** |
| 180d (secondary) | 73 | 0.0943 | **+0.0504** |

**The bias is positive at both horizons — the model predicts more survival than
occurred, in the direction predicted before the backtest was run.**

By stratum at 365d, three are optimistic and one pessimistic:

| Stratum | Cells | MAE | Bias |
|---|---:|---:|---:|
| `<=7d` | 8 | 0.0850 | +0.0427 |
| `8-31d` | 13 | 0.0751 | +0.0560 |
| `32-120d` | 13 | 0.1040 | **+0.1040** |
| `121d+` | 13 | 0.0510 | **−0.0510** |

`32-120d` has MAE exactly equal to its bias — **all 13 cells err in the same
direction**, which is systematic misprediction rather than noise. `121d+` is the
only under-predicted stratum, and is near-perfect at 180d (MAE 0.0089).

**Truncation does not drive forecast error.** Pearson r = **+0.045** at 365d and
**−0.015** at 180d, with a non-monotonic quartile breakdown — the most-truncated
quartile is middling at 365d and the *best* at 180d.

This was the project's largest documented risk, and the backtest finds no
evidence for it. That does not vindicate the entry times — the via-3 sensitivity
still moves S(730) by 2.93pp — but it does locate the optimism elsewhere, most
plausibly in the excluded fast-churning population, which the backtest cannot
see because those subjects are absent from both forecast and realised.

*Source: `src/backtest/evaluate.py`.*

---

## Where this breaks

- **Survival estimates are upper bounds.** Both remaining biases point the same
  way; the bracket above is the honest range.
- **The revenue and survival clocks are offset, and it matters.** Revenue
  accrues from the first transaction; the survival clock starts at registration.
  Restricting to incident subjects removes the offset entirely, and doing so
  moves LTV by **−18.8% (`<=7d`), −4.3% (`8-31d`), −1.1% (`32-120d`) and −0.7%
  (`121d+`)** at 365 days. The direction is consistent but the magnitude is not
  — a 27-fold spread across strata — so incident-only is the headline and the
  full-population figure is kept only as a sensitivity row.
- **Delayed entry assumes subjects were event-free before entry.** A truncated
  subject may have churned and returned before the window opened. Unfalsifiable
  with this data.
- **Promo cohorts distort short-horizon forecasts.** The worst single cell is
  2016-06 `<=7d`: forecast 0.491 against realised 0.911.
- **Reactivation is measured, not modelled.** It has no covariates and no
  uncertainty band.
- **Same-day ordering is arbitrary.** When a renewal and a cancellation share a
  date, the spell boundary depends on a tie-break the data does not justify.

---

## Reproducing

A fresh clone gets from two CSVs to full results with one command.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Place transactions.csv and members_v3.csv in data/raw/. Both come from the
# ORIGINAL KKBox competition files, not churn_comp_refresh.7z -- see the audit
# section above for why that distinction decides the project.
#   https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data

python run_all.py
```

`run_all.py` checks the raw data is present, then runs ingestion → marts →
Kaplan-Meier → LTV → backtest in order, stopping at the first stage that
errors rather than continuing on a half-built intermediate. It finishes by
re-deriving three headline figures from this run — total subjects, pooled
median survival, and the primary backtest MAE — and comparing them against
what is published above, so the README cannot quietly drift from the code.

**About 1 minute 45 seconds on the full 21.5M-row dataset** (measured:
ingestion 25s, marts 50s, Kaplan-Meier 4s, LTV 14s, backtest 8s).

Individual stages, if you want to run one in isolation:

```bash
python notebooks/step1_audit.py          # the audit that justified the dataset
python -m src.ingest.stage               # raw -> staged (DuckDB)
python -m src.cohorts.spells             # staged -> marts (spell table)
python -m src.cohorts.excluded_spells    # bracketing population
python -m src.cohorts.revenue            # revenue + reactivation
python -m src.models.km                  # Kaplan-Meier
python -m src.models.ltv                 # LTV
python -m src.backtest.evaluate          # backtest
```

Python 3.11 · DuckDB · polars · lifelines · pytest · ruff

**57 tests.** Three consecutive rebuilds verified byte-identical.
