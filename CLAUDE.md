# CLAUDE.md

## What this project is

A subscription retention and lifetime-value forecasting system built on the KKBox
Churn Prediction dataset. The pipeline ingests raw subscription transactions,
validates them at every stage boundary, constructs signup cohorts, fits survival
models for churn, forecasts LTV, and backtests those forecasts against held-out
mature cohorts.

This is a portfolio project for a graduate data scientist application. The
differentiator is **rigour, not feature count**: a validated pipeline with a
defensible backtest beats a dashboard with more charts.

## Non-negotiable rules

1. **Every stage boundary has validation.** raw→staged and staged→marts each run
   checks and write results to `data/validation_log.jsonl`. No stage completes
   silently.
2. **No silent data loss.** Never drop NaNs, duplicates, or outliers without
   logging the count and the reason. If rows disappear between stages, the
   reconciliation check must catch it.
3. **Every modelling choice gets a docstring** stating the assumption it makes
   and what in this dataset could violate it. These docstrings become the README.
4. **Ask before assuming on the open decisions below.** Do not silently pick a
   convention.
5. **Notebooks are exploration only.** Nothing in `notebooks/` is a deliverable.
   Anything that matters gets promoted into `src/` with a test.
6. **No invented numbers.** Every figure in the README traces to code that
   produced it.

## Open decisions — ask, do not assume

These are genuinely ambiguous and change the results materially. Surface them,
explain the trade-off, and let me decide:

- **Resubscribers.** A user who churns and later returns: new subject with fresh
  survival clock, or continued lifetime? (Precedent: treat as new customer,
  carrying covariates that describe prior subscription history.)
- **Observation window and cutoff.** Which date ends the calibration period, and
  which cohorts are mature enough to backtest against.
- **Censoring.** Subscriptions still active at cutoff are right-censored. Confirm
  the censoring rate before any survival model is fitted.
- **Churn definition.** Default is the dataset's own: no valid new subscription
  within 30 days of expiry. Do not redefine it without flagging.

## Stack

- Python 3.11
- **DuckDB** for the staging and mart layers — the transactions file is ~21.5M
  rows, so process it in DuckDB rather than loading into pandas
- **polars** for chunked ingestion, **pandas** where a library requires it
- **lifelines** for Kaplan-Meier and Cox proportional hazards
- **streamlit** for the front end (build last)
- **pytest** for tests, **ruff** for linting

Do not suggest alternative stacks. This is settled.

## Layout

```
data/raw/       # untouched source CSVs, never written to
data/staged/    # typed, deduplicated, validated (DuckDB)
data/marts/     # cohort and survival tables ready for modelling
src/ingest/     # chunked readers, schema enforcement
src/validate/   # validation rules and the check runner
src/cohorts/    # cohort construction, censoring, survival table build
src/models/     # Kaplan-Meier, Cox, LTV
src/backtest/   # held-out cohort forecasting and error metrics
app/            # streamlit UI (last phase)
tests/          # pytest
docs/           # decisions log, methodology notes
```

## Build phases — do not skip ahead

Work one phase per session. Commit between phases.

1. **Ingest + stage.** Chunked reads of `transactions` and `members` into DuckDB
   with an enforced schema. Skip `user_logs` entirely for v1.
2. **Validation layer.** Tests written before implementation.
3. **Cohort construction.** Slowest phase by design — this is where the project
   silently goes wrong. Resolve the open decisions here.
4. **Survival models.** Kaplan-Meier baseline first. Do not jump to Cox.
5. **LTV forecast + backtest.** Hold out mature cohorts, forecast, report error.
6. **Streamlit app + README.**

## How I want you to work

- On any statistical choice, explain the alternative you rejected and why.
- For `src/validate/` and `src/backtest/`, write the test first, then the
  implementation.
- Prefer boring, readable code over clever code. I have to defend this line by
  line in a technical interview.
- Keep functions small enough that I can explain any one of them from memory.
