# Time-safe prediction uplift design

## Goal

Increase KRA and KCYCLE prediction accuracy without promoting a candidate that learned from post-start data or from its final holdout.

## Decision

Run separate, race-level lanes rather than one blended score:

1. KRA Top-1: market-prior residual ranker, segment specialists, and calibrated selective fallback.
2. KCYCLE Top-1 and ordered trifecta: validation-selected board ensemble and uncertainty-aware routing, each evaluated independently.
3. Time-series market lane: collect timestamped, pre-start official snapshots; do not use the lane until outcomes are joined after the snapshot cutoff.

The current production market-anchor policies and KCYCLE V1 ensemble remain the baselines. A candidate may only replace one lane after train/validation-only selection, one untouched chronological holdout, positive paired-bootstrap lower bound, and a live shadow window. Abstention may improve the accuracy of accepted predictions only when its coverage is reported and it never removes the baseline display.

## Data and leakage contract

- Every feature has an as-of time. Post-start, stale, outcome-derived, and final-odds-after-cutoff values are excluded.
- Race-level aggregates, normalisation, calibration, specialist routing, and hyperparameters are fit only inside each training fold.
- Discovery uses expanding train/validation folds. The final holdout is read once per frozen candidate set.
- Top-1, unordered pair, ordered pair, and ordered trifecta exact are separate metrics and cannot justify one another.

## Promotion contract

- KRA Top-1 retains the existing absolute-lift, fresh-holdout, paired-bootstrap, and log-loss gates; market is the baseline whenever a fresh official board exists.
- KCYCLE Top-1 and trifecta retain validation-only artifact selection, immutable held-out test reporting, and pre-start board integrity checks.
- Any stale or post-start snapshot, failed calibration, coverage below its preset floor, or a negative worst chronological fold rolls the candidate back to the baseline.

## First implementation slice

Add a reproducible candidate-tournament contract and candidate manifest that record frozen definitions, chronological holdout accounting, calibration, coverage, and separate per-lane metrics. Both are fixture-tested, write only an explicit report path, and never alter production model artifacts. A later promotion remains a distinct change only after fresh evidence meets the contract.
