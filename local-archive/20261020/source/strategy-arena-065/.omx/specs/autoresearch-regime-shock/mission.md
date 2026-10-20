# Wave 34 regime-shock research mission

## Objective

Continue the short-horizon paper-only search without reusing the already opened
Wave 33 test interval as an acceptance holdout. Maximize robust pre-cutoff
performance under explicit funding, abrupt price shocks, volatility expansion,
fee spikes, and delayed execution.

## Search boundary

- Candidate generation and ranking may read only timestamps at or before
  `2025-09-30T23:59:59Z`.
- The interval after that cutoff is contaminated diagnostic evidence only. It
  must be opened after `FREEZE`, must never trigger proposal or reselection,
  and cannot support an `accepted` verdict.
- The run is paper research only: no account credentials, orders, exchange
  writes, or capital deployment.

## Required graph

`SEED -> TEMPORAL_CV -> SELECT -> PROPOSE -> ... -> STRESS -> FREEZE -> AUDIT -> REPORT`

## Completion contract

- Candidate count and every scenario evaluation are bounded and recorded.
- Six chronological pre-cutoff folds feed temporal cross-validation.
- A walk-forward replay selects each fold using earlier folds only.
- Historical BTCUSDT perpetual funding is charged with signed direction and
  fixed-position notional; a separate adverse-funding scenario is disclosed.
- Stress covers at least 2x and 3x costs, one-hour and two-hour extra execution
  delay, 12%, 20%, and 40% adverse entry shocks, and doubled volatility with
  doubled costs. The 40% tail is a rounded double of the pre-cutoff worst hourly
  BTCUSDT move and is deliberately more severe than the observed one-hour tail.
- Every intrabar low/high and synthetic entry shock must feed a separate risk
  drawdown path even if price recovers before the daily close.
- Financial risk gates reject liquidation, drawdown below -50%, trade ROI below
  -50%, insufficient trades, weak temporal breadth, and missing funding coverage.
- A fresh unseen holdout is mandatory for any future acceptance. This run must
  remain `research_only` even if every robustness gate passes.
- The persisted result binds current code and data fingerprints and passes an
  independent replay validator.

## Forbidden

- Test/audit metrics in candidate score, proposal, stress finalist selection,
  or stopping logic.
- Changing thresholds after reading audit output.
- Hiding failed scenarios behind averages.
- Claims of guaranteed or expected future profit from backtest output.

## Version note

Wave 34 v1 exhausted the search budget but exposed that recovered synthetic
entry shocks were absent from daily-close drawdown. Its exact artifact is
preserved as `run-v1-pre-intrabar.json`. Version 2 repairs that measurement and
adds the 40% tail. Because the diagnostic interval was already opened, v2 is a
new contaminated exploratory iteration and remains ineligible for acceptance.

Version 2 then showed a second architectural blind spot: only the twelve best
base-objective rows entered stress, while lower-leverage candidates that passed
the frozen stress gates were omitted. Version 3 therefore evaluates 72
finalists, split equally between primary base-objective rank and pre-cutoff
path-safety rank. Stress and audit results never participate in that split.

Version 3 exposed that the past-only walk-forward selector had no admission
risk gate and therefore chose 3x candidates before losing folds. Version 4
admits a candidate at each historical step only when its leverage-bounded 40%
shock loss is at most 50%, its observed past drawdown and worst trade are within
the same bound, and it has at least one past trade per observed fold. Synthetic
shock gaps also fill stops at the gap price and liquidate before a stop when the
gap crosses the liquidation boundary. Prior v1-v3 search and meta-policy trials
are included in the cumulative trial count.
