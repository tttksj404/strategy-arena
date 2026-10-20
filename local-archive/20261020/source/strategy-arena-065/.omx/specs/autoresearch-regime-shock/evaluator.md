# Wave 34 evaluator contract

## Primary objective

Among candidates that pass frozen risk gates, maximize the worst chronological
fold log-multiple, then the median fold log-multiple. Break ties by lower maximum
drawdown, lower leverage, and candidate id.

## Frozen policy

- Search cutoff: `2025-09-30T23:59:59Z`.
- Diagnostic audit end: last timestamp with complete local funding coverage.
- Maximum hold: 7 days.
- Candidate budget: 720 unique candidates.
- Stress finalists: 72: 36 by primary objective and 36 by pre-cutoff path
  safety, deduplicated before any stress result is read.
- Minimum aggregate pre-cutoff trades: 36.
- Minimum profitable folds: 4 of 6.
- Worst allowed fold multiple: 0.65.
- Maximum drawdown: -0.50.
- Worst trade ROI: -0.50.
- Liquidations: 0.
- Cost stress: 2x and 3x.
- Abrupt entry shock stress: 12%, 20%, and 40%; combined-extreme uses 40%.
- Drawdown is the worse of the close-sampled path and the intrabar risk path.
- Cumulative trials include the previous 439 trials, every new candidate, and
  every finalist-scenario evaluation. Wave 34 v4 starts from 3,548 prior trials
  after carrying forward the v1-v3 searches and one meta-policy revision.
- Walk-forward admission uses past data only, requires observed risk within
  -50%, at least one trade per observed fold, and an analytic 40% shock loss no
  greater than 50% of equity.

## Robustness outputs

- base temporal fold table and aggregate metrics;
- past-only walk-forward replay sequence and chained multiple;
- worst finalist stress scenario and all scenario outcomes;
- actual funding PnL/event count and funding source fingerprint;
- contaminated audit metrics, clearly excluded from selection;
- deflated Sharpe probability with cumulative trial count;
- structural findings and final `research_only` decision reasons.

## Improvement rule

An iteration is an improvement only when it raises the primary objective while
all frozen policy gates remain satisfied. Audit-window gains never count as an
improvement. Stop after the configured candidate budget or three proposal rounds
without a pre-cutoff objective improvement.
