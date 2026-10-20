# Wave 33 short-profit graph harness mission

## Objective

Build a deterministic, bounded graph-loop harness that searches only short-horizon paper strategies, maximizes validation performance subject to risk gates, and opens the sealed test interval only after a single candidate is frozen.

## Required graph

`SEED -> EVALUATE -> GATE -> SELECT -> PROPOSE -> ... -> STOP -> TEST -> REPORT`

## Completion contract

- Search candidates have a maximum holding period of seven days.
- Candidate evaluations are bounded by the configured unique-candidate budget.
- Search and selection use train and validation windows only.
- The sealed test window is referenced only after `STOP`, through exactly one `TEST` graph event.
- Results include fidelity, fees, 1.5x cost stress, drawdown, liquidations, trade count, holding period, two baselines, stability, cumulative trial count, and deflated Sharpe evidence.
- Financial approval requires every declared gate. A rejected strategy is a valid research result; pipeline integrity and profitability are separate verdicts.
- A machine-readable artifact passes `validate_wave33.py` and the run is reproducible with the same code, data, seed, and budget.

## Forbidden

- Live orders, account credentials, exchange secrets, or capital deployment.
- Test-window metrics in proposal, search, gate, or selection state.
- Increasing the budget or changing thresholds after seeing test results.
- Claims of future profit from backtest output.
