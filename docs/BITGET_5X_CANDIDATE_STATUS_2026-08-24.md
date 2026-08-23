# Bitget 5x candidate status — 2026-08-24

## Decision

`NO_RELIABLE_WINNER`. This is a research decision: no candidate may be represented as a profitable or liquidation-safe live strategy.

## Existing baseline evidence

The fixed 5x, 30-day, four-contract replay committed in `research/bitget_5x/results/2026-08-23-fixed-5x-replay.json` evaluated the four precommitted breakout-volume thresholds. All had negative post-cost validation return lower bounds, so none entered holdout promotion.

## Added candidate filters

The 2026-08-24 contract records six candidates under one completed-candle, chronological, fixed-cost evaluator:

1. `breakout-volume-1.0`
2. `breakout-volume-1.25`
3. `breakout-volume-1.5`
4. `breakout-volume-2.0`
5. `breakout-technical-score-v1-volume-1.0`
6. `breakout-bollinger-squeeze-v1-volume-1.0`

The technical-score and Bollinger filters are implemented only as causal entry filters. They do not change the shared 5x target, stop, fee, spread, or slippage calculation.

## Fresh limited smoke replay

A public, seven-day, one-contract replay was used only to verify that the new filters run through the same engine. It produced zero technical-score trades and one Bollinger trade; the latter was stop-first after cost. Both are far below the precommitted minimum of 80 validation and 20 later holdout signals. This is insufficient evidence, not a negative final verdict on the two ideas.

## Requirements before any promotion

- Completed candles only.
- Identical fixed-5x cost and loss model for every candidate.
- At least 80 validation signals with positive conservative post-cost return.
- Candidate selection without reading the later holdout.
- At least 20 later holdout signals clearing the same gates.
- Separate forward evidence before any human-authorized demo or real order.
