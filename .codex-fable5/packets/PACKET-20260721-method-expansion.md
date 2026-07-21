# Fable5 audit packet: crypto method expansion

Date: 2026-07-21 (Asia/Seoul)
Workspace: `strategy-arena-crypto` checkout under the user's Desktop workspace
Research branch at start: `wave-carry-research-20260720`

## Objective and boundary

The existing crypto waves had no $100-scale promotion candidate, so this run expanded the search rather than stopping. No live orders, exchange credentials, network fetches, or deployment code were added. The private KRA-EV checkout was not modified.

- Wave-8: 16 alternatives covering short-horizon reversal, volatility management, volume shocks, and funding structure.
- Wave-9: 16 additional families covering time-series trend/breakout, residual/pair relative value, candle/range structure, and drawdown/correlation filters.
- Wave-10: six fixed ensembles plus a lagged drawdown throttle; M10a is explicitly registered in the Wave-10 SPEC.

## Data and promotion contract

- Fixed universe: BTC, ETH, BNB, ADA, XRP, DOGE, SOL, AVAX, DOT, LINK, LTC, BCH.
- Common daily cache: 2020-09-23 through 2026-07-14 UTC, 2,121 rows, 12-symbol OHLCV/funding contract.
- Internal OOS split: 2025-10-01 UTC.
- Initial capital $100; $10 reserve; maximum gross 0.60; minimum order $5 per leg.
- Base cost: 0.06% taker fee + 0.03% slippage per unit turnover; stress slippage 0.06%.
- Gates: MC 10,000 paths, 90-day block-MDD 1,000 paths, historical MDD <=25%, ruin <5%, MC p05 >$100, OOS Sharpe >=1, positive stress and blocks, DSR >=95%.
- All three waves record `selection_independent=false`; they are retrospective research, not a prospective live-selection claim.
- Scope: the 38 candidates vary signals and portfolio construction within this fixed daily cache. Order-book/depth, cross-exchange basis, and a genuinely unseen forward window remain separate future research lanes.

## Results

- Wave-8: `eligible=[]`. Best V8d: OOS +25.59%, Sharpe 3.02, historical MDD 66.04%, block MDD p95 62.92%, MC p05 $36.95, ruin 13.75%.
- Wave-9: `eligible=[]`. Best D9c: OOS +16.69%, Sharpe 1.65, stress +16.43%, historical MDD 25.48%, block MDD p95 32.02%, MC p05 $94.47.
- Wave-10: `eligible=[]`. Closest E10a: OOS +9.38%, Sharpe 2.08, stress +8.95%, historical MDD 16.97%, block MDD p95 26.35%, MC p05 $90.02, ruin 0.01%, min order $5.00.

## Verification

- Validators: Wave-8 16 candidates, Wave-9 16, Wave-10 6; all PASS with `eligible=[]`.
- Scoped tests: 13 passed (Wave-8 7, Wave-9 4, Wave-10 2); scoped compileall exit 0.
- Determinism: all three full reruns byte-identical; valid `--only` candidate JSON matched full-run JSON; invalid IDs exited 2.
- Broader suite excluding `research/wave1-rwa`: 87 passed, 1 skipped, plus two pre-existing Wave-4 failures caused by missing `research/wave1/cache/universe.json`.
- Aggregate/report/manifest hashes are recorded in each wave's `report/*_manifest.json`; no external Fable5 verdict is claimed.

## Decision

Promotion and live $100 allocation remain **FAIL CLOSED**: zero candidates pass all gates. The next valid search is a separately registered new-data lane (orderbook/depth, cross-exchange basis, or prospective unseen window), not further in-sample parameter tuning or a live deposit.
