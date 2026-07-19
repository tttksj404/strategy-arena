# Wave1 RWA final evidence packet

## Scope

Bitget USDT-M RWA (`isRwa=YES`) collection and paper-fidelity sweep in this
workspace only.

## Evidence

- `out/data_manifest.json`: 233 contracts; tiers A=47, B=39, C=147; 233 files
  each under `data/candles_1h`, `data/candles_1d`, and `data/funding`.
- `python3 -m pytest -q`: 5 passed.
- `python3 -m compileall -q src tests`: passed.
- `out/leaderboard.csv` and `.json`: 3,847 rows, 86 A/B symbols, 7 strategy
  families; 5 test-gate passes.
- Selected `OXYUSDT/B1_donchian/L2` was reproduced from its source parquet with
  the reported `test_net_return=8.2189%`.
- `out/REPORT.md`: coverage, gates, $300 scenario, B0 comparison, N=3,847,
  and sample-length/tier-B limitations.

## Required audit questions

1. Confirm next-open execution and no look-ahead.
2. Confirm cost/funding/liquidation math and funding coverage handling.
3. Confirm train-only leverage selection, one-shot test evaluation, and gates.
4. Confirm report and leaderboard claims match artifacts.

## Local verdict

SELF-VERIFIED; independent Fable5 verdict: PENDING (no direct Fable5 provider
route is configured in this Codex surface).
