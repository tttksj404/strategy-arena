# Wave-6 report

New information sources at 1h resolution: Binance BTC/ETH/SOL funding-window timing, an intraday equity-open spillover, a weekend seasonal drift, a stock-token/underlying deviation fade, and a Bitget new-listing effect. IS ends 2025-09-30, OOS begins 2025-10-01 (wave-1 split, inherited). Intraday candidates carry 2x base slippage per the wave-6 preamble.

## Known proxy / approximation limitations (declared in SPEC.md, restated here)

- **W6a/W6b**: no historical predicted-funding series exists, so the entry signal uses the funding rate realized at the *prior* settlement as a proxy for the upcoming payment.
- **W6c**: hourly bars do not align to the pre-registered 12:30/13:30 UTC boundaries; the signal is approximated by the [12:00,13:00) bar's own return and the entry by the 13:00 bar's open.
- **W6f**: see its dedicated section below if UNDETERMINED — Bitget's `launchTime` field is empty across the entire contracts payload (verified against both the cached wave-3 snapshot and a live refetch).

## Standard candidates (19-gate table; gates 2, 3, 4, 5, 7, 9, 16 emphasized)

| Candidate | Verdict | G2 | G3 | G4 | G5 | G7 | G9 | G16 | OOS Sharpe | OOS trades |
|---|---|---|---|---|---|---|---|---|---:|---:|
| W6a | FAIL | FAIL | FAIL | PASS | UNDETERMINED | PASS | PASS | UNDETERMINED | 0.1624 | 8.0 |
| W6b | UNTESTED_IN_OOS | UNTESTED_IN_OOS | FAIL | UNTESTED_IN_OOS | UNTESTED_IN_OOS | UNTESTED_IN_OOS | UNTESTED_IN_OOS | UNTESTED_IN_OOS | 0.0000 | 0.0 |
| W6c | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | -2.4688 | 209.0 |
| W6d | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS | -0.2859 | 42.0 |

## Exploratory candidates (effect stats only; no deployment claim)

| Candidate | Verdict | Direction | t-stat | Cost-after mean | Sample |
|---|---|---|---:|---:|---:|
| W6e | EFFECT_NEGATIVE_OR_ZERO_COST_AFTER | negative | -2.8717 | -0.000421 | 1336 |
| W6f | UNDETERMINED | - | None | None | 0 |
  - W6f reason: Bitget contracts payload has no populated launchTime for any of the 702 listed usdt-futures symbols (checked against research/wave3/cache/bitget_contracts.json and a live /api/v2/mix/market/contracts refetch); the pre-registered listing-effect test cannot be evaluated.

## W2c combination check

No standard candidate passed all 19 gates, so the W2c combination step (SPEC.md: "생존자는 W2c와 결합 게이트 추가 판정") did not run. This is not a failed test -- there was no survivor to combine.

## Verdict

신규 정보원에서도 보완 없음 — 표준 4후보(W6a-d) 중 PASS 0건. FAIL 확정 3건(W6a, W6c, W6d); 판정 보류 1건(W6b — OOS 구간에 트리거 0회, 데이터 부족으로 UNTESTED이지 FAIL 확정 아님). 다음 발굴 축: 전진 수집 데이터(호가·심도).
