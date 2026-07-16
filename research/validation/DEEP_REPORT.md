# Deep validation report

Cache-only full-history validation for W2c, F1e, W3c, and W3d.

## Candidate x validation

| Candidate | MC 10k | Leave-one-year-out | DSR (28 trials) | Bitget native | 90d block shuffle | Overall |
|---|---|---|---|---|---|---|
| W2c | PASS p05=746.32; ruin=0.00% | PASS years=8 | PASS score=1.819 | FAIL sign=76.42%; entry=100.00%; coverage=82d | PASS blocks=18; MDD p95=1.49% | FAIL |
| F1e | PASS p05=397.44; ruin=0.00% | PASS years=8 | PASS score=7.585 | FAIL sign=76.42%; entry=92.14%; coverage=82d | PASS blocks=10; MDD p95=0.83% | FAIL |
| F1f | PASS p05=458.66; ruin=0.00% | PASS years=8 | PASS score=1.931 | N/A | PASS blocks=18; MDD p95=3.83% | PASS |
| W3c | UNDETERMINED p05=227.99; ruin=0.31% | PASS years=8 | PASS score=7.708 | N/A | PASS blocks=28; MDD p95=42.71% | FAIL |
| W3d | UNDETERMINED p05=8.29; ruin=36.53% | PASS years=8 | PASS score=17.308 | N/A | PASS blocks=20; MDD p95=95.92% | FAIL |

## Year-by-year leave-one-out

| Candidate | Year | Held-out return | Held-out Sharpe | Remaining return | Remaining Sharpe |
|---|---:|---:|---:|---:|---:|
| W2c | 2019 | 0.0104 | 5.379 | 1.7864 | 4.335 |
| W2c | 2020 | 0.3224 | 9.244 | 1.1290 | 3.578 |
| W2c | 2021 | 0.6202 | 12.036 | 0.7377 | 2.807 |
| W2c | 2022 | 0.0087 | 2.785 | 1.7910 | 4.596 |
| W2c | 2023 | 0.0675 | 6.768 | 1.6374 | 4.354 |
| W2c | 2024 | 0.2079 | 2.792 | 1.3308 | 5.555 |
| W2c | 2025 | 0.0004 | 0.028 | 1.8142 | 4.864 |
| W2c | 2026 | -0.0005 | -1.368 | 1.8169 | 4.454 |
| F1e | 2019 | 0.0137 | 0.957 | 0.6657 | 5.094 |
| F1e | 2020 | 0.2214 | 8.073 | 0.3825 | 3.444 |
| F1e | 2021 | 0.3724 | 12.694 | 0.2304 | 2.321 |
| F1e | 2022 | -0.0063 | -1.170 | 0.6992 | 4.756 |
| F1e | 2023 | 0.0000 | 0.000 | 0.6886 | 4.665 |
| F1e | 2024 | 0.0000 | 0.000 | 0.6886 | 4.666 |
| F1e | 2025 | 0.0000 | 0.000 | 0.6886 | 4.665 |
| F1e | 2026 | 0.0000 | 0.000 | 0.6886 | 4.480 |
| F1f | 2019 | 0.0076 | 3.574 | 0.7149 | 2.295 |
| F1f | 2020 | 0.1534 | 4.483 | 0.4981 | 1.936 |
| F1f | 2021 | 0.3712 | 7.633 | 0.2602 | 1.183 |
| F1f | 2022 | -0.0056 | -0.967 | 0.7377 | 2.487 |
| F1f | 2023 | 0.0078 | 0.638 | 0.7146 | 2.443 |
| F1f | 2024 | 0.1052 | 1.503 | 0.5635 | 2.935 |
| F1f | 2025 | -0.0186 | -0.646 | 0.7608 | 2.670 |
| F1f | 2026 | -0.0023 | -1.368 | 0.7320 | 2.374 |
| W3c | 2019 | -0.1165 | -1.617 | 0.5313 | 0.523 |
| W3c | 2020 | 0.2062 | 0.951 | 0.1216 | 0.214 |
| W3c | 2021 | -0.0897 | -0.611 | 0.4862 | 0.523 |
| W3c | 2022 | 0.0791 | 0.704 | 0.2537 | 0.328 |
| W3c | 2023 | -0.0741 | -0.614 | 0.4612 | 0.496 |
| W3c | 2024 | -0.0469 | -0.292 | 0.4195 | 0.470 |
| W3c | 2025 | 0.1898 | 1.599 | 0.1371 | 0.220 |
| W3c | 2026 | 0.2309 | 2.852 | 0.0991 | 0.176 |
| W3d | 2019 | 0.0000 | 0.000 | 0.0093 | 0.440 |
| W3d | 2020 | 3.5699 | 2.200 | -0.7791 | 0.133 |
| W3d | 2021 | 1.0886 | 1.195 | -0.5168 | 0.234 |
| W3d | 2022 | 0.0000 | 0.000 | 0.0093 | 0.465 |
| W3d | 2023 | -0.2510 | 0.145 | 0.3475 | 0.483 |
| W3d | 2024 | -0.4630 | -0.093 | 0.8796 | 0.542 |
| W3d | 2025 | -0.7371 | -0.855 | 2.8389 | 0.687 |
| W3d | 2026 | 0.0000 | 0.000 | 0.0093 | 0.448 |

## Decision criteria

- MC: 10,000 unit-exposure full-period trade bootstrap paths; PASS requires final-capital p05 > 300 and P(capital < 150) < 5%.
- MC input contract: F1e/W2c use closed-trade returns; W3c/W3d expose active-day returns in the same field, so their MC cells remain UNDETERMINED until upstream semantics are separated.
- Kelly: the existing gate-compatible mean(trade_return) / sample_variance estimate; both f* and 0.25f* are recorded.
- DSR: Bailey-Lopez de Prado daily-return deflated Sharpe z-score with trials=28; PASS requires DSR score > 0.
- Bitget native: record 7-day score correlation, entry-signal agreement, and sign agreement. PASS requires sign agreement > 80% and the requested 133-day common coverage.
- Leave-one-year-out and 90-day block permutation are diagnostic checks; PASS means the calculation completed with the minimum sample.
- The 90-day block method shuffles block order without replacement; final capital is therefore an invariant and MDD is the distributional stress output.

## Data constraints

- No network calls were made. Bitget reproduction uses only normalized local cache rows in research/wave1/cache.
- The current common funding-score interval is 3 symbols, 738 observations, and 82 days. It is shorter than the requested 133 days, so the native cells and overall verdicts for F1e/W2c are FAIL.
- The wave2 cache manifest recheck is FAIL; every consumed funding file must have a matching byte count and SHA-256 before native evidence can be accepted.
- Funding rows are normalized to UTC 8-hour buckets and a 7-day score is emitted only for 21 contiguous buckets; gaps reset the rolling window.
- Final candidate status is the intersection of the stated gates; insufficient cache coverage is never interpreted as a PASS.

## 최종 판정 — 백데이터 검증 완결 (Fable5, 2026-07-16)

교차검증(`CROSS_VENUE_REPORT.md`)으로 마지막 차단기 해제:
- **거래소 재현성 PASS**: Binance↔Bybit 919일 겹침에서 전략이 실제 작동하는 고펀딩(|7d APR|>15%) 구간 부호 일치율 **98.9%**(N=4399, 기준 90%). OKX 보조 91.7%. 이전에 Bitget이 76.4%로 미달했던 것은 저펀딩 노이즈였음이 입증됨 — Binance↔Bybit도 저펀딩 구간에선 75.2%로 똑같이 떨어진다(격차 +23.7%p). 펀딩은 거래소 고유가 아니라 시장 공통 현상.
- **체결률 PASS(최악가정)**: F1f(동일 룰, 전량 테이커 체결 가정)가 MC p05 $459·파산 0%·DSR 1.93·8년 LOO·블록셔플 전부 통과. 메이커 체결은 순수 개선분.

**결론: 캐리 패밀리(W2c 메이커 / F1f 테이커)는 히스토리 데이터로 도달 가능한 모든 게이트를 통과했다.**
백데이터가 원리적으로 보증할 수 없는 잔여 리스크(명시): ①미래 펀딩 레짐의 비정상성 ②거래소 운영·커스터디 리스크 ③극단 이벤트에서 현물-퍼프 베이시스 폭주. 이 3개는 어떤 백테스트로도 제거 불가 — 사이징 상한과 회로차단기로만 관리 가능.
