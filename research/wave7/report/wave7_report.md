# Wave-7 리포트 -- 캐리+모멘텀 결합 포트폴리오

사전등록: research/wave7/SPEC.md. 구성요소는 W2c(캐리, research/wave2/results/W2c.json)와 W3c(모멘텀, research/wave3/results/W3c.json)의 일수익 시리즈만 사용, 신규 신호 탐색 없음.

## 구성 개요 + 결합 자산곡선 지표

| Candidate | Definition | Total Ret | CAGR | Sharpe | MDD | Calmar |
|---|---|---:|---:|---:|---:|---:|
| W7a | 정적 70/30 (캐리/모멘텀), 일 리밸런스 | 127.16% | 12.72% | 2.3637 | 6.68% | 1.9055 |
| W7b | 정적 60/40 (캐리/모멘텀), 일 리밸런스 | 112.01% | 11.59% | 1.7594 | 9.41% | 1.2319 |
| W7c | 레짐 스위치: BTC/ETH 7d 펀딩 APR(평균)>15% -> 캐리100/모멘텀0, 아니면 캐리60/모멘텀40 | 198.59% | 17.31% | 2.7973 | 7.42% | 2.3316 |
| W7d | W7c + 모멘텀 크래시가드: BTC<MA200(시프트) 시 모멘텀 슬리브 현금(0) | 173.90% | 15.84% | 3.3841 | 3.89% | 4.0710 |

## 심층검증 배터리 (MC 1e4 / 블록셔플 90일 1e3 / 휴면기 OOS / Sharpe 비교 / W2c 상관)

| Candidate | MC p05 | Ruin P(<150) | Block MDD p95 | Dormant OOS | Sharpe (combined vs carry-alone) | Corr w/ W2c | Gates | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W7a | $547.93 | 0.00% | 9.69% | 10.23% | 2.3637 vs 4.1969 | 0.4781 | 4/5 | FAIL |
| W7b | $484.11 | 0.00% | 13.75% | 13.59% | 1.7594 vs 4.1969 | 0.3276 | 4/5 | FAIL |
| W7c | $703.66 | 0.00% | 8.64% | 13.59% | 2.7973 vs 4.1969 | 0.5888 | 4/5 | FAIL |
| W7d | $686.28 | 0.00% | 4.63% | 3.22% | 3.3841 vs 4.1969 | 0.7769 | 4/5 | FAIL |

### 게이트별 상세

| Candidate | Gate | Status | Value |
|---|---|---|---|
| W7a | mc_bootstrap_p05 | PASS | p05=547.93 (>300) |
| W7a | bankruptcy_probability | PASS | ruin=0.0000 (<0.05) |
| W7a | block_shuffle_mdd_p95 | PASS | mdd_p95=0.0969 (<=0.25) |
| W7a | dormant_oos_return | PASS | oos=0.1023 (> carry_alone=0.0064) |
| W7a | sharpe_vs_carry_alone | FAIL | sharpe=2.3637 (> carry_alone=4.1969) |
| W7b | mc_bootstrap_p05 | PASS | p05=484.11 (>300) |
| W7b | bankruptcy_probability | PASS | ruin=0.0000 (<0.05) |
| W7b | block_shuffle_mdd_p95 | PASS | mdd_p95=0.1375 (<=0.25) |
| W7b | dormant_oos_return | PASS | oos=0.1359 (> carry_alone=0.0064) |
| W7b | sharpe_vs_carry_alone | FAIL | sharpe=1.7594 (> carry_alone=4.1969) |
| W7c | mc_bootstrap_p05 | PASS | p05=703.66 (>300) |
| W7c | bankruptcy_probability | PASS | ruin=0.0000 (<0.05) |
| W7c | block_shuffle_mdd_p95 | PASS | mdd_p95=0.0864 (<=0.25) |
| W7c | dormant_oos_return | PASS | oos=0.1359 (> carry_alone=0.0064) |
| W7c | sharpe_vs_carry_alone | FAIL | sharpe=2.7973 (> carry_alone=4.1969) |
| W7d | mc_bootstrap_p05 | PASS | p05=686.28 (>300) |
| W7d | bankruptcy_probability | PASS | ruin=0.0000 (<0.05) |
| W7d | block_shuffle_mdd_p95 | PASS | mdd_p95=0.0463 (<=0.25) |
| W7d | dormant_oos_return | PASS | oos=0.0322 (> carry_alone=0.0064) |
| W7d | sharpe_vs_carry_alone | FAIL | sharpe=3.3841 (> carry_alone=4.1969) |

## 자본 현실성 ($300 x 0.9 = $270 동시마진 버퍼, 최소주문 5 USDT)

| Candidate | Max combined weight | Buffer<=90% | Carry min leg | Momentum min leg | Status |
|---|---:|---|---:|---:|---|
| W7a | 1.00 | False | $52.50 | $0.25 | FAIL |
| W7b | 1.00 | False | $45.00 | $0.34 | FAIL |
| W7c | 1.00 | False | $45.00 | $0.34 | FAIL |
| W7d | 1.00 | False | $45.00 | $0.34 | FAIL |

## 판정

**전멸: 4개 구성 모두 심층검증 게이트 미달 -> 캐리 단독(W2c)이 최적.**

개별 구성 수치는 위 표에 사실대로 기록. 상세 사유는 게이트별 상세표 참조.

모멘텀 슬리브 주의: W3c는 게이트 2(overfit_sensitivity)·게이트 3 계열(IS 일관성) 개별 FAIL, 전체 후보군 자체 딥밸리데이션에서 MC p05 $227.99(UNDETERMINED, trade_returns가 daily-active 시맨틱), 블록셔플 MDD p95 42.71%로 기록된 저확신 슬리브다. 위 결합 결과가 통과하더라도 모멘텀은 보조 비중으로만 취급해야 한다 (SPEC.md 핵심 규율).

