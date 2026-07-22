# Wave-9 리포트 -- $100 네이티브 단일레그 퍼프 (고변동 티어)

사전등록: research/wave9_100usd/SPEC.md. 배경: wave-8은 $300 기준 후보(캐리/모멘텀)가 $100에서 실행 불가함을 확인했다 (캐리 gross 2x, 모멘텀 레그 $0.10 < 최소주문 $5). 이 wave는 처음부터 $100·단일레그 퍼프·동시 포지션 <=2 제약 안에서 설계된 6개 후보를 캐시(research/wave3/cache, Binance USDT-M, crypto-only)만으로 백테스트한다. 목적함수는 MC 중앙값 최종자본 최대화 (샤프/Calmar는 참고 지표).

## 후보 개요

| Candidate | 정의 | Mode | Lev | Hold | Trades | 최종자본 | 총수익률 |
|---|---|---|---:|---:|---:|---:|---:|
| W9a | 집중 모멘텀 롱온리: W3c 랭킹(30d) top-1, 활성자본 100%, 주간 리밸런스, 1x | momentum_top1_long | 1x | 7d | 244 | $5.29 | -94.71% |
| W9b | W9a + 2x 레버리지 | momentum_top1_long | 2x | 7d | 16 | $0.00 | -100.00% |
| W9c | top-1 롱 + bottom-1 숏 (각 활성자본 50% = $45), 주간, 1x | momentum_top1_bottom1 | 1x | 7d | 106 | $10.77 | -89.23% |
| W9d | 초단기 모멘텀: 7d 수익률 top-1, 3일 보유, 1x | momentum_top1_long | 1x | 3d | 610 | $823.23 | 723.23% |
| W9e | 단일레그 펀딩 하베스트: 7d 펀딩 APR>30% 심볼 퍼프 숏 (헤지 없음), 3일 보유, 1x | funding_short | 1x | 3d | 135 | $0.00 | -100.00% |
| W9f | 변동성 돌파: BTC/ETH/SOL 중 전일 ATR(14) 대비 종가 돌파 심볼 롱, 1일 보유, 2x | vol_breakout_long | 2x | 1d | 74 | $0.42 | -99.58% |

## 심층검증 (MC 트레이드부트스트랩 1e4 / 블록셔플 90일 1e3 / OOS 2025-10~ / H1-H5)

| Candidate | MC 중앙값 | MC p05 | P(<$30) | Block MDD p95 | OOS 수익 | Sharpe(참고) | Gates | Overall | Hard(H1/H2/H4) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| W9a | $5.10 | $0.03 | 70.84% | 99.79% | no OOS trades (entry after 2025-10-01) | 0.3444 | 1/5 | FAIL | FAIL |
| W9b | $0.00 | $0.00 | 65.12% | 100.00% | no OOS trades (entry after 2025-10-01) | -0.9878 | 1/5 | FAIL | FAIL |
| W9c | $12.25 | $0.26 | 67.49% | 98.25% | no OOS trades (entry after 2025-10-01) | -0.2048 | 1/5 | FAIL | FAIL |
| W9d | $733.70 | $0.63 | 22.28% | 99.89% | oos_compounded_return=-0.9831 (n_oos_trades=71) | 1.1135 | 1/5 | FAIL | FAIL |
| W9e | $0.00 | $0.00 | 92.12% | 100.00% | no OOS trades (entry after 2025-10-01) | -1.3865 | 1/5 | FAIL | FAIL |
| W9f | $0.56 | $0.00 | 86.36% | 99.80% | no OOS trades (entry after 2025-10-01) | -3.5686 | 1/5 | FAIL | FAIL |

### 게이트별 상세

| Candidate | Gate | Status | Detail |
|---|---|---|---|
| W9a | H1 bankruptcy_probability | FAIL | P(final<$30)=0.7084 (must be <0.20); n_trades=244 |
| W9a | H2 mc_p05_floor | FAIL | p05=$0.03 (must be >$50) |
| W9a | H3 oos_return_positive | UNDETERMINED | no OOS trades (entry after 2025-10-01) |
| W9a | H4 capital_feasibility | PASS | min_notional=$90.00 (>=$5); gross=0.900x equity (<= active*leverage=0.900x); single_leg=True; leverage=1x (<=3x); infeasible_cycles_during_run=61 |
| W9a | H5 block_shuffle_mdd_p95 | FAIL | mdd_p95=0.9979 (must be <=0.50); blocks=22; paths=1000 |
| W9b | H1 bankruptcy_probability | FAIL | P(final<$30)=0.6512 (must be <0.20); n_trades=16 |
| W9b | H2 mc_p05_floor | FAIL | p05=$0.00 (must be >$50) |
| W9b | H3 oos_return_positive | UNDETERMINED | no OOS trades (entry after 2025-10-01) |
| W9b | H4 capital_feasibility | PASS | min_notional=$180.00 (>=$5); gross=1.800x equity (<= active*leverage=1.800x); single_leg=True; leverage=2x (<=3x); infeasible_cycles_during_run=0 |
| W9b | H5 block_shuffle_mdd_p95 | FAIL | mdd_p95=1.0000 (must be <=0.50); blocks=2; paths=1000 |
| W9c | H1 bankruptcy_probability | FAIL | P(final<$30)=0.6749 (must be <0.20); n_trades=106 |
| W9c | H2 mc_p05_floor | FAIL | p05=$0.26 (must be >$50) |
| W9c | H3 oos_return_positive | UNDETERMINED | no OOS trades (entry after 2025-10-01) |
| W9c | H4 capital_feasibility | PASS | min_notional=$45.00 (>=$5); gross=0.900x equity (<= active*leverage=0.900x); single_leg=True; leverage=1x (<=3x); infeasible_cycles_during_run=378 |
| W9c | H5 block_shuffle_mdd_p95 | FAIL | mdd_p95=0.9825 (must be <=0.50); blocks=10; paths=1000 |
| W9d | H1 bankruptcy_probability | FAIL | P(final<$30)=0.2228 (must be <0.20); n_trades=610 |
| W9d | H2 mc_p05_floor | FAIL | p05=$0.63 (must be >$50) |
| W9d | H3 oos_return_positive | FAIL | oos_compounded_return=-0.9831 (n_oos_trades=71) |
| W9d | H4 capital_feasibility | PASS | min_notional=$90.00 (>=$5); gross=0.900x equity (<= active*leverage=0.900x); single_leg=True; leverage=1x (<=3x); infeasible_cycles_during_run=0 |
| W9d | H5 block_shuffle_mdd_p95 | FAIL | mdd_p95=0.9989 (must be <=0.50); blocks=28; paths=1000 |
| W9e | H1 bankruptcy_probability | FAIL | P(final<$30)=0.9212 (must be <0.20); n_trades=135 |
| W9e | H2 mc_p05_floor | FAIL | p05=$0.00 (must be >$50) |
| W9e | H3 oos_return_positive | UNDETERMINED | no OOS trades (entry after 2025-10-01) |
| W9e | H4 capital_feasibility | PASS | min_notional=$90.00 (>=$5); gross=0.900x equity (<= active*leverage=0.900x); single_leg=True; leverage=1x (<=3x); infeasible_cycles_during_run=0 |
| W9e | H5 block_shuffle_mdd_p95 | FAIL | mdd_p95=1.0000 (must be <=0.50); blocks=11; paths=1000 |
| W9f | H1 bankruptcy_probability | FAIL | P(final<$30)=0.8636 (must be <0.20); n_trades=74 |
| W9f | H2 mc_p05_floor | FAIL | p05=$0.00 (must be >$50) |
| W9f | H3 oos_return_positive | UNDETERMINED | no OOS trades (entry after 2025-10-01) |
| W9f | H4 capital_feasibility | PASS | min_notional=$180.00 (>=$5); gross=1.800x equity (<= active*leverage=1.800x); single_leg=True; leverage=2x (<=3x); infeasible_cycles_during_run=189 |
| W9f | H5 block_shuffle_mdd_p95 | FAIL | mdd_p95=0.9980 (must be <=0.50); blocks=7; paths=1000 |

## 자본 현실성 (H4: 총자본 $100, 현금버퍼 10% -> 활성자본 $90, 최소주문 $5)

| Candidate | 동시포지션 | 레버리지 | 시작 최소노셔널 | gross(활성자본 배수) | 단일레그 | H4 |
|---|---:|---:|---:|---:|---|---|
| W9a | 1 | 1x | $90.00 | 0.90x | True | PASS |
| W9b | 1 | 2x | $180.00 | 1.80x | True | PASS |
| W9c | 2 | 1x | $45.00 | 0.90x | True | PASS |
| W9d | 1 | 1x | $90.00 | 0.90x | True | PASS |
| W9e | 1 | 1x | $90.00 | 0.90x | True | PASS |
| W9f | 1 | 2x | $180.00 | 1.80x | True | PASS |

## 판정

**전멸: 6개 후보 모두 H1-H5 게이트 중 하나 이상 미달. $100·단기·고수익 조건에서 검증된 엣지 없음.**

후보별 실패 원인 (엣지 부재 / 비용 초과 / 청산 발생 중 구분):

| Candidate | 실패 원인 |
|---|---|
| W9a | 엣지 부재 (비용 차감 전 gross P&L=$-31.48 <= 0; 신호 자체가 우위 없음) |
| W9b | 청산 발생 (거래 16건 중 청산 1건, 이벤트 1회; gross P&L=$-106.56, net P&L=$-110.20) |
| W9c | 엣지 부재 (비용 차감 전 gross P&L=$-75.07 <= 0; 신호 자체가 우위 없음) |
| W9d | 거래 자체는 순이익(net P&L=$723.23)이지만 MC/블록셔플/OOS 게이트 중 하나 이상 미달 |
| W9e | 엣지 부재 (비용 차감 전 gross P&L=$-96.34 <= 0; 신호 자체가 우위 없음) |
| W9f | 엣지 부재 (비용 차감 전 gross P&L=$-87.05 <= 0; 신호 자체가 우위 없음) |

## 다중검정 보정 (참고)

SPEC.md: "이 6개가 전부. 사후 파라미터 조정 금지. 기존 52후보와 합산한 시행횟수(58)로 DSR 보정 표기." DSR(deflated Sharpe ratio)은 trials=58로 계산했으며 참고 지표일 뿐 채택 기준이 아니다.

| Candidate | DSR score | DSR probability | trials |
|---|---:|---:|---:|
| W9a | 12.9712 | 1.0000 | 58 |
| W9b | -6.3522 | 0.0000 | 58 |
| W9c | -7.1177 | 0.0000 | 58 |
| W9d | 17.3905 | 1.0000 | 58 |
| W9e | -8.1855 | 0.0000 | 58 |
| W9f | -6.2131 | 0.0000 | 58 |

## 모델링 노트

- 체결: 시그널 바 종가 확정 -> 다음 바 시가 체결, 룩어헤드 없음.
- 비용: 테이커 0.06%/사이드 + 슬리피지(메이저 1bp, 알트 3bp); 메이커 가정 없음.
- 청산: research/wave4_leverage/sweep.py의 liquidation_loss를 그대로 재사용 (유지증거금 0.5%, 청산수수료 0.06%); 진입가 대비 누적 최악역행을 매 보유일마다 점검.
- 펀딩은 전 후보에 동일하게 적용(모든 퍼프 포지션은 실제로 펀딩을 주고받음); W9e는 방향손익과 펀딩수취가 모두 반영된다 (헤지 없음).
- 유니버스: research/wave3/cache 기반 Binance USDT-M 퍼프, crypto-only (토큰화 주식 제외). SPEC.md의 'Bitget USDT-M'은 목표 실거래 venue의 계약 성격을 서술한 것이며, 이 백테스트의 캐시 데이터 소스는 아니다 (Bitget 크립토 퍼프 OHLC 캐시 자체가 저장소에 없음).
- 비용 보수화: 동일 심볼이 다음 주기에도 재선정되더라도 매 주기 진입+청산 수수료를 새로 부과한다 (포지션 이월 최적화 없음) -- 실제보다 비용을 과소평가하지 않기 위한 보수적 단순화.
- 매 결과 JSON의 equity 시리즈는 거래 경계(진입/청산)에서만 기록된다 (일별 스무딩 없음); 청산 판정 자체는 보유 중 매일 점검한다.

