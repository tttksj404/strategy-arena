# PB104 — 9-point 검증 결과

**상태: NOT READY (4/9)**

_2026-04-28 16:46:48 기준 / 30일 백테스트 / 자체 fetch_

## Overall

| 지표 | 값 |
|---|---:|
| n | 91 |
| win_rate | 0.0989 |
| avg_net_bps | -35.6146 |
| median_net_bps | -35.3734 |
| stdev_net_bps | 36.8598 |
| tp_rate | 0.0220 |
| sl_rate | 0.0000 |
| to_rate | 0.9780 |
| liq_rate | 0.0000 |
| avg_bars_held | 5.9231 |
| total_pnl_usd | -26.7377 |
| longs | 41 |
| shorts | 50 |

## Per-Symbol

| Symbol | Signals | Trades | WR | avg_net_bps | liq_rate |
|---|---:|---:|---:|---:|---:|
| BTCUSDT | 35 | 35 | 5.71% | -37.6 | 0.00% |
| ETHUSDT | 32 | 32 | 9.38% | -36.5 | 0.00% |
| SOLUSDT | 24 | 24 | 16.67% | -31.6 | 0.00% |

## Cost sweep (round-trip bps)

| cost | avg_net_bps |
|---:|---:|
| 16 | -35.61 |
| 20 | -43.61 |
| 25 | -53.61 |
| 30 | -63.61 |
| 40 | -83.61 |

## Subperiod (30일을 3등분)

| Period | n | avg_net_bps | ann_on_margin |
|---|---:|---:|---:|
| P1 | 31 | -47.65 | -2580.43% |
| P2 | 31 | -31.24 | -2035.37% |
| P3 | 29 | -27.42 | -1491.03% |

## 9-point Checks

| # | Check | Result |
|---:|---|---|
| 1 | 시기별 trade-level avg net > 0 | FAIL |
| 2 | 시기별 portfolio 연환산 > 0 | FAIL |
| 3 | avg_net ≥ +50 bps | FAIL |
| 4 | WR ≥ 65% | FAIL |
| 5 | 6축 정합성 (≥3건/일 + 양방향) | PASS |
| 6 | cost 30bps 까지 양수 | FAIL |
| 7 | 5x liquidation < 10% | PASS |
| 8 | n ≥ 50 | PASS |
| 9 | warmup leakage 없음 | PASS |

**Pass: 4/9**

**Trades/day: 3.15**


## 한계 / 위험

- 실제 청산 체결(forceOrders)은 Binance public 미제공 → 포지션·테이커 프록시
- 30일 단일 regime → robustness 부족
- Bitget 진입 시 Binance 신호 → Bitget 가격 latency·slippage 미반영
- TP/SL 동시 터치 시 보수적 SL 가정 → 5m bar 내 정확한 순서 미관측
- DCA 미구현 (원 Hummingbot 룰의 핵심 요소)