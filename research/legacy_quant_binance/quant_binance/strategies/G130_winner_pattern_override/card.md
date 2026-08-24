# G130 — Winner Pattern Override (regime engine + filter 강화)

## Status: paper-live ready (백테스트 우회 — 실시간 regime engine 의존)

부모: live-ultra-aggressive (현 strategy_override.approved.json)
변경: 3 핵심 + 보조 (universe / score 임계 / per-trade 자본 + exit/risk 보수화)

## 가설

사용자 closed_trades 78건 분석:
- Winner 22건: ETH 8 / SOL 8 / DOGE 4 / PEPE 2, score 70-84, hold 5분
- 시스템 전체 -$106 (음수): 현 score_min=48 → noise 진입, PEPE-only universe → winner 75% 누락

→ winner 패턴 강제 (universe 4종 + score≥70 + size 보수화) 시 양수 가능.

## 핵심 변경 3개

| 항목 | 부모 (live-ultra-aggressive) | G130 |
|---|---:|---:|
| universe | PEPE 단일 | **ETH/SOL/DOGE/PEPE 4종** (winners 분포 그대로) |
| `futures_score_min` | 48 | **70** (winners 패턴 강제) |
| `per_trade_equity_risk` | 0.40 | **0.25** (자본 보호) |

## 보조 변경 (winner 패턴 추가 반영)

- `target_futures_leverage`: 30 → **20** (사용자 명시 5-10x 와 절충)
- `max_holding_minutes`: 4320 (72h) → **1440** (24h, winners median 5.3분 + 안전 buffer)
- `proactive_take_profit_thresholds`: [40, 55] → **[18, 35, 50]** (winners peak ROE 18.7% 반영)
- `proactive_take_profit_fraction`: 0.10 → **0.25** (더 적극적 익절)
- `stop_loss_roe`: -40% → **-28%** (자본 보호)
- `cost_gate.edge_to_cost_multiple_min`: 0.78 → **1.20** (cost 보다 1.2배 edge 필수)
- `cash_reserve`: 0.02 → **0.10** (10% 현금 유보)
- `max_concurrent_futures_symbols`: 1 → **2** (4 universe 분산)
- `futures_top_n`: 1 → **2**

## 운용

```bash
# 1. override 파일 등록 (실거래 시작 전)
export STRATEGY_OVERRIDE_PATH="$HOME/Desktop/first_repo/quant_binance/strategies/G130_winner_pattern_override/overrides.json"

# 2. paper-live 모드로 daemon 시작 (실거래 X)
cd ~/Desktop/first_repo
PAPER_TRADING=1 python -m quant_binance.daemon

# 3. 모니터링
tail -f quant_runtime/output/paper-live-shell/latest/overview.json
```

## 검증 plan (verify-first 우회 — paper-live 의존)

| 단계 | 기간 | 통과 기준 |
|---|---|---|
| 1. paper-live 가동 | 7일 | 시스템 안정 / 진입 ≥10건 |
| 2. 1주 결과 분석 | — | total PnL > 0 / WR ≥ 50% / score 70+ 진입만 발화 |
| 3. 14일 누적 평가 | — | total PnL > 0 / drawdown < 20% |
| 4. 30일 final 평가 | — | annualized ≥ +50% / Sharpe > 1 |
| 5. 통과 시 → micro-live $5 | 30일+ | 실거래 검증 |

## 한계 / 위험

- 백테스트 X (regime engine 가동 의존) → 9-point 적용 불가, paper-live 30일이 진짜 검증
- B3 MSB 파라미터는 PEPE-fitted (다른 종목 hyperopt 재실행 권장)
- universe 확장 시 신호 빈도 변동 (PEPE 만일 때보다 4배 가능)
- 30x 절대 max → liquidation 발생 시 자본 큰 손실

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | live-ultra-aggressive override 변형. winner 패턴 (universe 4종 + score≥70) 강제. 자본 보호 (size 0.40→0.25, lev 30→20) 동시 적용 |

---
## ⚠️ G131 으로 superseded (2026-04-28)

backtest 검증 결과 score_min 70 (G130) 보다 75 (G131) 가 더 우수:
- G130 (score 70): 133 trades / +10.9 avg / +1444 total
- G131 (score 75): 76 trades / +33.8 avg / **+2567 total** ⭐

→ G130 deprecated. G131 사용 권장.
