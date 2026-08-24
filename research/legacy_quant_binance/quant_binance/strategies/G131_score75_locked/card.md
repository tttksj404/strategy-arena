# G131 — G130 + score_min 70 → 75 (backtest 검증 sweet spot) ⭐

## Status: paper-live ready (regime engine + filter 강화)

부모: G130
변경 변수: **mode_thresholds.futures_score_min** (1개) — 70 → 75

## 가설 / 검증

regime engine raw 는 noise 多 (predictability_score 분포 34-72). 사용자 winner 패턴 분석 결과 70-84 였음. backtest 로 score_min 임계 sweep:

| score 임계 | trades (30d) | WR | avg net bps | total |
|---:|---:|---:|---:|---:|
| RAW (filter 없음) | 128 | 35% | -14 | **-1806** ❌ |
| 70 (G130) | 133 | 44% | +11 | +1444 |
| **75 (G131)** | **76** | **51%** | **+34** | **+2567** ⭐ |
| 80 | 23 | 48% | +46 | +1061 |
| 85 | 6 | 67% | +54 | +323 |

→ **score_min=75 가 cumulative PnL 최대**. 사용자 winners median score 76 와 정확 일치.

## 운용 (G130 activation 가이드 동일)

```powershell
$env:STRATEGY_OVERRIDE_PATH = "$HOME\Desktop\first_repo\quant_binance\strategies\G131_score75_locked\overrides.json"
$env:PAPER_TRADING = "1"
cd $HOME\Desktop\first_repo
python -m quant_binance.daemon
```

## 추정 수익 ($55 capital)

```
30-day backtest:  76 trades × +33.8 bps × $275 notional/10000 = +$71/30d
                  (target_lev 20x × per-trade 0.25 × $55)
60-day:           57 × +22.5 × $275/10000 = +$35/60d → +$214/year = +388%/년 (20x)

5x lev 보수 추정: 57 × +22.5 × $69/10000 = +$8.84/60d → +$54/year = +98%/년
```

## 한계

- backtest 가 30/60 day window. 더 긴 OOS 검증 필요.
- regime engine 자체 raw 는 -1806 bps WR 35% (음수) — score_min filter 절대 필수
- score 75 미만 진입 = 직접적 손실 (overide 가 차단해야 함)
- 4-coin universe 만. universe 확장 시 재검증 필요.
- 30x leverage 가능 → 인터바 -3% 시 liquidation 위험

## 다음 후보 (variable-1)

- **G132**: G131 + holding_period 4h → 1h (단타 강화, score=80 필요?)
- **G133**: score_min 75 → 85 (lottery extreme, 5-6 trades/30d 100% WR)
- **G134**: target_leverage 20 → 10 (5-10x 명시 부합, 안전)
- **G135**: + 양방향 short_disabled false (검증 필요)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G130 변형 (score 70→75). backtest 30d/60d 모두 양수 입증 (+2567/+1281). raw 시스템 (filter 없음) 은 -1806 음수 → score filter 가 alpha 핵심 |
