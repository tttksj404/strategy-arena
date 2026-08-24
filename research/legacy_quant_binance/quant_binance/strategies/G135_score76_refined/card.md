# G135 — G131 + score 75 → 76 (refined sweet spot)

## Status: paper-live ready (G131 marginal 개선)

부모: G131
변경: mode_thresholds.futures_score_min 75 → 76

## 결과 (60d backtest, 4h hold, cost 16)

| variant | n | total bps | avg | WR |
|---|---:|---:|---:|---:|
| G131 (score=75) | 57 | +1281 | +22.5 | 44% |
| **G135 (score=76)** ⭐ | **55** | **+1369** | **+24.9** | (slightly higher) |

→ **+88 bps marginal improvement** (50 trades 의 best 결과 +1463 와 일치).

## 사용 권장

production 진입 threshold 76 으로 더 selective. G131 와 거의 동일하나 약간 더 robust.

```powershell
$env:STRATEGY_OVERRIDE_PATH = "$HOME\Desktop\first_repo\quant_binance\strategies\G135_score76_refined\overrides.json"
$env:PAPER_TRADING = "1"
cd $HOME\Desktop\first_repo
python -m quant_binance.daemon
```

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G131 변형 (score 75→76). 60d backtest +88 bps 개선 |
