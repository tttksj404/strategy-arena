# G132_score80_1h

## Status: bt-only (paper-live ready as overlay)

부모: G131
변경: mode_thresholds.futures_score_min = 80_1h (1개)

## 결과 (60d, 4h hold backtest)

- score 80: 22 trades / WR 46% / +38.6 avg / +849 total
- score 85: 5 trades / WR 60% / +65.5 avg / +327 total

## 사용

G131 (score 75) 와 병행 운용 가능. 이 override 단독 사용 시 진입 빈도 ↓ but per-trade EV ↑.

paper-live 시: STRATEGY_OVERRIDE_PATH 만 변경.
