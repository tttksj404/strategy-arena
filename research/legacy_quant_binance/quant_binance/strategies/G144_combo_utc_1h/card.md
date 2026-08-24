# G144 — G135 + UTC 06-09 + holding 1h (combo)

## Status: REJECTED (G135 대비 -2303 bps)

부모: G135
변경: G140 (UTC 06-09 entry) + G141 (1h hold) 결합

## 결과 (60d backtest, score>=76, 4-coin, cost 16, post-filter)

| variant | n | WR | avg net | total | annPct |
|---|---:|---:|---:|---:|---:|
| G135 baseline (4h) | 126 | 46.0% | +18.9 | +2378 | +144.7% |
| G140 (UTC + 4h) | 12 | 66.7% | +31.8 | +382 | +23.2% |
| G141 (전체 + 1h) | 126 | 40.5% | +1.3 | +168 | +10.2% |
| **G144 (UTC + 1h)** | **12** | **33.3%** | **+6.2** | **+75** | **+4.5%** |

→ G140 의 시간대 quality 우위가 1h hold 로 사라짐. WR 33.3% 까지 떨어짐.
→ UTC 06-09 entry 의 강점은 **4h hold 와 결합할 때만** 발현. 1h hold 와는 anti-synergy.

## 결론

UTC 시간 filter + 짧은 hold 결합은 의미 X. Production 후보 X.

## 변경 이력
| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G140+G141 combo 시뮬. 양 변형이 anti-synergy 보여 REJECT |
