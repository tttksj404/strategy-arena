# G141 — G135 + holding 1h (vs 4h)

## Status: REJECTED (G135 대비 -2210 bps)

부모: G135
변경: holding_period 4h → 1h (analyze_backtest CLI 인자, override 변경 X)

## 결과 (60d backtest, score>=76, 4-coin, cost 16)

| variant | n | WR | avg net | total | annPct |
|---|---:|---:|---:|---:|---:|
| G135 baseline (4h) | 126 | 46.0% | +18.9 | +2378 | +144.7% |
| **G141 (1h)** | **126** | **40.5%** | **+1.3** | **+168** | **+10.2%** |

→ **WR -5.5pp, avg -17.6 bps, total -2210 bps**.
→ 동일 entry slate 인데 1h forward return 이 4h 대비 net edge 부족 (cost 16bps fixed).

## 결론

4h hold 가 cost 16bps 환경에서 단연 우월. 1h 단축은 거래당 평균 alpha 가 cost 와 거의 같은 수준(+1.3 bps)으로 무의미.
**5분 micro-hold 시뮬은 본 backtest 인프라(1h klines)로 불가** — 별도 5m kline 기반 backtest 필요.

## 변경 이력
| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G135 변형 (4h→1h hold). 60d 검증 후 REJECT |
