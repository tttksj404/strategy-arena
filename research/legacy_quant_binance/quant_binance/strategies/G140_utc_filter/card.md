# G140 — G135 + UTC 06-09 entry gate

## Status: PROMISING (per-trade quality 우월) — 단, **trade 빈도 부족**

부모: G135
변경: entry time UTC 06-09 (KST 15-18) 에만 신규 진입 허용. 사용자 closed_trades 분석에서 UTC 07 (KST 16시) 가 winners 22건 중 16건 (73%) 으로 압도적이었던 패턴 재현 시도.

## 결과 (60d backtest, score>=76, 4-coin, cost 16, post-filter)

| variant | n | WR | avg net | total | annPct |
|---|---:|---:|---:|---:|---:|
| G135 baseline | **126** | 46.0% | +18.9 | **+2378** | +144.7% |
| **G140 UTC 06-09** | **12** | **66.7%** | **+31.8** | +382 | +23.2% |

→ **per-trade quality 두 지표(WR +20.7pp, avg +12.9 bps) 모두 우월**.
→ 그러나 **trade 빈도가 10배 감소** (126→12). 60d 12 trades = 5d 당 1건. 사용자 lottery 컨텍스트 (≥3건/일) 와 정반대.
→ 사용자 closed_trades 의 UTC 07 winners 우세는 **5분 단타** 환경 (median hold 5.3분). 4h hold backtest 는 다른 게임.

## hour distribution (G135 4h)

| UTC | n | wins | wr% | total bps |
|---:|---:|---:|---:|---:|
| 13 | 6 | 5 | **83.3** | **+1297** |
| 16 | 8 | 6 | **75.0** | **+1231** |
| 4 | 4 | 1 | 25 | +776 |
| 18 | 3 | 3 | **100** | +355 |
| **14** | **12** | **3** | **25** | **-1058** ⚠ |
| 19 | 8 | 1 | 12.5 | -424 |

→ 4h hold 기준 best hours: UTC 13, 16. 사용자 5분 패턴(UTC 07)과 **불일치** — 시간대 효과는 hold 기간에 강하게 의존.

## 결론

G140 (UTC 06-09 4h hold) 는 baseline 대비 trade 빈도가 너무 낮아 production 가치 X. 단, hour 분포 데이터에 따르면 **UTC 13/16 entry + 4h hold 게이트** 검토 가치 있음 — 후속 G147 후보.

## 변경 이력
| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G135 + UTC 06-09 post-filter 시뮬. trade 빈도 부족으로 REJECT, 단 시간대 효과는 4h hold 환경에서 UTC 13/16 으로 시프트 |
