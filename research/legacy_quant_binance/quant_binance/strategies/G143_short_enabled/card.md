# G143 — G135 + short enabled (롱·숏 양방향)

## Status: NEUTRAL — G135 대비 marginal 변화 (-106 bps)

부모: G135
변경:
- futures_exposure.short_disabled: true → false
- short_extra_score_floor: 20.0 → 5.0 (relax)
- short_extra_edge_bps: 10.0 → 2.0 (relax)

## 결과 (60d backtest, score>=76, 4-coin, cost 16, 4h hold)

| variant | n | WR | avg net | total | annPct |
|---|---:|---:|---:|---:|---:|
| G135 baseline | 126 | 46.0% | +18.9 | +2378 | +144.7% |
| **G143** | **127** | **45.7%** | **+17.9** | **+2272** | **+138.2%** |

→ **+1 trade (short), 거의 동일** (-106 bps).
→ Short slot 활성화돼도 짧은 트리거가 score>=76 + extra floors 통과하기 거의 어려움.
→ "거래당 큰 수익 lottery" 컨텍스트에서 양방향 노출은 의미 있으나, 현 score gate 가 너무 빡빡.

## 후속 권장

short_extra_score_floor 를 0~3 까지 더 낮추거나, short 전용 별도 score curve 적용한 G146 검토. 현 G143 은 G135 대비 불필요한 복잡도 추가.

## 변경 이력
| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G135 변형 (short 활성화). 60d 검증 후 NEUTRAL — 활성 의미 X |
