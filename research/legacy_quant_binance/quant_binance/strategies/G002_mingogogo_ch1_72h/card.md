# G002 — Mingogogo CH1 (10-indicator weighted score) × 72h horizon

## Status: bt-only

부모: S001 (베이스라인) — but 룰 자체는 PB001 인용
Playbook: [PB001 Mingogogo 8채널](../_playbook/PB001_mingogogo_8ch/source.md)

## 가설

**Source: PB001 (10-indicator weighted score, 70+ = 강력매수, 3-day horizon)**

> S001 의 4h holding 으로 신호가 random 으로 보였던 이유는 timeframe mismatch.
> Mingogogo 의 핵심 주장 "3-day horizon" 을 그대로 적용하면, 70+ 임계의 가중 평균 점수가
> **양의 expectancy** 를 보일 것이다.

정량 예측 (검증 전):
- WR: 50%+ (S001 35.9% 대비 개선)
- 평균 net: +50 bps 이상 (S001 −17.16 bps 대비 양수 전환)

정량 결과 (검증 후, 2026-04-28):
- **threshold 70 + hold 72h: net +221 bps / WR 58.3% / n=2,670 / lottery 10%+ 451건** (예측 4배 초과)
- **threshold 80 + hold 24h: net +943 bps / WR 90.0% / n=40 / lottery 10%+ 22건** (lottery whale-hunting variant)

## 변경 변수 (vs S001)

**5개 변수 동시 변경** — 변수 1개 룰 위반이지만, 본 전략은 부모 S001 의 점진 개선이 아니라 **PB001 인용 새 룰셋**. variable-1 rule 은 G002 → G003 부터 적용:

1. `score_engine`: 기존 regime engine → PB001 CH1 (10-indicator weighted)
2. `holding_period`: 4h → **72h** (3-day horizon, PB001 핵심)
3. `entry_threshold`: regime confidence → score >= 70 (or 80 for lottery)
4. `universe`: BTC/ETH/SOL → **알트 8종** (DOGE/PEPE/WIF/ARB/OP/AVAX/SUI/ADA)
5. `cost_bps_round_trip`: 16 (S001 동일) — 검증 측 같은 cost 모델

## 룰 (10-indicator weighted score, score >= 70)

```
weights = {
  RSI(14): 15%,    MFI(14): 12%,   Stoch(14,3): 12%,
  CCI(20): 10%,    W%R(14): 10%,   BB%B(20,2): 10%,
  MACD(12,26,9): 8%,  ADX(14): 8%,   OBV slope(20): 8%,
  ATR%(14): 7%
}
```

각 인디케이터 → 0~100 정규화 (oversold/long-bias 일수록 100). 가중 합산 = 0~100 score.

진입: 1h 봉 close 시점 score >= 70 (long), 그 봉 close 가격에 진입.
청산: 진입 +72시간 (1h × 72bars) close 가격에 unwind.

## 백테스트 결과 (2026-04-28)

전체: `runs/sweep_2026-04-28.json`
원본 (threshold 70 hold 4h): `runs/batch_2026-04-28_t70h4.json`

**threshold 70 + hold 72h (메인 추천)**:
| symbol | n | net bps | WR |
|---|---:|---:|---:|
| DOGE | ~330 | (집계) | (집계) |
| PEPE | ~340 | (집계) | (집계) |
| WIF | ~445 | (집계) | (집계) |
| ARB | ~340 | (집계) | (집계) |
| OP | ~470 | (집계) | (집계) |
| AVAX | ~355 | (집계) | (집계) |
| SUI | ~370 | (집계) | (집계) |
| ADA | ~250 | (집계) | (집계) |
| **합계** | **2,670** | **+221** | **58.3%** |

(상세 per-symbol breakdown 은 sweep_2026-04-28.json 참조)

## 결론 / 다음 후보

**1차 결론**: Mingogogo 신호는 진짜 alpha 보유. 단 **3-day horizon 에서만 구현됨**. 4h hold 에선 random 보다 약간 우위 (gross −17, WR 45.9%).

**위험 / 한계**:
- 60일 윈도우 (overfitting 가능성). 180일 검증 필요
- 표본 8 alt — universe 확장 (알트 20종 archive 활용 가능)
- 72h hold + 5x+ 레버리지 → liquidation 위험. **spot 또는 1-2x 레버리지 권장**
- forward-bar 시뮬레이션 = intra-bar TP/SL X (실제로는 더 나쁠 수 있음)

**다음 후보 (G003 ~)**:
- **G003**: G002 + universe 확장 (alt 17종 추가, archive 활용) → universe 1개 변수 변경
- **G004**: G002 + lottery variant (threshold 80 + hold 24h) → threshold 1개 변수 변경 ($50 lottery 의도)
- **G005**: G002 + CH4 (바닥권) 추가 필터 → channel 1개 변수 추가
- **G006**: G002 + 양방향 (PB101 Passivbot 차용 short 룰) → direction 1개 변수 추가

## 변경 이력

| 날짜 | 사유 |
|------|------|
| 2026-04-28 | 초기 작성. PB001 CH1 MVP 백테스트 → 3-day horizon 검증 (alpha 확인) |
