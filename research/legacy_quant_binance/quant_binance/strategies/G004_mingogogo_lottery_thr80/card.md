# G004 — G002 + threshold 70 → 80 (lottery whale-hunting)

## Status: bt-only (lottery overlay)

부모: G002
Playbook: PB001 (Mingogogo CH1)
변경 변수: **entry_threshold** 만 (70 → 80) — variable-1 룰 준수

## 가설

PB001 mining 결과: CH1 score 임계가 5단계 (70/55/46/31/30) 이며 **70 = "강력매수", 80+ = 사실상 점수 분포의 99 percentile**.

> threshold 80 으로 올리면 진입 빈도는 급감하지만, 진입한 순간의 평균 EV 는 폭발적으로 증가할 것이다.
> 이게 사용자 lottery 의도 ($50 도박성, 거래당 큰 수익) 정확히 핏.

## 결과 (2026-04-28, 374-day window)

| 지표 | G002 (thr 70) | **G004 (thr 80)** | 변화 |
|---|---:|---:|---:|
| trades | 2,667 | **40** | −98.5% |
| trades/day | 7.13 | **0.11** | −98.5% |
| avg net bps | +221 | **+978** | **+343%** |
| avg net % | 2.21% | **9.78%** | +343% |
| WR | 58.3% | **85.0%** | +26.7pp |
| lottery 10%+ | 451 | 20 | (50% of trades) |
| lottery 20%+ | 75 | 7 | (17.5% of trades) |

→ **40건 중 34건이 winner (85% WR)**. 평균 수익 +9.78% **거래당**. 7건은 +20% 이상 daejackpot.

## 사용 방식 권장

**G003 (production main) 와 병행 운용**:
- G003 = 일상 운용 (하루 13.6건 후보, ≥3/일 충족, 1년 +200~500% 추정)
- G004 = lottery 신호 트리거 시 별도 큰 포지션 (3~10일에 1건) — 사용자 도박성 컨텍스트의 핵심 트리거

자본 배분 권장 ($50 기준):
- G003: 60~70% capacity (~$30~35), 1x perp, alt spread
- G004: 30~40% capacity (~$15~20), 2~3x perp 가능 (40건 표본 작아 보수적), 단일 큰 포지션

## 한계

- **표본 40건만** (374일에 0.11/일) — 통계적 신뢰도 ★★★ 보통
- G006 (universe 18 + threshold 80) 도 49건만 → universe 확장으로 표본 증가 어려움
- threshold 90 = 0건 (너무 빡빡)
- **2025-2026 시장 한정 결과** — 베어 마켓 / 다른 regime 검증 X
- forward-bar 시뮬 → 72h 인터바 변동 미반영

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G002 변형 (threshold only). lottery whale-hunting variant 등록 |
