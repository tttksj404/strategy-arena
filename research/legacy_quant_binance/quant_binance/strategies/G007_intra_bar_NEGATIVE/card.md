# G007 — G003 + intra-bar TP/SL (NEGATIVE result)

## Status: dead — 학습용 보존

부모: G003
변경 변수: **exit_logic** (close-after-72h → intra-bar TP1.5R/SL1.0R)

## 가설

72h hold 의 인터바 liquidation 위험을 ATR×1.5 SL + TP 1.5R 로 차단하면, 안전성과 EV 둘 다 개선될 것이다.

## 결과 (NEGATIVE)

| 변형 | n | net bps | WR | TP% | SL% | TO% |
|---|---:|---:|---:|---:|---:|---:|
| **G003 baseline (no SL/TP)** | 5089 | **+208** | **59.1%** | — | — | — |
| G007 SL×1.5 / TP 1.5R | 1609 | **−25** | 36.7% | 35.9% | 63.0% | 1.1% |
| G007b SL×3.0 / TP 1.5R | 1151 | −24 | 40.9% | 25.5% | 52.2% | 22.3% |
| G007c SL×3.0 / TP 3.0R | 1143 | +1 | 38.4% | 8.1% | 53.8% | 38.1% |
| G007d SL×5.0 / TP 3.0R | 948 | +8 | 46.6% | 2.4% | 34.0% | 63.6% |

## 결론

**TP/SL 자체가 PB001 alpha 를 파괴**. 핵심 메커니즘:
- PB001 신호의 winner 는 진입 후 인터바 drawdown 을 거친 후 72h close 시점에 +218 bps 회수
- SL 1.5×ATR 적용 시 **63% 가 인터바 SL hit** → 진짜 winner 들이 미리 stop-out
- TP 1.5R 적용 시 lottery 10%+ (450~700건) 가 cap 됨 → 우상단 분포 잘림

→ **인터바 TP/SL = PB001 부적합**. G003 운용 시 다음 제약 필수:
- **레버리지 절대 X**: spot 또는 1x perp 만
- **사이즈 ≤ 10% 자본/거래**: 한 알트 -50% drawdown 발생해도 자본 -5% 에 그침
- **72h 까지 hold 강제**: SL 없이 시간만료 청산만

## 의의 (학습 자산)

이 음수 결과는 G003 production 운용 조건을 명확히 했다:
- 5x+ 레버리지 자제 룰이 단순 권고가 아니라 **수학적 필수**
- "인터바 안전장치" 라는 본능적 충동이 어떻게 alpha 를 죽이는지 정량 입증
- 다음 후보 G011 = G003 + Naver overlay (사이즈 cap, cash reserve) 가 정공법

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G003 변형 (exit_logic). 4개 SL/TP 조합 모두 G003 baseline 보다 열등. dead 처리. 학습용 보존 |
