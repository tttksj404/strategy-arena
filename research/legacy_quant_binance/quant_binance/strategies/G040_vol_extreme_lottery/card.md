# G040 — Volatility Extreme Lottery (ATR ratio ≥ 2.5/3.0 + hold 72h)

## Status: bt-only (Q3 specialized — recovery regime detector)

부모: 신규 (PB 인용 없는 발견)
변경 변수: 신호 클래스 자체가 신규 (vol expansion)

## 셋업

ATR(14) / ATR-MA(20) ≥ 2.5 (또는 3.0) AND 직전 봉 close > open → long 진입, 72h hold.

→ **회복기 momentum burst 정확히 포착**: 알트가 변동성 폭발 + 양봉 = "바닥 찍고 반등 시작" 시그널.

## 결과 (2026-04-28, 374-day, 18 alts)

| ratio | n | net bps | WR | 분기 분포 |
|---:|---:|---:|---:|---|
| **2.5×** | **31** | **+1459** | **100%** | 31/31 in Q3 |
| **3.0×** | **23** | **+1524** | **100%** | 23/23 in Q3 |

→ **거래당 +14~15% 수익, WR 100%**. 단 374-day 중 Q3 (회복기 ~94일) 에만 작동.

## 운용 룰

```
if ATR(14) / ATR_MA(20) >= 2.5:
    if close > open:
        long 진입, 72h hold (no SL — full position)
        size: 자본 5~10% (lottery 단일 베팅)
```

## 의의

이 시그널은 "**시장 회복기 진입 자동 감지기**" 로도 동작:
- Vol expansion 2.5x+ 가 발화 = 시장이 "흥분" 상태
- 양봉 동반 = 회복 방향
- 이 신호 자체가 G041 의 walk-forward gate 보완 정보

→ G046 (가칭) = G041 + G040 vol-burst 시 가중 size up 가능

## 한계

- 표본 23~31건 (적음). 통계적 신뢰도 ★★★ 보통
- Q1/Q2 무진입 → 그 시기에 작동 X (휴면)
- 100% WR 은 **374-day 단일 cycle 결과** — 다른 회복기 (2021, 2023) 에서 재검증 필요
- ATR ratio 2.5/3.0 임계는 sample 의존 — 실제 deploy 전 walk-forward 검증

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | 신규 발견. G040 sweep 결과 ratio 2.5/3.0 hold 72 양수. Q3 specialized. PB 미인용 (자체 발견 → PB200 후속 등록 검토) |
