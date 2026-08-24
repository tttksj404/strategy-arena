# G041 — G003 + Walk-forward adaptive gate (PRODUCTION CANDIDATE) ⭐

## Status: bt-only (모든 분기 양수)

부모: G003
변경 변수: **deployment_gate** (1개) — 진입 시점에 직전 30일 누적 net PnL 평가, 양수일 때만 진입 허용

## 가설

G003 의 regime 의존성 (Q2/Q4 음수) 을 **walk-forward rolling backtest** 로 자동 해결:
- 직전 30일 G003 누적 net 이 양수 → 다음 진입 허용
- 음수 → 다음 진입 보류 (regime 회복 대기)
- 첫 30일은 warm-up (무조건 trade)

이는 backtest hindsight 가 아니라 **legitimate walk-forward** (각 시점에서 그 시점의 과거만 사용).

## 결과 (2026-04-28, 374-day window, 18 alts)

| 지표 | G003 baseline | **G041 adaptive** | 변화 |
|---|---:|---:|---:|
| trades | 5,089 | **3,398** (33% cut) | −33% |
| avg net bps | +208 | **+298** | **+43%** |
| WR | 59.1% | **65.0%** | **+6pp** |
| Q1 net | +361 | +363 | ≈ |
| **Q2 net** | **−26** | **+93** | **NEGATIVE → POSITIVE** ✅ |
| Q3 net | +487 | +452 | ≈ |
| **Q4 net** | **−21** | **+19** | **NEGATIVE → POSITIVE** ✅ |

→ **모든 분기 양수**. regime 의존 해결.

## 운용 룰

```python
def should_enter(now_ts, recent_30d_trades):
    if recent_30d_trades is None or len(recent_30d_trades) == 0:
        return True  # warm-up
    return sum(t.net_bps for t in recent_30d_trades) > 0
```

추가 안전 룰:
- 30일 net 음수일 때 strategy "휴면", 단 매일 재평가
- 휴면 중에도 score 계산은 계속 (재진입 시점 판단)
- 휴면 7일+ 시 사용자 알림

## 다음 후보

- **G042**: G041 + 14일 lookback (더 짧은 reaction time)
- **G043**: G041 + drawdown 기반 (recent max DD < 5%)
- **G044**: G041 + position sizing (recent net 클수록 size up)
- **G045**: G041 + G004 lottery overlay (CH1 80 신호 시 별도 큰 포지션)

## 한계

- 첫 30일은 warm-up → 실제 deploy 시 30일 paper-live 필요
- 374-day 윈도우 단일 regime cycle만 검증 — 더 긴 데이터로 재확인 필수
- 30일 lookback 이 최적인지 (14/60/90 sweep 미실시)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G003 변형 (deployment_gate). regime 의존성 자동 해결. 모든 분기 양수 → production candidate |
