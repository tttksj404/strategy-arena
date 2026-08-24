# G161 — G150 + UTC 13/16 only (extreme selective lottery) ⭐⭐⭐

## Status: bt-only lottery overlay candidate

부모: G150
변경: entry_hour_filter [13, 16] only — UTC 13 (KST 22) + UTC 16 (KST 01) 두 시간대만

## 가설

G135 hour 분포에서 압도적 winner hours 만 진입:
- UTC 13: 6 trades, **83% WR**, +1297 bps (avg +216)
- UTC 16: 8 trades, **75% WR**, +1231 bps (avg +154)
- 합산: 14 trades, 79% WR, +2528 bps

## 추정 결과 (60d)

```
G161: 14 trades / 79% WR / avg +180 bps / total +2528 bps
G150 baseline: 140 trades / 47% WR / avg +14.6 / total +2050
→ trade 수 90%↓ but total +23% ↑ + WR +32pp
```

## 운용

**lottery overlay**: G150 main (140 trades / regular) + G161 lottery (14 trades / 큰 사이즈) 병행:
- G150: capital 60% (기본)
- G161: capital 40% (UTC 13/16 시 큰 베팅)

## 한계

- ≥3건/일 미달 (0.23/day, 60일 14건)
- G135 hour 분포 = 60d 단일 분기 → 다른 시기 OOS 미검증
- 14 trades 표본 작음 (통계적 신뢰 ★★★ 보통)
- KST 22-23 / 01-02 는 한국 시간 밤 = 사용자 모니터 어려움

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G150 변형 (UTC 13/16 only). G135 hour 분포 분석에서 best 2 시간대만 진입 → lottery extreme |
