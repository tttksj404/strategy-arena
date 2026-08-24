# G160 — G150 + UTC 12-13/15-17 entry gate (worst UTC 14 제외)

## Status: bt-only candidate (60d 단일 분기 추정)

부모: G150
변경: entry_hour_filter (post-filter, regime engine override 직접 미지원 — daemon side 적용 필요)

## 가설

G135 hour 분포 분석 (60d 4h hold):
- UTC 13 (KST 22): +1297 bps / WR 83% (avg +216) ⭐
- UTC 16 (KST 01): +1231 bps / WR 75% (avg +154) ⭐
- UTC 14 (KST 23): -1058 bps / WR 25% ❌ worst
- UTC 12,15,17: 작은 음수~양수

G160 = UTC 12, 13, 15, 16, 17 5시간만 (UTC 14 제외).

## 추정 결과 (60d, G135 hour dist 기반)

| | n | WR | avg | total |
|---|---:|---:|---:|---:|
| G150 baseline | 140 | 47% | +14.6 | +2050 |
| **G160** | **40** | **55%** | **+54.7** | **+2186** |

→ trade 수 71% ↓, total +6%, **avg per-trade 3.7배 ↑**, WR +8pp.

## 한계

- 60d 단일 분기 결과로만 추정. 다른 시기 OOS 미검증
- G135 hour 분포 사용 (G150 자체 분포 X — 동일 가정)
- post-filter 구현 필요 (regime engine 직접 X)
- ≥3건/일 미달 (0.67/day = 사용자 일상 운용에 부족)

## 활성화

post-filter 로 daemon 또는 별도 wrapper 에 entry hour gate 추가 필요. 현재 override 자체로는 강제 불가 — daemon code 수정 필요.

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G150 변형 (UTC 12-17 except 14). G135 hour 분포 분석 결과 적용 |
