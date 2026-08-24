# G053 — G050 + Drawdown safety net (combined gate)

## Status: bt-only — OOS+IS 균등, paper-live 후보 (LUNA 보호)

부모: G050 (lookback 14d, OOS-validated)
변경 변수: **dd_safety_net** (1개) — 직전 7일 net < -3000bps 시 14일 휴면

## 가설

G050 (14d gate) 도 2022 Q2 LUNA crash 분기에서 -664 bps 누적 손실. 빠른 drawdown 감지 (7일 cumulative -3000bps 이하) + 14일 휴면 = 추가 보호.

OOS / IS 모두 +200 bps 수준으로 **수익률은 살짝 낮지만 균등성** ↑ → 라이브 변동성 ↓.

## 결과 (in-sample + out-of-sample)

| Gate | OOS 2022-23 | IS 2025-26 | diff | std |
|---|---:|---:|---:|---:|
| G050 (14d only) | n=2814, **+263** WR=59% | n=3343, **+341** WR=67% | 78 bps | 균형 OK |
| **G053 (combined)** ⭐ | **n=1616, +213 WR=60%** | **n=1652, +202 WR=62%** | **11 bps** | **거의 동일** ✅ |

→ **G053 가 G050 보다 OOS-IS gap 작음 (78→11)**. 라이브 신뢰도 더 높음.

## 룰 (combined gate 의사코드)

```python
def gate_active(now_ts, all_candidates_in_history):
    # 1. DD safety: 직전 7일 cumulative net < -3000bps 시 14일 휴면
    cutoff_dd = now_ts - 7 * 86400 * 1000
    recent_dd = [c for c in all_candidates_in_history if cutoff_dd <= c.entry_ts < now_ts]
    if recent_dd and sum(c.net_bps for c in recent_dd) < -3000:
        return False, "dd_pause_14d"
    # (paused_until 따로 관리: 1회 발동 시 ts + 14d 까지 무조건 휴면)

    # 2. Gate: 직전 14일 net > 0
    cutoff_gate = now_ts - 14 * 86400 * 1000
    recent_gate = [c for c in all_candidates_in_history if cutoff_gate <= c.entry_ts < now_ts]
    if not recent_gate:
        return True, "warmup"
    return sum(c.net_bps for c in recent_gate) > 0, "gate_open" if positive else "gate_closed"
```

핵심:
- **DD 가 14일 휴면 트리거 (시간 기반, 거래 수 무관)** — 빠른 reaction
- **Gate 14d 는 평소 운용** — 일반 regime 변화 처리
- 둘 다 hypothetical (모든 candidate 사용) — gate stuck 방지

## 운용 권장 ($55 capital)

```
Portfolio:
├── max 3 동시 포지션
├── size 30% × $55 = $16.5/거래
├── 1x perp 또는 spot
├── G053 룰 (gate14d + DD safety)
└── 기대: +200 bps avg / 거래, WR 60% (양 시기 균등)
```

## 한계

- 표본 1616 (OOS) + 1652 (IS) = 충분
- LUNA 분기 손실은 크게 줄었지만 **0 은 아님** (gate 가 7일 전이라 1주차 손실 누적 가능)
- 2024 갭 미검증 — G054 (next) 에서 채울 예정
- forward-bar 시뮬 (intra-bar 변동 미반영)

## 다음 후보

- **G054**: G053 + 2024 OOS 검증 (Binance 2024 fetch)
- **G056**: G053 size 적응 (recent net 클수록 size up, Kelly approx)
- **G057**: G053 + G040 vol-burst overlay
- **G058**: G053 + Naver overlay (cash 30% reserve)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G050 변형 (DD safety 추가). OOS+IS 균등 (+213 vs +202) → 라이브 신뢰도 ↑ |
