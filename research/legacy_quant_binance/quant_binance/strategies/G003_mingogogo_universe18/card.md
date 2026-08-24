# G003 — G002 + universe 8 → 18 alts

## Status: bt-only (production-grade candidate)

부모: G002
Playbook: PB001 (Mingogogo CH1)
변경 변수: **universe** 만 (8 → 18) — variable-1 룰 준수

## 가설

G002 의 +221 bps net 이 8개 cherry-picked 알트의 우연인지, PB001 CH1 자체의 alpha 인지 검증.

> universe 를 18종으로 확장해도 net expectancy 가 유지된다면, PB001 CH1 은 **종목 비특화 알파** 보유.
> 만약 net 이 절반 이하로 떨어지면 G002 는 8종 overfitting.

## 결과 (2026-04-28, 374-day window)

| 지표 | G002 (8 univ) | **G003 (18 univ)** | 변화 |
|---|---:|---:|---:|
| trades | 2,667 | **5,089** | +91% |
| avg net bps | +221 | **+208** | −6% |
| WR | 58.3% | **59.1%** | +0.8pp |
| lottery 10%+ | 451 | **737** | +63% |
| lottery 10%+ /day | 1.21 | **1.97** | +63% |

→ **net 거의 유지 + 표본 2배 + lottery 빈도 1.6배**. PB001 alpha 검증 완료.

## Universe (18 alts)

```
DOGE PEPE WIF ARB OP AVAX SUI ADA   (G002 baseline 8)
APT BNB DOT LINK LTC NEAR SOL UNI XRP BTC   (추가 10, MATIC/ETH 제외)
```

## 결론

- **production candidate 1번**. 거래 빈도 (5089/374 = 13.6/일) 사용자 ≥3건/일 충족
- 1년 누적 PnL (이론): 5089 trades × +208 bps = **+1.06M bps** = +10,600% gross
  - 단 동시 보유 capacity 한계로 capture rate 100% 불가능 — 실제 1x 자본 spread 가정 +200~+500% 추정
- **다음 후보**:
  - G005 (hold 변형) → 24h 는 열등 확인 → 폐기
  - G007 = G003 + intra-bar TP/SL (R 1.5/3.0) → liquidation 방지 + 추가 EV 가능
  - G008 = G003 + CH4 (바닥권) 필터 → selectivity ↑ 시도
  - G009 = G003 + Naver overlay (cash reserve 30% / 매크로 필터) → DD 감소

## 한계

- 374-day 윈도우 (2025-03 ~ 2026-04) — 단일 시장 regime
- forward-bar 시뮬 (intra-bar TP/SL 미모델링) — 72h hold 시 -50%+ 인터바 likely → 실제 결과 더 나쁠 수
- **5x+ 레버리지 절대 금지** (liquidation). spot 또는 1-2x perp 권장
- Cherry-pick 의심: 18종은 archive 의 모든 가용 alt → 자유도 적음 (이미 random sample 에 가까움)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G002 변형 (universe only). overfitting 검증 통과 → production candidate |
