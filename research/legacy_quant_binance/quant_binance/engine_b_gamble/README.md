# Engine B — 도박 거버너 (Gambling Governor)

**엣지 없음**(PEPE 백테스트 2000/2000 파산 실증). 이 엔진은 알파를 찾지 않는다 —
**파산 스파이럴을 구조적으로 차단**하는 규율 강제기다.

## 왜 거버너인가
스파이크-페이드·고배 도박은 −EV(비용 후 ~공정 동전). 승률은 높지만(83%) 전재산+무손절+반복 = **100% 파산**.
죽이는 건 레버 숫자가 아니라 **버티기 + 수익 재투입 + 정산후 재진입**. 거버너가 이 셋을 물리적으로 막는다.

## 규율 (불변식 런타임 강제)
- **I1 링펜스**: at_risk ≤ disposable. 잃어도 되는 고정액 초과 위험 불가.
- **I2 banked 보호**: 수익은 banked로 인출, 절대 재위험화 안 됨.
- **I3 단일 도박**: 동시 2개 불가.
- **I4 강제 익절/손절**: target+stop 없으면 진입 불가(버티기 금지).
- **I5 정산후 잠금**: 승/패 무관 정산 즉시 locked. `fund()`(의식적 새 자금) 전까지 재진입 불가 = 스파이럴 차단.

## 사용 흐름
```python
from gamble_governor import GambleGovernor
g = GambleGovernor("state.json")
g.fund(7.0)                       # 잃어도 되는 $7만 링펜스
g.open_gamble(target_mult=2.0, stop_mult=0.01)   # 원샷: +100% 익절 / -99% 손절(고배)
# ... 거래소에서 실제 베팅(별도 실행층, 승인 게이트) ...
g.settle("win")                   # 목표달성 -> banked로 전액 인출, 자동 locked
# 다시 하려면 의식적으로 g.fund(...) — 자동 스파이럴 불가
g.save()
```

## 검증 (`test_governor.py` → `governor_results.json`)
**8/8 PASS** — 8가지 스파이럴 유발 시도 전부 차단. 5연승 후 손실에도 banked 보존, 손실은 disposable 한정.

## 정직표 (`bold_play.py`)
$7 bold-play: 2x 도달 47%, 5x 17%, 30x 2.5%. **EV<0** — 오락이지 투자 아님.

## ⚠️ 실행층 (미배선, 승인 게이트)
거버너는 *규율*만 강제한다. 실제 거래소 주문(레버리지 진입·익절/손절 OCO)은 ccxt로 별도 배선 + 사용자 승인 필요.
**절대 규칙**: 버티기 금지, compound 금지, 정산후 자동 재진입 금지. Engine A(복리)와 자금 분리.
