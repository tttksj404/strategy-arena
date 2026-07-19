# Wave-4 레버리지 스윕 사전등록

상태: 고정됨. 실행 후 조합 추가 금지.

## 대상

- 기존 엔진 임포트 재사용: W2c는 `research.wave2.funding.run_maker_portfolio`, F1f는 `research.wave1.fam_funding.run_portfolio`.
- 기존 W2c/F1f carry 진입·퇴출·랭킹·비용 룰은 변경하지 않는다. wave4는 trace를 엔진 equity path와 대조한 뒤 자본구조만 적용한다.
- 입력은 `research/wave1/cache`와 기존 `universe.json`만 사용한다. 네트워크 호출은 금지한다.

## 고정 그리드

`{W2c, F1f} × {SYM, ASYM} × {1.0, 1.5, 2, 3, 5, 10}` = 24조합.

## 자본구조

- SYM: 각 pair의 두 레그 노셔널을 `L/2` 배율로 스케일한다. 현물 차입분은 `max(0, L/2 - 0.5)`이며 연 10%를 일할 부과한다.
- ASYM: 현물 노셔널은 현금, 퍼프 노셔널은 같은 pair 노셔널이며 퍼프 증거금은 `노셔널/L`이다. 총 자본효율은 `2/(1+1/L)`이다.

## 청산 및 지표

- 보유일의 보수적 최악 basis는 `spot_low/spot_open - perp_high/perp_open`이다.
- 퍼프 손실이 `초기 퍼프 증거금 - 노셔널×0.005` 이상이면 pair를 청산한다.
- 청산 손실은 `노셔널×실제 역행분 + 노셔널×0.0006`이다.
- CAGR, 실제 equity MDD, 일별 순수익률 재표본화 MC 10,000경로의 final-capital p05 및 final `< $150` 확률, 청산 횟수, 총 현물 차입이자를 기록한다.
- 게이트: `MC p05 > 300 ∧ 파산확률 < 5% ∧ MDD ≤ 25%`.
