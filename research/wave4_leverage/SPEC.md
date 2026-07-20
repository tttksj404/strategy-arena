# Wave-4 레버리지 스윕 사전등록

상태: 고정됨. 실행 후 조합 추가 금지.

## 대상

- 기존 엔진 임포트 재사용: W2c는 `research.wave2.funding.run_maker_portfolio`, F1f는 `research.wave1.fam_funding.run_portfolio`.
- 기존 W2c/F1f carry 진입·퇴출·랭킹·비용 룰은 변경하지 않는다. wave4는 trace를 엔진 equity path와 대조한 뒤 자본구조만 적용한다.
- 입력은 `research/wave1/cache`와 기존 `universe.json`만 사용한다. 네트워크 호출은 금지한다.

## 고정 그리드

`{W2c, F1f} × {SYM, ASYM} × {1.0, 1.5, 2, 3, 5, 10}` = 24조합.

## 자본구조

- 원본 엔진은 한 pair의 두 레그 종가 MTM을 하나의 pair 수익률로 정규화한다. 따라서 스윕의 `notional_multiplier`는 레그별 배율이 아니라 이 pair 수익률에 적용하는 총 pair-P&L 스케일이다.
- SYM pair-P&L 스케일은 `L`이며, 현물 차입분은 `max(0, L/2 - 0.5)`이고 연 10%를 일할 부과한다.
- ASYM pair-P&L 스케일은 총 자본효율 `2/(1+1/L)`이며 현물 차입은 없다. L=1에서는 두 구조 모두 pair-P&L 스케일이 1이어야 원본 엔진 대사가 성립한다.

## 일별 손익, 청산 및 지표

- 일별 손익은 W2c/F1f 원본 엔진의 동시점 종가 두 레그 MTM 경로를 그대로 재생한다. 스윕은 이 경로에 레버리지 스케일, 현물 차입이자, 청산 오버레이만 추가한다.
- L=1의 SYM/ASYM은 원본 엔진의 CAGR·MDD를 재현해야 하며, 두 지표의 상대오차가 각각 `≤ 1%`가 아니면 대사 게이트가 실패하고 리포트를 발행하지 않는다.
- 청산 체크용 기준 바 내 최악 basis 역행은 `abs(simultaneous_close_basis_change) + max(0, perp_intraday_range_pct - spot_intraday_range_pct) * 0.5`이다. 스트레스 변형은 이 값을 `× 1.5`로 계산하고 별도 청산 횟수 열에 기록한다.
- 퍼프 손실이 `초기 퍼프 증거금 - 노셔널×0.005` 이상이면 pair를 청산한다.
- 청산 손실은 `노셔널×abs(최악 basis 역행분) + 노셔널×0.0006`이다.
- CAGR, 실제 equity MDD, 원본 엔진 거래수익률을 레버리지 스케일한 MC 10,000경로의 final-capital p05 및 final `< $150` 확률, 청산 횟수, 스트레스 청산 횟수, 총 현물 차입이자를 기록한다.
- 게이트: `MC p05 > 300 ∧ 파산확률 < 5% ∧ MDD ≤ 25%`. 이 위험 게이트의 그리드와 기준은 불변이다.
