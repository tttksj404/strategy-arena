# FIX PACKET R1 — 적대적 리뷰 결함 3건 수정

리뷰 전문: 아래 요약이 전부다. 대상 코드 `research/wave1/`. 수정 후 pytest 전부 통과 필수.

## 1. CRITICAL — 트레일링 스탑 레버리지 오염
`backtest.py:63`(`_stop_fill`) + `fam_tsmom.py:111-118`: 스탑 거리 계산이 `distance * state.position`으로 vol-target 레버리지 크기(≤3x)까지 곱함. 스탑은 **방향 부호(±1)만** 반영해야 함 — 가격 기준 스탑 레벨은 레버리지와 무관. position=2.5일 때 스탑이 2.5배 멀어지는 버그. `sign(position)`으로 고정하고, 회귀 테스트 추가: 동일 가격경로에서 position 1.0 vs 2.5 모두 같은 바에서 스탑 발동.

## 2. HIGH — F1 캐리 엔진 룩어헤드 불변성 테스트 부재
`tests/test_no_lookahead.py`는 F2 `run_backtest` 경로만 검증. `fam_funding.py`의 `run_portfolio`에 동일 계약 테스트 추가: 합성 펀딩/가격 시리즈에서 미래 구간(마지막 N바) 값을 변형해도 그 이전 에쿼티 곡선이 비트 단위 동일해야 함. 위반 발견 시 코드도 수정.

## 3. MEDIUM — 부분 리밸런스 trade_weights 고정
`fam_funding.py:196-199`: 보유 중 비중 변경 시 `trade_weights[symbol]`이 최초 진입 비중에 고정 → 청산 시 `trade_values` 왜곡 → Kelly(게이트9)/MC(게이트5·16) 입력 오염. 비중 변경 시점마다 갱신(또는 비중변경을 부분청산+재진입 트레이드로 분해)하고, 주 지표(에쿼티) 불변임을 테스트로 확인.

## 완료 기준
- pytest 전부 통과(기존 13 + 신규 ≥3).
- 수정 diff는 최소 침습. 보고 50자 이내 + 변경 파일 목록만.
