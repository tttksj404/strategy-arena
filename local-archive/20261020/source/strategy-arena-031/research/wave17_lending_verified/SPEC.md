# Wave-17 사전등록 -- 대여이자 실수취 검증 (동결: 2026-07-24)

## 발견 (실측, 2026-07-24 -- 이 wave의 존재 이유)
wave-16 E1/E2/E3는 OKX `lending-rate-summary`의 `avgRate`를 대여 수익으로 썼다. **이건 틀렸다.**
- OKX `lending-rate-history?ccy=X&limit=100`이 존재하며 **1시간 간격 100건(약 4일)**을 반환한다.
- 응답에 두 필드가 있다: `rate`(차입자 지불, avgRate와 같은 자릿수) vs **`lendingRate`(대여자 쪽
  실현값)**. THETA로 예를 들면 rate 약 18~19% vs lendingRate 약 6.6~7.1%.
- wave16이 쓴 `avgRate`는 `lendingRate`보다 항상 크다 -- 즉 **wave16의 대여수익 가정은
  과대평가**다. 정확한 과대평가 배율은 이 wave가 전체 유니버스에 대해 다시 잰다(사전 스팟체크
  수치는 재확인용이며 이 wave의 공식 결과가 아니다).
- 시계열 깊이는 여전히 limit=100 = 약 4일뿐 -> 과거(4일 이전) 백테스트는 여전히 불가능. 이 한계는
  wave16 치명적 한계 1과 동일하게 유지된다.

## 범위 (wave16과의 차이 -- 반드시 구분해서 읽을 것)
이 wave는 wave16의 5개 후보(E0-E4)를 재실행하지 않는다. **오직 E1(랭킹=펀딩만, 대여수익만
가산)의 재산출**만 다룬다 -- E2/E3/E4처럼 대여이자를 랭킹에 섞는 변형은 이 wave의 대상이 아니다
(그 변형들은 avgRate 문제와 별개로 이미 wave16 REGISTRY.md에서 기각됐다: E4<E0). 랭킹에 대여이자를
섞지 않는 E1류만 다시 볼 가치가 있다는 판단.

## 후보 4개 (동결, 사후 추가 금지)
공통: $100/활성 $90/레그 $45, 델타중립 1x, wave-13 실측비용, top200 유니버스(L4/E0 승계),
랭킹은 항상 펀딩APR만(대여이자 랭킹반영 없음 -- wave16 E1과 동일 전제).
| ID | 대여수익(펀딩 위에 가산) | 역할 |
|---|---|---|
| F0 | 0 (없음) | L4/E0 재현 -- 기준선 |
| F1 | 실측 lendingRate 중앙값(4일, 시간당 100건) | 핵심 재산출 -- wave16 E1의 정정판 |
| F2 | F1 x 50% (보수 할인) | 스트레스 -- 변동성/미체결 대비 |
| F3 | 0 (F0과 동일해야 함) | 회귀/무결성 검증 -- 실패 시나리오 겸용 |

추가 참고(게이트 대상 아님): F_min -- 코인별 4일 관측 구간 중 **최저** lendingRate를 대여수익으로
가정한 시나리오(할인 없음, 이미 최악값이므로). "변동성이 이 수익원을 얼마나 갉아먹을 수 있는가"의
하한 참고용.

## 방법
1. `fetch17.py`: wave16 스냅샷 유니버스(cache/lending_snapshot.json의 `by_symbol`에서
   `lending_available=True`인 심볼들의 `base_ccy_matched` 고유집합, 112종)에 대해
   `lending-rate-history`(limit=100)를 코인당 1회 수집 + 같은 세션에 `lending-rate-summary`를
   재수집(avgRate 신선도 일치) -> `cache/lending_realized.json`. 코인별
   rate/lendingRate 중앙값·평균·표준편차·최소·최대와 `ratio = lendingRate중앙값/avgRate` 저장.
2. `recompute17.py`: `research.wave16_duallayer.engine16`을 임포트 재사용(엔진 재구현 금지).
   `engine16.build_runner()`에 실측 lendingRate 기반의 합성 lending_snapshot을 넣고,
   `DualLayerRunner.run_variant(ranking_discount=0.0, pnl_discount=X)`를 X=0/1/0.5/0으로 호출해
   F0-F3을 산출(ranking_discount는 F0-F3 전부 0.0 고정 -- E1류 전제).
3. `volatility17.py`: 코인별 lendingRate 표준편차·범위(max-min)·변동계수(CV=표준편차/평균) 표.
4. 운영 리스크(코드 아님, 조사): OKX 공개 도움말·API 문서에서 flexible/fixed-term, 한도, 계층,
   수수료 구조를 확인 시도 -> 확인 안 되는 항목은 UNCONFIRMED로 리포트에 명시(추정 금지).
5. 게이트(gates17.py): F1이 F0를 이기는지, F2가 F0를 이기는지, F3가 F0와 (허용오차 내) 동일한지
   -- 3개뿐. wave16의 S1-S5 통계 배터리는 재실행하지 않는다(F0-F3 전부 ranking_discount=0.0 고정이라
   funding-only 성분이 항상 F0 자기 자신과 동일 -- wave16 E0가 이미 그 성분을 S1-S5 PASS시켰다,
   research/wave16_duallayer/results/E0.json 참조, 재검증 아니라 참조).

## 판정
- F1 > F0 ∧ F2 > F0 ∧ F3 == F0(허용오차) -> "실수취 기준 재확인 유효".
- F1이 F0를 못 이기면 못 이긴다고 그대로 보고한다 -- 수치 조정 금지.
- 모든 "결합" 수치는 wave16과 동일한 라벨("단면 근거, 시계열 미검증")을 유지한다. 이 wave는 그
  라벨을 없애지 않는다 -- 대여수익 크기 추정치를 avgRate에서 lendingRate로 **정정**할 뿐, 시계열
  백테스트 불가라는 근본 한계는 그대로다.

## 정직성 제약 (SPEC 차원에서 고정)
- `lendingRate`가 대여자 실수취분이라는 것도 필드명 기반 추론이다 -- OKX 문서로 확인 시도하고,
  확정 못 하면 "필드명 근거 추론, 실계좌 미확인"으로 명시한다.
- 4일 시계열의 중앙값을 연환산 상수로 쓰는 것은 강한 가정이다 -- 리포트 최상단에 명시한다.
- research/wave17_lending_verified/ 밖 수정 금지. research/wave4_leverage/ 접근 금지.
