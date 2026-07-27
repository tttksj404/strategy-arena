# Wave-22 사전등록 -- G1 과최적화 정밀판정 (동결: 2026-07-27)

## 왜 필요한가
wave-21 GA가 산출하고 gross 1x 제약 복원(top_k_pairs 3->1)으로 확정된 G1(`research/STRATEGY_CARD.md` "G1 확정")은 wave21 자체 게이트(H1/H2/H3/H5)를 통과했다. 그러나 IS-OOS 격차가 33.5%p(I5 자체 격차 8.2%p 대비 상대 25.3%p)로, 레짐효과를 넘는 약한 과최적화 신호가 남아 있다. 이 신호가 "진짜 엣지의 자연스러운 IS>OOS 감쇠"인지 "우연히 잘 맞은 파라미터 조합"인지는 wave21의 6개 게이트만으로 구분되지 않는다 -- 그 게이트들은 전부 "이 특정 조합이 기준을 넘는가"만 보고, "이 조합의 이웃/시간/레짐/유전자별/무작위 대비 위치가 어떤가"는 보지 않기 때문이다. wave22는 이 질문 전용이다.

## G1 정의 (동결, 변경 금지)
`research/wave22_overfit/genomes.py`의 `G1_GENOME`. wave21 GA 원본 출력에서 `top_k_pairs`만 3->1로 수정한 것 -- 그 외 6개 유전자는 GA 원본 그대로.

## 평가 엔진 (재사용, 재구현 금지)
`research/wave21_ga/fitness.py`의 `build_market_cache`/`run_backtest`/`cagr`/`oos_slice`/`_max_drawdown`. wave22는 새로운 탐색을 하지 않으므로(고정 후보 감사) OOS seal을 재사용하지 않고 `mode=MODE_OOS_FINAL`로 전체 구간에 접근한다 -- `research/wave22_overfit/evaluate.py` 모듈 docstring 참조.

## 6종 검증 및 사전등록 방법론

1. **파라미터 안정성 지형** (`sensitivity.py`): 연속형 3축은 현재값의 +-10%/+-20%(경계 클립), 순서형 3축은 정렬된 선택지에서 1/2 step, idle_mode(무순서)는 나머지 3개 값. 축별 "이웃평균 전기간CAGR / G1 전기간CAGR" 비율 산출, **7축 중 최솟값**을 종합 안정성 비율로 채택(평균이 아님 -- 한 축이라도 무너지면 그 축의 리스크는 실재하므로 관대한 평균으로 덮지 않는다). 격자 2쌍: (entry_threshold_apr, exit_threshold_ratio), (top_k_pairs, leg_fraction).
2. **시간 안정성** (`rolling.py`): 2019-09~데이터 끝까지 **비중첩** 6개월 구간(중첩 윈도는 승률 표본을 인위적으로 부풀림). 구간별 G1 vs I5 CAGR, 전체/IS전용/OOS포함/전반부/후반부 승률.
3. **레짐 분해** (`regime.py`): 고펀딩 {2020,2021,2024} vs 저펀딩 {2022,2023,2025,2026}(과제 지정). 연도별 anchored annualized return은 `gates21.yearly_annualized_returns` 재사용.
4. **DSR 재계산** (`dsr.py`): trials = `gates21.PRIOR_CUMULATIVE_TRIALS`(121) + `gates21.GA_TRIALS`(7,500) = 7,621. G1 자체 equity로 재계산(wave21_report.md의 0.23196은 top_k=3 원본의 수치이므로 그대로 재사용하지 않음). wave22 자체 진단 평가(~150회)는 trials에 불포함 -- 선택 탐색이 아닌 고정 후보 감사이므로.
5. **유전자 기여도 분해** (`attribution.py`): I5->G1 one-at-a-time(정방향) + G1->I5 leave-one-out(역방향). 집중 판정: 양의 기여분 합 대비 최대축 비중 >60% = 단일축집중, <=60%이고 2개 이상 축이 각 >=15%면 2축이상분산.
6. **거짓 발견 대조** (`shuffle_control.py`): `genome.random_genome`과 동일분포로 30개 무작위 유전자(gross<=1x 재추첨 조건, G1과 동일 사이징 제약). 전기간 CAGR 기준 G1의 백분위.

## 판정 기준 (과제 원문, 사전등록)
- **PASS**: 안정성비율(최솟값) >=0.8 AND 롤링승률 >=55% AND 개선분 2축이상분산 AND DSR(누적)>0 AND 무작위대조 상위5%이내
- **FAIL**: 안정성비율 <0.6 OR 개선분 단일축집중 OR 롤링승률 <50% (하나라도 해당하면 다른 기준과 무관하게 FAIL)
- **CONDITIONAL**: 위 두 경우가 아니고 PASS 5개 기준을 전부 충족하지 못함 -- "paper 전진검증 필수"
- FAIL 우선순위 원칙: FAIL 조건이 하나라도 성립하면 PASS 조건 충족 여부와 무관하게 FAIL로 확정한다 (`verdict.py` 참조).

## 정직성 원칙
- G1은 사용자가 승격시킨 구성이며 이 wave는 그것을 봐주지 않는다. 과최적화면 FAIL로 쓰고 승격 철회를 권고한다.
- 각 검증 결과 JSON/리포트 절에 표본수·구간수 한계를 명시한다.
- `research/wave22_overfit/` 밖 수정 금지, `research/wave4_leverage/` 접근 금지.
