# Wave-19 registry

| Candidate | Family | State | Verdict | 승격 | 분류 |
|---|---|---|---|---|---|
| signal_test (1단계) | wave19_regime | EVALUATED | FAIL | NO | 펀딩 스파이크 예측 신호 검정 |
| R0-R4 (2단계 순환 시스템) | wave19_regime | NOT_BUILT | N/A | N/A | 1단계 FAIL로 SPEC.md 절대 규칙에 따라 미착수 |

**판정**: 1단계(신호 검정) FAIL -- 정밀도는 base rate 대비 유의하게 높았으나(15.68% vs
14.00%, lift 1.12배) 경제적으로 결정적인 지표인 조건부 기대수익이 무작위(동일자-타심볼)
대조 대비 유의하게 **낮았다**(평균 -0.33%p, P(평균≤0)=99.64%, 7개 난수 시드 전부 동일
방향). SPEC.md 판정 기준("정밀도 유의" AND "조건부 기대수익 > 무작위")을 충족하지 못해
2단계(engine19.py/gates19.py/configs19.py, R0-R4)는 만들지 않았다.

**캐리(wave-18 I5, 전기간 CAGR 10.27%)가 이 wave가 확인할 수 있었던 상한.**

자세한 내용은 report/wave19_report.md, 원자료는 results/signal_report.json 참조.
