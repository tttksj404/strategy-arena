# Wave-22 registry

| Candidate | Family | 종합판정 | 안정성비율 | 롤링승률 | 개선분집중축 | DSR(누적) | 무작위대조 상위% |
|---|---|---|---|---|---|---|---|
| G1 | wave21_ga (top_k 3->1 수정본) | CONDITIONAL | 0.852 | 57.1% | 윈도우(일) | -0.7597 | 3.2% |

**최종 판정**: CONDITIONAL -- G1 승격을 'paper 전진검증 필수' 조건부로 유지 -- 일부 기준만 충족(미충족: dsr_positive). 실거래 자본 투입 전 paper 전진검증(실시간, out-of-sample) 결과를 추가로 확인할 것.

근거: `research/wave22_overfit/report/wave22_report.md`
