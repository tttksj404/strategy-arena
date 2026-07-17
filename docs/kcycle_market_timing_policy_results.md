# KCYCLE market timing policy experiment

status: waiting_for_outcome_linked_timed_snapshots
records: 14113
timed_with_outcome: 0
timed_without_outcome: 162

## Policy
- early: 30분 초과: 단승 5%, 삼쌍 축/강쏠림 미적용
- mid: 10~30분: 단승 15%, 삼쌍 축/강쏠림 미적용
- late: 10분 이내: 단승 30%, 삼쌍 축/강쏠림 적용 가능
- post_start: 시작 후: 예측 신호 차단

## Archive final-market proxy
- n=13889, exact=16.55%, top1=61.55%, strong_pull_n=2275, strong_pull_exact=35.16%, strong_pull_top1=81.32%

## Live timed phase metrics
- waiting for outcome-linked timed live snapshots

## Counts
- phase:early: 88
- phase:late: 29
- phase:mid: 1
- phase:post_start: 39
- phase:unknown: 5
- source:archive_import: 13951
- source:collector: 148
- source:live_decision: 14
