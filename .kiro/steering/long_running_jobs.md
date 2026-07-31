# 장시간 실행 작업 규칙 (command failed / timed out 재발 방지)

이 레포의 백테스트는 한 번에 수백 초가 걸린다. 셸 호출은 그보다 먼저 중단되므로, **작업을 짧게 쪼개는 것이 아니라 스크립트가 중단을 견디도록 설계**해야 한다. 아래는 실제로 반복 실패한 뒤 확정한 규칙이다.

## 절대 규칙

1. **셸 호출 1회 = 작업 1개.** 여러 무거운 작업을 `&&` 나 루프로 한 호출에 묶지 않는다. 실패 사례: `capital_probe.py`가 160초짜리 구간 3개를 한 호출에서 돌려 두 번 연속 중단됐다.
2. **호출당 실측 4분(240초) 이내로 설계한다.** `timeout` 값도 그에 맞춘다. 8분은 상한이지 목표가 아니다.
3. **백그라운드 프로세스에 의존하지 않는다.** 툴 호출 사이에 프로세스가 유지되지 않으므로 `run_in_background`로 띄운 작업은 다음 호출에서 사라진다.

## 장시간 스크립트 설계 패턴 (필수)

무거운 반복 작업을 쓸 때는 **세 가지를 반드시** 넣는다.

1. **증분 저장** — 한 단위가 끝날 때마다 결과 JSON을 쓴다. 중단되어도 완료분이 남는다.
2. **캐시 재사용** — 시작 시 기존 JSON을 읽어 완료된 단위는 건너뛰고 `(캐시)`로 표시한다.
3. **단일 단위 실행 플래그** — `--only <값>` 으로 한 단위만 돌리고 종료할 수 있게 한다. 남은 단위를 안내 출력한다.

`research/wave37_walkforward/capital_probe.py`가 이 패턴의 참조 구현이다.

```bash
V=/projects/sandbox/.venv30/bin/python
$V research/wave37_walkforward/capital_probe.py --only 100     # 한 구간만 (약 180s)
$V research/wave37_walkforward/capital_probe.py --only 1000    # 이어서
$V research/wave37_walkforward/capital_probe.py                # 남은 것만 채우고 최종 해석 출력
```

## 파이썬 환경

- **반드시** `/projects/sandbox/.venv30/bin/python` (3.11.15, numpy 2.4.6, pandas 3.0.5)
- 시스템 `python3`은 3.9라서 이 레포 코드가 실행되지 않는다
- Python 3.11은 f-string 안에서 **같은 종류의 인용부호 중첩을 지원하지 않는다.** `f"{', '.join(f'${r['capital']}')}"` 형태는 SyntaxError다. `"{:,.0f}".format(...)` 로 분리하거나 바깥/안쪽 인용부호를 다르게 쓴다.

## 실행 시간 참고치 (측정값)

| 작업 | 시간 |
|---|---|
| `run_wave37.py` (인과 워크포워드 + 비용×3) | 약 320s — **경계선, 단독 호출로만** |
| `capital_probe.py` 자본 1구간 | 약 180s |
| `dataio37.py` 패널 구축 (273종목) | 약 60s |
| `pytest research/wave30_qd/tests/test_wave30.py` | 약 25s |
| 엔진 1조합 평가 | 약 36ms |

## 네트워크

- Binance API = **HTTP 451 차단**. 단 `research/wave3/cache`의 로컬 캐시는 온전하므로 재수집이 불필요하다.
- Bitget / OKX = 200 정상. Bybit = 403.
