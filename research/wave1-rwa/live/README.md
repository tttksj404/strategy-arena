# 라이브 실행기 (부스터, Bitget USDT-M)

국면전환 크로스섹션 부스터의 실주문 실행기. **기본 DRY_RUN**(주문 안 냄).

## 안전장치 (전부 negative 테스트 통과)
- **DRY_RUN 기본** — `LIVE_DRY_RUN=0` 명시해야만 실주문 경로 진입
- **라이브 주문 경로 gate** — `_execute_reconcile`는 NotImplementedError로 봉인(DRY_RUN 패리티 확인 + 입금 후에만 구현)
- **invariant 3종** (매 틱 강제): lev ≤ MAX_LEV / 델타중립(net≈0) / gross ≤ 2.0
- **회로차단기**: 트레일링 낙폭 ≥ LIVE_DD_HALT(기본 45%) → kill_switch + flatten
- invariant 위반 시 즉시 kill_switch (loop 종료)

## 실행
```bash
# DRY_RUN (주문 없이 타깃·주문계획 확인)
LIVE_DRY_RUN=1 LIVE_CAPITAL_USD=100 python3 -m live.live_booster

# 실전 (입금 + DRY_RUN 패리티 확인 후에만):
#   1) 계좌 입금  2) _execute_reconcile 구현  3) LIVE_DRY_RUN=0
```

## ENV (리포에 절대 커밋 금지)
BITGET_API_KEY / BITGET_API_SECRET / BITGET_API_PASSPHRASE
LIVE_CAPITAL_USD(기본 100) / LIVE_DRY_RUN(기본 1) / LIVE_MAX_LEV(기본 2.3) / LIVE_DD_HALT(기본 0.45)

## 남은 단계 (실전 이행 전)
1. Bitget 계좌 입금 + API 키 발급(선물 거래 권한, 출금 권한 OFF)
2. DRY_RUN 며칠 관찰 → paper3와 타깃 일치 확인
3. `_execute_reconcile` 구현 (reduce-only 청산 + market 진입) + 재-스모크
4. 오라클 systemd 배포 (맥 독립) — 헌법 oracle_deploy.sh
5. 소액 실투입 → 첫 리밸런스 실체결 확인 → 단계적 증액
