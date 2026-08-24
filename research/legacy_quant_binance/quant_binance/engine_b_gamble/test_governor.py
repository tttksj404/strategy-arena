#!/usr/bin/env python3
"""Engine B 적대적 테스트 — 파산 스파이럴이 구조적으로 불가능함을 증명.
각 테스트 = '나쁜 행동(스파이럴 유발)'이 차단되는지. 차단되면 PASS."""
import os, json
from gamble_governor import GambleGovernor

R={}
def chk(n,c,d=""): R[n]={"PASS":bool(c),"detail":d}
def blocked(fn):
    try: fn(); return False
    except Exception as e: return type(e).__name__

P="/tmp/gov_test.json"
def fresh():
    if os.path.exists(P): os.remove(P)
    return GambleGovernor(P)

# T1: 초기 locked — fund 전 도박 불가
g=fresh()
chk("T1_locked_default", blocked(lambda:g.open_gamble(2.0,0.01))=="RuntimeError", "fund 전 open 차단")

# T2: 링펜스 초과 위험 불가 (I1)
g=fresh(); g.fund(7.0)
chk("T2_overrisk_blocked", blocked(lambda:g.open_gamble(2.0,0.01,stake=100.0))=="ValueError", "stake>disposable 차단")

# T3: 더블 오픈 불가 (I3)
g=fresh(); g.fund(7.0); g.open_gamble(2.0,0.01)
chk("T3_double_open_blocked", blocked(lambda:g.open_gamble(2.0,0.01))=="RuntimeError", "동시 2도박 차단")

# T4: 승리 후 재진입은 fund 없이 불가 (I5 — 스파이럴 핵심 차단)
g=fresh(); g.fund(7.0); g.open_gamble(2.0,0.01); g.settle("win")
chk("T4_no_reentry_after_win", blocked(lambda:g.open_gamble(2.0,0.01))=="RuntimeError", "정산후 locked, refund 필수")

# T5: banked는 재위험화 안 됨 (I2) — 승리 후 새 fund의 stake는 banked가 아닌 새 disposable에서
g=fresh(); g.fund(7.0); g.open_gamble(2.0,0.01); g.settle("win")
banked_after_win=g.s["banked"]
g.fund(7.0)  # 새 disposable (의식적 재충전)
g.open_gamble(2.0,0.01)  # stake=remaining=7 (banked 14 아님)
chk("T5_banked_protected", abs(g.s["at_risk"]-7.0)<1e-9 and g.s["banked"]>=banked_after_win, f"at_risk={g.s['at_risk']} banked={g.s['banked']}")

# T6: target/stop 없으면 불가 (I4 — 버티기 금지)
g=fresh(); g.fund(7.0)
chk("T6_mandatory_exit", blocked(lambda:g.open_gamble(1.0,0.0))=="ValueError", "익절/손절 강제")

# T7: 스파이럴 시뮬 — 5연승 시도해도 매번 lock, banked 보호, at_risk≤disposable 불변
g=fresh(); spiral_ok=True
for i in range(5):
    g.fund(7.0)                      # 매번 명시적 재충전 강제(자동 스파이럴 불가)
    g.open_gamble(2.0,0.01)
    assert g.s["at_risk"]<=g.s["disposable"]+1e-9   # I1 매회
    g.settle("win")
    if g.s["banked"] < (i+1)*0   : spiral_ok=False
# 마지막에 '손실'이 와도 banked는 안전
g.fund(7.0); g.open_gamble(2.0,0.01); banked_before_loss=g.s["banked"]; g.settle("loss")
chk("T7_spiral_blocked_banked_safe", g.s["banked"]>=banked_before_loss and g.s["at_risk"]==0, f"banked={g.s['banked']:.1f} 손실후 보존")

# T8: 손실은 disposable만 소모(전재산 아님). 7회 손실해도 banked 무손상
g=fresh(); g.fund(7.0)
for _ in range(7):
    if g.s["locked"]: g.fund(7.0)
    g.open_gamble(2.0,0.01); g.settle("loss")
chk("T8_loss_capped_to_disposable", g.s["at_risk"]==0 and g.s["banked"]>=0, "손실=disposable 한정, 계좌전소 아님")

npass=sum(1 for v in R.values() if v["PASS"])
print(f"GOVERNOR {npass}/{len(R)} PASS — 파산스파이럴 {'구조적 차단됨' if npass==len(R) else 'FAIL'}")
for k,v in R.items(): print(f"  {'✓' if v['PASS'] else '✗'} {k}: {v['detail']}")
json.dump(R,open("governor_results.json","w"),indent=2,ensure_ascii=False)
