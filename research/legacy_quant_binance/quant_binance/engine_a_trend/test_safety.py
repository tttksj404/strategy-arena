#!/usr/bin/env python3
"""실행층 안전장치 강제발동 + 회계 정확성 테스트."""
import os, json, time
import numpy as np
from executor import Executor, LEV_CAP, DRY_RUN

R={}
def chk(name,cond,detail=""): R[name]={"PASS":bool(cond),"detail":detail}

# --- A. kill_switch가 목표 익스포저를 강제 현금화하는가 ---
ks="/tmp/test_ks.json"; json.dump({"armed":True,"reasons":["TEST"]},open(ks,"w"))
st="/tmp/test_state_a.json"; open(st,"w").close(); os.remove(st)
ex=Executor(st, init_equity=100)
prices={"BTC":50000.0,"ETH":3000.0}
# 엔진이 풀 익스포저(각 $75) 원해도 ks armed면 현금화 돼야
rec=ex.step("2026-06-19", {"BTC":37.5,"ETH":37.5}, prices, ks_path=ks)
chk("A_killswitch_forces_cash", rec["exposure_pct"]==0 and rec["halted"] and rec["reason"]=="KILL_SWITCH", f"exp={rec['exposure_pct']} reason={rec['reason']}")
os.remove(ks)

# --- B. 일일손실 -15% 초과 시 회로차단 ---
st="/tmp/test_state_b.json";
for p in [st]:
    if os.path.exists(p): os.remove(p)
ex=Executor(st, init_equity=100)
# day1: 풀 롱 진입 (1.5x = $150 익스포저)
ex.step("2026-06-19", {"BTC":37.5,"ETH":37.5}, {"BTC":50000.0,"ETH":3000.0})
# 같은 날 -11% 하락 -> 1.5x면 equity -16.5% -> 회로차단
crash={"BTC":50000*0.79,"ETH":3000*0.79}   # -21%: 0.75x 노출서 equity -15.75% → 일일손실 회로차단 발동
rec=ex.step("2026-06-19", {"BTC":37.5,"ETH":37.5}, crash)
chk("B_daily_loss_breaker", rec["halted"] and "DAILY_LOSS" in str(rec["reason"]) and rec["exposure_pct"]==0, f"reason={rec['reason']} exp={rec['exposure_pct']}")

# --- C. 회계 정확성: executor equity vs 순수계산 (동일 가격경로) ---
st="/tmp/test_state_c.json"
if os.path.exists(st): os.remove(st)
ex=Executor(st, init_equity=1000)
# 합성: day0 진입 BTC $500(1x), 가격 +10% -> equity = 1000 - fee + 500*0.10
p0={"BTC":100.0,"ETH":100.0}
ex.step("d0", {"BTC":500.0,"ETH":0.0}, p0)              # BTC $500 매수
fee0=500*0.0006
eq_after_entry=ex.equity(p0)
p1={"BTC":110.0,"ETH":100.0}                            # BTC +10%
eq1=ex.equity(p1)
expected=1000 - fee0 + 500*0.10                          # 자본 - 진입수수료 + 포지션수익
chk("C_accounting", abs(eq1-expected)<0.01, f"exec={eq1:.4f} expected={expected:.4f}")

# --- D. 숏 금지(음수 target 클램프) ---
st="/tmp/test_state_d.json"
if os.path.exists(st): os.remove(st)
ex=Executor(st, init_equity=100)
rec=ex.step("d0", {"BTC":-50.0,"ETH":0.0}, {"BTC":100.0,"ETH":100.0})   # 음수 목표
chk("D_no_short", ex.s["positions"]["BTC"]>=0, f"btc_qty={ex.s['positions']['BTC']}")

# --- E. min-notional 잔돈 churn 방지 ---
st="/tmp/test_state_e.json"
if os.path.exists(st): os.remove(st)
ex=Executor(st, init_equity=100)
rec=ex.step("d0", {"BTC":3.0,"ETH":0.0}, {"BTC":100.0,"ETH":100.0})     # $3 < min $5
chk("E_min_notional_skip", len(rec["orders"])==0, f"orders={len(rec['orders'])}")

npass=sum(1 for v in R.values() if v["PASS"])
print(f"SAFETY {npass}/{len(R)} PASS  (DRY_RUN={DRY_RUN})")
for k,v in R.items(): print(f"  {'✓' if v['PASS'] else '✗'} {k}: {v['detail']}")
json.dump(R,open("safety_results.json","w"),indent=2,ensure_ascii=False)
