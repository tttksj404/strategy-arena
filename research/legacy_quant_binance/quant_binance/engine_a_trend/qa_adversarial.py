#!/usr/bin/env python3
"""Engine A 결정모듈 Adversarial QA — 11 attack vector (HIGH risk WRITE 게이트).
순수 stateless 결정함수라 일부 벡터(동시성·상태손상·라이프사이클)는 N/A 표기.
모든 출력은 finite & [0,CAP] & 하락장 0 이어야 함(불변식). nan/크래시/음수/초과 = FAIL."""
import numpy as np, math, json
from trend_engine import target_exposure, decide, LEV_CAP, MIN_BARS

def is_valid_exp(x):
    return isinstance(x,float) and math.isfinite(x) and 0.0-1e-12 <= x <= LEV_CAP+1e-12

up = list(np.linspace(100,300,300))      # 상승추세
dn = list(np.linspace(300,100,300))      # 하락추세
flat = [100.0]*300                        # 무변동(vol=0)
R={}
def t(name, fn):
    try:
        ok,detail = fn(); R[name] = {"PASS":bool(ok),"detail":detail}
    except Exception as e:
        R[name] = {"PASS":False,"detail":f"EXCEPTION {type(e).__name__}: {e}"}

# V2 외부데이터 결함: None/empty/NaN/malformed
t("V2a_empty", lambda:(target_exposure([])==0.0,"empty->0"))
t("V2b_None_in_data", lambda:(is_valid_exp(target_exposure(up[:-1]+[None])),"None last"))
t("V2c_NaN_in_data", lambda:(is_valid_exp(target_exposure(up[:-1]+[float('nan')])),"nan last"))
t("V2d_all_None", lambda:(target_exposure([None]*300)==0.0,"allNone->0"))
t("V2e_inf", lambda:(is_valid_exp(target_exposure(up[:-1]+[float('inf')])),"inf last"))
# V3 수치 경계: zero/negative/tiny/huge/정확히 SMA
t("V3a_zero_price", lambda:(is_valid_exp(target_exposure([0.0]*300)),"all zero"))
t("V3b_negative", lambda:(is_valid_exp(target_exposure([-5.0]*300)),"negatives"))
t("V3c_tiny", lambda:(is_valid_exp(target_exposure(list(np.linspace(1e-8,3e-8,300)))),"tiny prices"))
t("V3d_huge", lambda:(is_valid_exp(target_exposure(list(np.linspace(1e9,3e9,300)))),"huge prices"))
t("V3e_at_sma", lambda:(target_exposure([100.0]*300)==0.0,"flat==sma->cash(<=)"))
# V6 데이터 신선도/부족
t("V6a_short", lambda:(target_exposure(up[:MIN_BARS-1])==0.0,"<MIN_BARS->0"))
t("V6b_exact_min", lambda:(is_valid_exp(target_exposure(up[:MIN_BARS])),"==MIN_BARS valid"))
# V3f 무변동(vol=0) -> 0/0 가드
t("V3f_zerovol", lambda:(target_exposure(flat)==0.0,"vol=0->0 guard"))
# V8 mutator 가드: gross<=CAP 강제 (decide assert)
t("V8_gross_cap", lambda:(decide({"BTC":up,"ETH":up},100)["_gross_frac"]<=LEV_CAP+1e-9,"gross<=CAP"))
# V11 불변식: 하락장 절대 0(숏 없음) + 상승장 양수(sanity)
t("V11a_downtrend_cash", lambda:(target_exposure(dn)==0.0,"downtrend->0(no short)"))
t("V11b_uptrend_pos", lambda:(0<target_exposure(up)<=LEV_CAP,"uptrend->(0,CAP]"))
# V1/V4/V5/V7/V9/V10: 순수 stateless 함수 -> N/A
for na in ["V1_state_corrupt","V4_time_anomaly","V5_concurrency","V7_dedup","V9_cost_model","V10_lifecycle"]:
    R[na]={"PASS":True,"detail":"N/A (stateless pure fn)"}

npass=sum(1 for v in R.values() if v["PASS"]); ntot=len(R)
R["_SUMMARY"]={"pass":npass,"total":ntot,"verdict":"ALL_PASS" if npass==ntot else "FAIL"}
json.dump(R,open("qa_results.json","w"),indent=2,ensure_ascii=False)
fails=[k for k,v in R.items() if not k.startswith("_") and not v["PASS"]]
print(f"{npass}/{ntot} PASS")
if fails:
    print("FAILS:")
    for k in fails: print(" ",k,R[k]["detail"])
