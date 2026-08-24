#!/usr/bin/env python3
"""Engine B — 도박 거버너 (Gambling Governor).
엣지 없음(PEPE 2000/2000 파산 실증) → 알파 X. 목적 = 파산 스파이럴 구조적 차단.
규율 강제: 링펜스 + 원샷 bold-play + 강제 익절/손절 + 승리시 즉시 banked + 정산후 잠금.

불변식(런타임 assert):
  I1. at_risk <= disposable (링펜스 초과 위험 불가)
  I2. banked는 절대 재위험화 안 됨(수익 보호) & 감소 안 함
  I3. locked 또는 open 중엔 새 도박 불가(스파이럴 차단)
  I4. 모든 도박은 target+stop 필수(버티기 금지)
  I5. 정산되면 즉시 locked — refund(명시적 새 자금) 전까지 재진입 불가
"""
import os, json, time

class GambleGovernor:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.s = json.load(open(path))
        else:
            self.s = {"disposable":0.0,"remaining":0.0,"at_risk":0.0,"banked":0.0,
                      "open":None,"settled_count":0,"locked":True,"log":[]}

    # 링펜스 설정/재충전 — 사용자가 의식적으로 '잃어도 되는 돈'만 투입
    def fund(self, disposable):
        assert disposable >= 0
        if self.s["open"] is not None:
            raise RuntimeError("open gamble 중엔 refund 불가")
        self.s["disposable"] = float(disposable)
        self.s["remaining"] = float(disposable)
        self.s["locked"] = False
        self.s["log"].append({"ev":"fund","amt":disposable})
        self._check(); return True

    # 원샷 bold-play 진입. stake 미지정시 remaining 전액(bold). target/stop 필수.
    def open_gamble(self, target_mult, stop_mult, stake=None):
        if self.s["locked"]:
            raise RuntimeError("locked — fund() 후에만 가능(스파이럴 차단 I5)")
        if self.s["open"] is not None:
            raise RuntimeError("이미 open 도박 존재 (I3)")
        if not (target_mult and stop_mult) or target_mult<=1.0 or not (0<stop_mult<1.0):
            raise ValueError("target_mult>1, 0<stop_mult<1 필수 (I4: 버티기 금지)")
        stake = self.s["remaining"] if stake is None else float(stake)
        if stake <= 0 or stake > self.s["remaining"] + 1e-12:
            raise ValueError(f"stake {stake} > remaining {self.s['remaining']} (I1)")
        self.s["remaining"] -= stake
        self.s["at_risk"] = stake
        self.s["open"] = {"stake":stake,"target_mult":target_mult,"stop_mult":stop_mult}
        self.s["log"].append({"ev":"open","stake":stake,"tgt":target_mult,"stop":stop_mult})
        self._check(); return self.s["open"]

    # 정산: outcome 'win'/'loss'. win이면 stake*target_mult가 banked로(원금+수익 전액 인출).
    def settle(self, outcome):
        g = self.s["open"]
        if g is None: raise RuntimeError("open 도박 없음")
        if outcome == "win":
            payout = g["stake"] * g["target_mult"]
            self.s["banked"] += payout          # 원금+수익 전액 banked (재위험 불가 I2)
            result = {"outcome":"win","banked_add":payout}
        else:
            payout = g["stake"] * g["stop_mult"] # 손절 잔액(있으면) — bold full-loss면 stop_mult~0
            self.s["banked"] += payout
            result = {"outcome":"loss","recovered":payout,"lost":g["stake"]-payout}
        self.s["at_risk"] = 0.0
        self.s["open"] = None
        self.s["settled_count"] += 1
        self.s["locked"] = True                 # I5: 정산 즉시 잠금 — 재진입 차단
        self.s["log"].append({"ev":"settle", **result})
        self._check(); return result

    def status(self):
        return {k:self.s[k] for k in ("disposable","remaining","at_risk","banked","locked","settled_count")}

    def _check(self):
        s = self.s
        assert s["at_risk"] <= s["disposable"] + 1e-9, "I1 violated: at_risk > disposable"
        assert s["banked"] >= 0, "I2 violated: banked negative"
        assert s["remaining"] >= -1e-9, "remaining negative"
        if s["open"] is not None:
            assert s["open"]["target_mult"] > 1.0 and 0 < s["open"]["stop_mult"] < 1.0, "I4 violated"

    def save(self):
        json.dump(self.s, open(self.path,"w"), indent=2)
