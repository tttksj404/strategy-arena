#!/usr/bin/env python3
"""Bold-play 정직한 기댓값 계산 (Dubins-Savage).
엣지 없는 게임에서 목표도달확률 최대화 = 최소 베팅수(bold). 거버너 규율과 함께 쓴다.
PEPE 실증: 1회 더블 승률 p~0.49(비용후). 반복/버티기=100%파산이므로 '원샷+출금'만."""
import sys, math
def bold_reach(D, target, p=0.49):
    """disposable D -> 목표 target 도달확률(연속 더블 bold). 정직: EV<0."""
    if target<=D: return 1.0
    nd = math.log2(target/D)
    return p**nd
def report(D=7.0):
    p_list=[0.49,0.47,0.45]
    print(f"링펜스 ${D} bold-play 목표도달확률(정직, EV<0):")
    print("목표 | "+" | ".join(f"p={p}" for p in p_list))
    for mult in [2,3,5,10,30]:
        T=D*mult
        print(f"{mult:>2}x(${T:>5.0f}) | "+" | ".join(f"{bold_reach(D,T,p)*100:5.1f}%" for p in p_list))
    print()
    print("핵심: 도달확률 ~ D/목표 (선형). 비용 빼면 기대값 음수.")
    print("거버너 규율로 '손실=고정 disposable, 수익=banked 출금'으로만 한정 →")
    print("PEPE식 100% 파산(반복/버티기)을 '잃어도 되는 돈만 잃는 오락'으로 전환.")
    print("절대 금지: 버티기(무손절), 수익 재투입(compound), 정산후 자동 재진입.")
if __name__=="__main__":
    report(float(sys.argv[1]) if len(sys.argv)>1 else 7.0)
