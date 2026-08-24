#!/usr/bin/env python3
"""프로덕션 결정모듈(trend_engine)이 검증 백테스트를 재현하는지 확인.
각 날 t에 target_exposure(closes[:t+1])만 호출 -> 구조적 룩어헤드 불가.
기대: OOS(2024+) CAGR~29% Calmar~0.95 재현."""
import urllib.request, json, time, math
import numpy as np
from trend_engine import target_exposure, ASSETS

def fetch(sym):
    out=[];start=1502928000000
    while True:
        u=f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&startTime={start}&limit=1000"
        d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=30))
        if not d:break
        out+=d
        if len(d)<1000:break
        start=d[-1][0]+86400000;time.sleep(0.12)
    ts=np.array([int(x[0]) for x in out]);c=np.array([float(x[4]) for x in out])
    _,i=np.unique(ts,return_index=True);return ts[i],c[i]

FEE=0.0010
def met(r,ts):
    eq=np.cumprod(1+r);yrs=(ts[-1]-ts[0])/1000/86400/365.25
    cagr=eq[-1]**(1/yrs)-1 if eq[-1]>0 else -1
    peak=np.maximum.accumulate(eq);mdd=np.max((peak-eq)/peak)
    sh=r[r!=0].mean()/r[r!=0].std()*math.sqrt(365) if r[r!=0].std()>0 else 0
    return dict(cagr=round(cagr*100,1),mdd=round(mdd*100,1),sharpe=round(sh,2),calmar=round(cagr/mdd,2) if mdd>0 else 0)

data={a:fetch(b) for a,b in [('BTC','BTCUSDT'),('ETH','ETHUSDT')]}
# build per-asset strategy daily returns using ONLY past closes at each step
series=[]
for a in ASSETS:
    ts,c=data[a]; ret=c[1:]/c[:-1]-1
    pos=np.zeros(len(ret))
    for i in range(len(ret)):
        # decision at close index i (uses closes[:i+1]); applied to ret[i]=c[i+1]/c[i]
        exp=target_exposure(c[:i+1])
        pos[i]=0.5*exp                      # 50% weight
    sw=np.abs(np.diff(np.concatenate([[0.0],pos])))
    series.append((pos*ret - sw*FEE, ts[1:]))
m=min(len(series[0][0]),len(series[1][0]))
port=series[0][0][-m:]+series[1][0][-m:]; tsr=series[0][1][-m:]
cut=__import__('datetime').datetime(2024,1,1).timestamp()*1000
print(json.dumps({
  'ALL':met(port,tsr),
  'OOS2024':met(port[tsr>=cut],tsr[tsr>=cut]),
  'expected_OOS':'cagr~29.2 mdd~30.7 calmar~0.95 (검증값)'
},indent=2))
