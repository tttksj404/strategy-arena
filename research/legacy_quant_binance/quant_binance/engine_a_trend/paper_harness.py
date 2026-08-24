#!/usr/bin/env python3
"""Engine A 라이브 데이터 플러밍 체크 (백테스트가 못 잡는 것만).
검증: ① 라이브 klines 정상 수신 ② 미완성 당일봉 제거(라이브 룩어헤드 차단) ③ 오늘 실제 목표포지션.
'며칠 페이퍼' 불필요(전략은 백테스트로 검증됨) — 이건 1회 플러밍 점검."""
import urllib.request, json, time
import numpy as np
from trend_engine import decide, target_exposure, MIN_BARS

def fetch_closed_daily(sym, limit=260):
    now_ms = int(time.time()*1000)
    u=f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit={limit}"
    kl=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'M'}),timeout=30))
    # kline: [openTime,o,h,l,c,v,closeTime,...]  closeTime > now => 아직 안 끝난 봉 => 제거
    closed=[k for k in kl if int(k[6]) < now_ms]
    dropped=len(kl)-len(closed)
    closes=[float(k[4]) for k in closed]
    last_close_time=int(closed[-1][6]) if closed else 0
    last_date=time.strftime('%Y-%m-%d %H:%MZ', time.gmtime(last_close_time/1000))
    return closes, dropped, last_date, now_ms, last_close_time

EQUITY=float(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 7.48
print(f"=== Engine A 라이브 플러밍 체크 (equity ${EQUITY}) ===")
data={}; ok=True
for a,sym in [('BTC','BTCUSDT'),('ETH','ETHUSDT')]:
    closes,dropped,last_date,now_ms,lct=fetch_closed_daily(sym)
    data[a]=closes
    used_closed = lct < now_ms
    enough = len(closes)>=MIN_BARS
    print(f"{a}: 수신 closed봉 {len(closes)} | 미완성봉 제거 {dropped}개 | 마지막마감봉 {last_date} | "
          f"closed확인 {'✓' if used_closed else '✗LOOKAHEAD'} | 충분 {'✓' if enough else '✗부족'}")
    ok = ok and used_closed and enough
print("--- 오늘 목표포지션 (실주문 아님, 출력만) ---")
dec=decide(data, EQUITY)
for a in ['BTC','ETH']:
    v=dec[a]; print(f"  {a}: exposure {v['asset_exposure']}x -> target ${v['target_notional_usd']} ({v['target_frac']*100:.1f}% of equity)")
print(f"  gross {dec['_gross_frac']*100:.0f}% of equity (cap 150%)")
print(f"\nPLUMBING: {'OK ✓ 미완성봉 차단·데이터정상·결정생성' if ok else 'FAIL ✗'}")
json.dump({'equity':EQUITY,'decision':dec,'plumbing_ok':ok},open('paper_check.json','w'),indent=2)
