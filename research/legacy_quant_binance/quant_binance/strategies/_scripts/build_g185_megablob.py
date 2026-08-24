"""Build single-paste megablob to deploy G185 emulator on Oracle instance via Cloud Shell."""
import base64, gzip, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "quant_binance" / "strategies" / "_scripts"

# 1) Linux-friendly emulator (paths under ~/g185)
EMULATOR_LINUX = r'''"""G185 paper-live emulator (Linux)."""
import json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HOME = Path.home()
APP = HOME / "g185"
sys.path.insert(0, str(APP))
from g002_mingogogo_ch1_backtest import compute_ch1_score, atr_pct  # type: ignore

OUT = APP / "runtime"; OUT.mkdir(parents=True, exist_ok=True)
TRADES = OUT / "trades.jsonl"
STATE  = OUT / "state.json"
LOG    = OUT / "emulator.log"

UNIVERSE = ["DOGEUSDT","PEPEUSDT","WIFUSDT","ARBUSDT","OPUSDT","AVAXUSDT","SUIUSDT","ADAUSDT",
            "APTUSDT","BNBUSDT","DOTUSDT","LINKUSDT","LTCUSDT","NEARUSDT","SOLUSDT","UNIUSDT","XRPUSDT","BTCUSDT"]
EQUITY_USD=100.0; SIZE_PCT=0.40; LEVERAGE=5.0; THRESHOLD=80; HOLD_BARS=24
MAX_CONC=5; ATR_GUARD_PCT=8.0; COST_BPS_RT=16.0; CYCLE_SECONDS=300
BINANCE_KLINES="https://api.binance.com/api/v3/klines"

def log(m):
    line=f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f: f.write(line+"\n")

def load_state():
    if STATE.exists():
        try: return json.loads(STATE.read_text(encoding="utf-8"))
        except: pass
    return {"started_at": datetime.now(timezone.utc).isoformat(),"open_positions":{},"closed_count":0,"wins":0,"losses":0,"cumulative_pnl_usd":0.0,"last_cycle":None}

def save_state(s):
    s["last_cycle"]=datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(s,indent=2,ensure_ascii=False), encoding="utf-8")

def append_trade(e):
    with TRADES.open("a", encoding="utf-8") as f: f.write(json.dumps(e,ensure_ascii=False)+"\n")

def fetch_klines(symbol, limit=200):
    url=f"{BINANCE_KLINES}?symbol={symbol}&interval=1h&limit={limit}"
    try:
        req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            chunk=json.loads(r.read())
    except Exception as e:
        log(f"  fetch {symbol} FAIL: {e}"); return None
    if not chunk: return None
    bars=[]
    for b in chunk[:-1]:
        bars.append({"open_time":b[0],"open_price":float(b[1]),"high_price":float(b[2]),"low_price":float(b[3]),
                     "close_price":float(b[4]),"base_volume":float(b[5]),"quote_volume":float(b[7])})
    return pd.DataFrame(bars)

def evaluate(symbol):
    df=fetch_klines(symbol)
    if df is None or len(df)<100: return None
    score,_=compute_ch1_score(df)
    atrp=atr_pct(df["high_price"],df["low_price"],df["close_price"],14)
    return {"symbol":symbol,"score":float(score.iloc[-1]) if not pd.isna(score.iloc[-1]) else 0.0,
            "atr_pct":float(atrp.iloc[-1]) if not pd.isna(atrp.iloc[-1]) else 0.0,
            "close":float(df["close_price"].iloc[-1]),"bar_ts":int(df["open_time"].iloc[-1])}

def cycle(state):
    now_ms=int(time.time()*1000)
    open_pos=state["open_positions"]
    expired=[s for s,p in open_pos.items() if now_ms >= p["entry_ts_ms"]+HOLD_BARS*3600*1000]
    for sym in expired:
        pos=open_pos.pop(sym)
        df=fetch_klines(sym, limit=2)
        if df is None:
            log(f"  EXIT {sym} fetch fail keeping open"); open_pos[sym]=pos; continue
        ec=float(df["close_price"].iloc[-1])
        gp=(ec/pos["entry_close"]-1)*100
        npp=(gp/100 - COST_BPS_RT/10000)*LEVERAGE
        if npp < -0.90: npp=-0.90
        pnl=round(EQUITY_USD*SIZE_PCT*npp, 4)
        state["cumulative_pnl_usd"]=round(state["cumulative_pnl_usd"]+pnl, 4)
        state["closed_count"]+=1
        if pnl>0: state["wins"]+=1
        else: state["losses"]+=1
        append_trade({"type":"EXIT","symbol":sym,"entry_ts_ms":pos["entry_ts_ms"],"entry_close":pos["entry_close"],
                      "exit_ts_ms":now_ms,"exit_close":ec,"gross_pct":round(gp,4),"net_pct_levered":round(npp,4),
                      "pnl_usd":pnl,"score_at_entry":pos.get("score"),"atr_pct_at_entry":pos.get("atr_pct")})
        log(f"  EXIT  {sym}: {gp:+.2f}% gross / {npp*100:+.2f}% lev / PnL ${pnl:+.2f}")

    cands=[]
    for sym in UNIVERSE:
        if sym in open_pos: continue
        info=evaluate(sym)
        if info: cands.append(info)
        time.sleep(0.05)
    cands.sort(key=lambda x:x["score"], reverse=True)

    for info in cands:
        sym=info["symbol"]
        if len(open_pos)>=MAX_CONC: break
        if info["score"]<THRESHOLD: continue
        if info["atr_pct"]>ATR_GUARD_PCT:
            log(f"  SKIP {sym}: score {info['score']:.1f} but ATR {info['atr_pct']:.2f}% > {ATR_GUARD_PCT}%"); continue
        pos={"entry_ts_ms":now_ms,"entry_close":info["close"],"score":info["score"],"atr_pct":info["atr_pct"]}
        open_pos[sym]=pos
        append_trade({"type":"ENTRY","symbol":sym,"entry_ts_ms":now_ms,"entry_close":info["close"],
                      "score":info["score"],"atr_pct":info["atr_pct"],
                      "size_usd_margin":EQUITY_USD*SIZE_PCT,"notional_usd":EQUITY_USD*SIZE_PCT*LEVERAGE})
        log(f"  ENTRY {sym}: score {info['score']:.1f} ATR {info['atr_pct']:.2f}% close {info['close']:.6f}")

    state["open_positions"]=open_pos
    save_state(state)
    top5=[(c["symbol"], round(c["score"],1)) for c in cands[:5]]
    log(f"  HEARTBEAT open={len(open_pos)} closed={state['closed_count']} W/L={state['wins']}/{state['losses']} cumPnL=${state['cumulative_pnl_usd']:+.2f} top5={top5}")

def main():
    log(f"=== G185 emulator START === eq=${EQUITY_USD} size={SIZE_PCT} lev={LEVERAGE}x thr={THRESHOLD} hold={HOLD_BARS}h max_conc={MAX_CONC} atr_guard={ATR_GUARD_PCT}%")
    log(f"universe ({len(UNIVERSE)}): {','.join(UNIVERSE)}")
    state=load_state()
    while True:
        try: cycle(state)
        except KeyboardInterrupt: log("STOP"); break
        except Exception as e: log(f"CYCLE_ERROR: {type(e).__name__}: {e}")
        time.sleep(CYCLE_SECONDS)

if __name__=="__main__": main()
'''

g002_text = (SCRIPTS / "g002_mingogogo_ch1_backtest.py").read_bytes()

em64 = base64.b64encode(gzip.compress(EMULATOR_LINUX.encode("utf-8"))).decode()
g264 = base64.b64encode(gzip.compress(g002_text)).decode()

REMOTE_SH = r'''
set -e
echo "=== installing python deps ==="
command -v python3 >/dev/null || sudo dnf install -y python3
python3 -m pip install --user --quiet pandas numpy 2>/dev/null || { python3 -m ensurepip --user; python3 -m pip install --user --quiet pandas numpy; }

echo "=== creating systemd user service ==="
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/g185-emulator.service << UNIT
[Unit]
Description=G185 paper-live emulator
After=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/g185
ExecStart=/usr/bin/python3 %h/g185/g185_emulator.py
Restart=always
RestartSec=30
StandardOutput=append:%h/g185/runtime/stdout.log
StandardError=append:%h/g185/runtime/stderr.log

[Install]
WantedBy=default.target
UNIT
echo "=== enable lingering ==="
sudo loginctl enable-linger "$USER"
echo "=== start emulator service ==="
systemctl --user daemon-reload
systemctl --user enable --now g185-emulator.service
sleep 5

echo "=== adding SSH Port 443 for outside-firewall access ==="
if ! sudo grep -q "^Port 443" /etc/ssh/sshd_config; then
    sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
    if grep -q "^Port " /etc/ssh/sshd_config; then
        echo "Port 22 already explicit"
    else
        echo "Port 22" | sudo tee -a /etc/ssh/sshd_config >/dev/null
    fi
    echo "Port 443" | sudo tee -a /etc/ssh/sshd_config >/dev/null
fi

echo "=== firewalld open 443 ==="
sudo firewall-cmd --permanent --add-port=443/tcp || true
sudo firewall-cmd --reload || true

echo "=== SELinux allow ssh on 443 ==="
sudo semanage port -a -t ssh_port_t -p tcp 443 2>/dev/null || sudo semanage port -m -t ssh_port_t -p tcp 443 2>/dev/null || true

echo "=== iptables allow 443 (in case firewalld inactive) ==="
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || true

echo "=== restart sshd ==="
sudo systemctl restart sshd

echo "=== verify sshd listens on 443 ==="
sleep 2
sudo ss -tlnp | grep -E ":(22|443)" || true

echo "=== STATUS emulator ==="
systemctl --user status g185-emulator.service --no-pager | head -15
echo "=== LOG ==="
tail -20 ~/g185/runtime/emulator.log 2>/dev/null || echo "(log not yet written)"
'''

OCI_SECLIST_SH = r'''
echo "=== adding Security List ingress rule for TCP 443 (via OCI CLI) ==="
TENANCY=$(oci iam compartment list --query 'data[0]."compartment-id"' --raw-output 2>/dev/null)
[ -z "$TENANCY" ] && TENANCY=$OCI_CLI_TENANCY
echo "tenancy: $TENANCY"
VCN_ID=$(oci network vcn list --compartment-id "$TENANCY" --query 'data[?"display-name"==`g185-vcn`].id | [0]' --raw-output 2>/dev/null)
echo "vcn-id: $VCN_ID"
SL_ID=$(oci network security-list list --vcn-id "$VCN_ID" --compartment-id "$TENANCY" --query 'data[0].id' --raw-output 2>/dev/null)
echo "seclist-id: $SL_ID"
if [ -n "$SL_ID" ]; then
    CURRENT=$(oci network security-list get --security-list-id "$SL_ID" --query 'data."ingress-security-rules"' 2>/dev/null)
    if echo "$CURRENT" | grep -q '"min": 443'; then
        echo "443 ingress already present"
    else
        NEW=$(echo "$CURRENT" | python3 -c "import json,sys; d=json.load(sys.stdin); d.append({'protocol':'6','source':'0.0.0.0/0','source-type':'CIDR_BLOCK','is-stateless':False,'tcp-options':{'destination-port-range':{'min':443,'max':443}}}); print(json.dumps(d))")
        oci network security-list update --security-list-id "$SL_ID" --ingress-security-rules "$NEW" --force 2>&1 | head -10
        echo "443 ingress rule added"
    fi
fi
'''

# Build the megablob
out = []
out.append("#!/bin/bash")
out.append("# G185 deploy megablob — paste in Cloud Shell. SSH alias 'g185' must work first (set up earlier).")
out.append("set -e")
out.append("")
out.append(f"EM64='{em64}'")
out.append(f"G264='{g264}'")
out.append("")
out.append('echo "=== preparing remote dirs ==="')
out.append('ssh g185 "mkdir -p ~/g185 ~/g185/runtime"')
out.append("")
out.append('echo "=== uploading emulator + dependency ==="')
out.append('echo "$EM64" | ssh g185 "base64 -d | gunzip > ~/g185/g185_emulator.py"')
out.append('echo "$G264" | ssh g185 "base64 -d | gunzip > ~/g185/g002_mingogogo_ch1_backtest.py"')
out.append('ssh g185 "ls -la ~/g185/"')
out.append("")
out.append('echo "=== running remote setup (emulator + ssh port 443) ==="')
out.append('ssh g185 bash << \'REMOTE_END\'')
out.append(REMOTE_SH.strip())
out.append('REMOTE_END')
out.append("")
out.append(OCI_SECLIST_SH.strip())
out.append("")
out.append('echo')
out.append('echo "=================================================="')
out.append('echo "G185 emulator + SSH Port 443 deployed."')
out.append('echo "Test from outside SSAFY firewall:"')
out.append('echo "  ssh -p 443 opc@140.245.66.2"')
out.append('echo "=================================================="')

text = "\n".join(out) + "\n"
out_path = SCRIPTS / "g185_deploy_megablob.sh"
out_path.write_text(text, encoding="utf-8", newline="\n")
print(f"wrote {out_path} ({len(text)} bytes)")
