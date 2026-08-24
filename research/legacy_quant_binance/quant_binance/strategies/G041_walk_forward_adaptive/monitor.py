"""G041 Paper-Live status dashboard. 한 번 실행하면 현재 상태 출력."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
STATE = ROOT / "paper_live_state.json"
TRADES = ROOT / "paper_live_trades.jsonl"


def main():
    if not STATE.exists():
        print("paper-live 미시작. python paper_live.py 한 번 실행 필요.")
        return
    s = json.loads(STATE.read_text())
    started = datetime.fromisoformat(s["started_at"])
    elapsed = datetime.now(timezone.utc) - started
    elapsed_d = elapsed.days
    elapsed_h = elapsed.seconds // 3600

    print(f"=== G041 Paper-Live Dashboard ===")
    print(f"  시작: {started.strftime('%Y-%m-%d %H:%M UTC')} ({elapsed_d}d {elapsed_h}h 경과)")
    print(f"  cycles run: {s['cycles_run']}")
    print(f"  equity: ${s['equity_usd']:.2f}  ({(s['equity_usd']-55)/55*100:+.1f}% from $55)")
    print(f"  warmup: {'완료' if s['warmup_complete'] else '진행 중 (30일 필요)'}")

    print(f"\n  열린 포지션: {len(s['open_positions'])}/3")
    for p in s['open_positions']:
        entry_dt = datetime.fromtimestamp(p['entry_ts']/1000, tz=timezone.utc)
        exit_dt = datetime.fromtimestamp(p['exit_ts_planned']/1000, tz=timezone.utc)
        remaining = exit_dt - datetime.now(timezone.utc)
        rem_h = max(0, int(remaining.total_seconds() / 3600))
        print(f"    {p['sym']}: entry {entry_dt.strftime('%m-%d %H:%M')} @ {p['entry_price']:.6f}, score {p['score']}, exit in {rem_h}h, size ${p['size_usd']:.2f}")

    closed = s['closed_history']
    print(f"\n  closed trades: {len(closed)}")
    if closed:
        wins = sum(1 for h in closed if h['net_bps'] > 0)
        losses = len(closed) - wins
        wr = wins / len(closed)
        total_pnl = sum(h['pnl_usd'] for h in closed)
        avg_net = sum(h['net_bps'] for h in closed) / len(closed)
        big_wins = sorted([h for h in closed if h['net_bps'] > 1000], key=lambda h: -h['net_bps'])[:3]
        big_losses = sorted([h for h in closed if h['net_bps'] < -500], key=lambda h: h['net_bps'])[:3]
        print(f"    win rate: {wr*100:.1f}% ({wins}W / {losses}L)")
        print(f"    avg net: {avg_net:+.1f} bps")
        print(f"    total PnL: ${total_pnl:+.3f}")
        bw = [(h['sym'], '%+.0fbps' % h['net_bps']) for h in big_wins]
        bl = [(h['sym'], '%+.0fbps' % h['net_bps']) for h in big_losses]
        print(f"    Top 3 winners: {bw}")
        if big_losses:
            print(f"    Top 3 losers: {bl}")

    # adaptive gate 상태
    if s['warmup_complete']:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000
        recent = [h for h in closed if h['exit_ts'] >= cutoff]
        recent_net = sum(h['net_bps'] for h in recent) if recent else 0
        gate = "ACTIVE" if recent_net > 0 else "PAUSED"
        print(f"\n  gate (recent 30d): {gate} (net={recent_net:+.0f}bps n={len(recent)})")


if __name__ == "__main__":
    main()
