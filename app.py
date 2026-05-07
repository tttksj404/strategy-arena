"""Strategy Builder — Flask backend.

Serves the drag-and-drop strategy builder UI.
Fetches Binance kline data, runs backtests, streams real-time prices.

Usage:
    pip install flask numpy requests
    python scripts/strategy_builder/app.py

Then open http://localhost:5555 in your browser.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, Response

sys.path.insert(0, str(Path(__file__).parent))
from engine import (
    COMPONENTS, run_backtest, compute_indicators, recommend_components,
    evaluate_live_signal,
    ema, rsi, adx, volume_ratio, bollinger, atr,
)

app = Flask(__name__, template_folder="templates")

BINANCE_API = "https://api.binance.com"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "LTCUSDT", "MATICUSDT", "NEARUSDT", "UNIUSDT", "OPUSDT",
    "ARBUSDT", "APTUSDT", "PEPEUSDT", "SUIUSDT", "WIFUSDT",
]

VALID_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d"}

INTERVALS = [
    {"value": "1m", "label": "1분"},
    {"value": "5m", "label": "5분"},
    {"value": "15m", "label": "15분"},
    {"value": "1h", "label": "1시간"},
    {"value": "4h", "label": "4시간"},
    {"value": "1d", "label": "1일"},
]


def _safe_int(val, default: int, lo: int = 1, hi: int = 1000) -> int:
    try:
        return max(lo, min(int(val), hi))
    except (ValueError, TypeError):
        return default


def _validate_symbol(sym: str) -> str:
    sym = str(sym).upper().strip()
    return sym if sym in SYMBOLS else "BTCUSDT"


def _validate_interval(iv: str) -> str:
    return iv if iv in VALID_INTERVALS else "1h"


def fetch_klines(symbol: str, interval: str, limit: int = 500) -> dict:
    url = f"{BINANCE_API}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    timestamps = []
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for k in raw:
        timestamps.append(k[0])
        opens.append(float(k[1]))
        highs.append(float(k[2]))
        lows.append(float(k[3]))
        closes.append(float(k[4]))
        volumes.append(float(k[5]))

    return {
        "timestamps": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def fetch_ticker(symbol: str) -> dict:
    url = f"{BINANCE_API}/api/v3/ticker/24hr"
    resp = requests.get(url, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {
        "symbol": data["symbol"],
        "price": float(data["lastPrice"]),
        "change_pct": float(data["priceChangePercent"]),
        "volume": float(data["quoteVolume"]),
        "high": float(data["highPrice"]),
        "low": float(data["lowPrice"]),
    }


def compute_realtime_indicators(close, high, low, volume):
    rsi_vals = rsi(close, 14)
    adx_vals = adx(high, low, close, 14)
    vratio = volume_ratio(volume, 20)
    atr_vals = atr(high, low, close, 14)
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    _, _, _, pct_b = bollinger(close, 20, 2.0)

    def last(arr):
        v = arr[-1] if len(arr) > 0 else None
        return None if (v is not None and np.isnan(v)) else (round(float(v), 2) if v is not None else None)

    trend = "상승" if (ema9[-1] > ema21[-1] > ema50[-1] and not any(np.isnan([ema9[-1], ema21[-1], ema50[-1]]))) else \
            "하락" if (ema9[-1] < ema21[-1] < ema50[-1] and not any(np.isnan([ema9[-1], ema21[-1], ema50[-1]]))) else "횡보"

    return {
        "rsi": last(rsi_vals),
        "adx": last(adx_vals),
        "volume_ratio": last(vratio),
        "atr": last(atr_vals),
        "atr_pct": round(float(atr_vals[-1] / close[-1] * 100), 3) if not np.isnan(atr_vals[-1]) and close[-1] > 0 else None,
        "ema9": last(ema9),
        "ema21": last(ema21),
        "ema50": last(ema50),
        "boll_pctb": last(pct_b),
        "trend": trend,
        "price": round(float(close[-1]), 4),
    }


# ─── Routes ───

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/manifest.json")
def manifest():
    m = {
        "name": "Strategy Arena",
        "short_name": "StratArena",
        "description": "퀀트 전략 빌더 & 백테스터",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0c14",
        "theme_color": "#0a0c14",
        "orientation": "portrait",
        "icons": [
            {"src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚔</text></svg>", "sizes": "any", "type": "image/svg+xml"}
        ],
    }
    return jsonify(m)


@app.route("/api/components")
def get_components():
    return jsonify(COMPONENTS)


@app.route("/api/symbols")
def get_symbols():
    return jsonify({"symbols": SYMBOLS, "intervals": INTERVALS})


@app.route("/api/klines")
def get_klines():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))
    interval = _validate_interval(request.args.get("interval", "1h"))
    limit = _safe_int(request.args.get("limit", 500), 500, 10, 1000)
    try:
        data = fetch_klines(symbol, interval, limit)
        return jsonify(data)
    except Exception:
        return jsonify({"error": "데이터 조회 실패"}), 500


@app.route("/api/ticker")
def get_ticker():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))
    try:
        data = fetch_ticker(symbol)
        return jsonify(data)
    except Exception:
        return jsonify({"error": "시세 조회 실패"}), 500


@app.route("/api/indicators")
def get_indicators():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))
    interval = _validate_interval(request.args.get("interval", "1h"))
    try:
        data = fetch_klines(symbol, interval, 200)
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        indicators = compute_realtime_indicators(close, high, low, volume)
        return jsonify(indicators)
    except Exception:
        return jsonify({"error": "지표 조회 실패"}), 500


@app.route("/api/backtest", methods=["POST"])
def run_backtest_api():
    body = request.get_json(force=True, silent=True) or {}
    symbol = _validate_symbol(body.get("symbol", "BTCUSDT"))
    interval = _validate_interval(body.get("interval", "1h"))
    limit = _safe_int(body.get("limit", 500), 500, 50, 1000)
    components = body.get("components", [])
    initial_equity = max(100, min(float(body.get("initial_equity", 10000)), 1e8))
    fee_pct = max(0, min(float(body.get("fee_pct", 0.075)), 1.0))

    if not components:
        return jsonify({"error": "컴포넌트를 추가해주세요"}), 400

    try:
        data = fetch_klines(symbol, interval, limit)
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        timestamps = np.array(data["timestamps"])

        chart_indicators = compute_indicators(close, high, low, volume, components)
        result = run_backtest(close, high, low, volume, timestamps,
                              components, initial_equity, fee_pct, interval)

        return jsonify({
            "result": result.to_dict(),
            "klines": data,
            "indicators": chart_indicators,
        })
    except Exception:
        return jsonify({"error": "백테스트 실행 실패"}), 500


@app.route("/api/recommend")
def get_recommendations():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))
    interval = _validate_interval(request.args.get("interval", "1h"))
    try:
        data = fetch_klines(symbol, interval, 200)
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        timestamps = np.array(data["timestamps"])
        recs = recommend_components(close, high, low, volume, timestamps)
        return jsonify({"recommendations": recs})
    except Exception:
        return jsonify({"error": "추천 조회 실패"}), 500


@app.route("/api/realtime-stream")
def realtime_stream():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))

    def generate():
        for _ in range(600):
            try:
                ticker = fetch_ticker(symbol)
                yield f"data: {json.dumps(ticker)}\n\n"
            except Exception:
                yield f"data: {json.dumps({'error': 'fetch failed'})}\n\n"
            time.sleep(3)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/depth")
def get_depth():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))
    limit = _safe_int(request.args.get("limit", 20), 20, 5, 100)
    try:
        url = f"{BINANCE_API}/api/v3/depth"
        resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception:
        return jsonify({"error": "호가 조회 실패"}), 500


@app.route("/api/trades")
def get_recent_trades():
    symbol = _validate_symbol(request.args.get("symbol", "BTCUSDT"))
    limit = _safe_int(request.args.get("limit", 50), 50, 5, 100)
    try:
        url = f"{BINANCE_API}/api/v3/trades"
        resp = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception:
        return jsonify({"error": "체결 조회 실패"}), 500


@app.route("/api/live-signal", methods=["POST"])
def live_signal():
    body = request.get_json(force=True, silent=True) or {}
    symbol = _validate_symbol(body.get("symbol", "BTCUSDT"))
    interval = _validate_interval(body.get("interval", "1h"))
    components = body.get("components", [])

    if not components:
        return jsonify({"error": "컴포넌트를 추가해주세요"}), 400

    try:
        data = fetch_klines(symbol, interval, 200)
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        timestamps = np.array(data["timestamps"])

        result = evaluate_live_signal(close, high, low, volume, timestamps, components)
        return jsonify(result)
    except Exception:
        return jsonify({"error": "신호 분석 실패"}), 500


@app.route("/api/scan", methods=["POST"])
def scan_symbols():
    body = request.get_json(force=True, silent=True) or {}
    interval = _validate_interval(body.get("interval", "1h"))
    components = body.get("components", [])
    raw_symbols = body.get("symbols", SYMBOLS[:10])
    symbols = [s for s in raw_symbols if s in SYMBOLS][:20]

    if not components:
        return jsonify({"error": "컴포넌트를 추가해주세요"}), 400

    results = []
    for sym in symbols:
        try:
            data = fetch_klines(sym, interval, 200)
            close = np.array(data["close"])
            high = np.array(data["high"])
            low = np.array(data["low"])
            volume = np.array(data["volume"])
            timestamps = np.array(data["timestamps"])

            sig = evaluate_live_signal(close, high, low, volume, timestamps, components)
            sig["symbol"] = sym
            results.append(sig)
        except Exception:
            results.append({"symbol": sym, "signal": "error", "signal_text": "데이터 오류", "checks": []})

    results.sort(key=lambda x: (0 if x["signal"] in ("buy", "sell") else 1))
    return jsonify({"results": results})


@app.route("/sw.js")
def service_worker():
    sw_js = """self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));"""
    return Response(sw_js, mimetype="application/javascript",
                    headers={"Service-Worker-Allowed": "/"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5555))
    print(f"\n  Strategy Arena")
    print(f"  http://localhost:{port}")
    print(f"  Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
