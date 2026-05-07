"""Strategy Builder — Flask backend.

Serves the drag-and-drop strategy builder UI.
Compute-only backend: browser fetches Binance data directly,
sends it here for backtest/indicator/signal computation.

Usage:
    pip install flask numpy
    python scripts/strategy_builder/app.py

Then open http://localhost:5555 in your browser.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, Response

sys.path.insert(0, str(Path(__file__).parent))
from engine import (
    COMPONENTS, run_backtest, compute_indicators, recommend_components,
    evaluate_live_signal,
    ema, rsi, adx, volume_ratio, bollinger, atr,
)

app = Flask(__name__, template_folder="templates")

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


def _validate_interval(iv: str) -> str:
    return iv if iv in VALID_INTERVALS else "1h"


def _extract_klines(body: dict) -> dict:
    klines = body.get("klines")
    if not klines or not isinstance(klines, dict):
        return None
    required = ["timestamps", "open", "high", "low", "close", "volume"]
    if not all(k in klines and isinstance(klines[k], list) and len(klines[k]) > 0 for k in required):
        return None
    return {
        "timestamps": klines["timestamps"],
        "open": [float(v) for v in klines["open"]],
        "high": [float(v) for v in klines["high"]],
        "low": [float(v) for v in klines["low"]],
        "close": [float(v) for v in klines["close"]],
        "volume": [float(v) for v in klines["volume"]],
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


@app.route("/api/indicators", methods=["POST"])
def get_indicators():
    body = request.get_json(force=True, silent=True) or {}
    data = _extract_klines(body)
    if not data:
        return jsonify({"error": "klines 데이터가 필요합니다"}), 400
    try:
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        indicators = compute_realtime_indicators(close, high, low, volume)
        return jsonify(indicators)
    except Exception:
        return jsonify({"error": "지표 계산 실패"}), 500


@app.route("/api/backtest", methods=["POST"])
def run_backtest_api():
    body = request.get_json(force=True, silent=True) or {}
    interval = _validate_interval(body.get("interval", "1h"))
    components = body.get("components", [])
    initial_equity = max(100, min(float(body.get("initial_equity", 10000)), 1e8))
    fee_pct = max(0, min(float(body.get("fee_pct", 0.075)), 1.0))

    if not components:
        return jsonify({"error": "컴포넌트를 추가해주세요"}), 400

    data = _extract_klines(body)
    if not data:
        return jsonify({"error": "klines 데이터가 필요합니다"}), 400

    try:
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


@app.route("/api/recommend", methods=["POST"])
def get_recommendations():
    body = request.get_json(force=True, silent=True) or {}
    data = _extract_klines(body)
    if not data:
        return jsonify({"error": "klines 데이터가 필요합니다"}), 400
    try:
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        timestamps = np.array(data["timestamps"])
        recs = recommend_components(close, high, low, volume, timestamps)
        return jsonify({"recommendations": recs})
    except Exception:
        return jsonify({"error": "추천 조회 실패"}), 500


@app.route("/api/live-signal", methods=["POST"])
def live_signal():
    body = request.get_json(force=True, silent=True) or {}
    components = body.get("components", [])

    if not components:
        return jsonify({"error": "컴포넌트를 추가해주세요"}), 400

    data = _extract_klines(body)
    if not data:
        return jsonify({"error": "klines 데이터가 필요합니다"}), 400

    try:
        close = np.array(data["close"])
        high = np.array(data["high"])
        low = np.array(data["low"])
        volume = np.array(data["volume"])
        timestamps = np.array(data["timestamps"])

        result = evaluate_live_signal(close, high, low, volume, timestamps, components)
        return jsonify(result)
    except Exception:
        return jsonify({"error": "신호 분석 실패"}), 500


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
