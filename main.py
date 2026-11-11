import os
import re
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
import yfinance as yf
from groq import Groq

# 1. Railway Variables තුළින් පිරිසිදු Groq Key එක ලබාගැනීම
def load_clean_groq_key():
    candidates = [
        os.getenv("AI_API_KEY", ""),
        os.getenv("GROQ_API_KEY", "")
    ]
    for k, v in os.environ.items():
        if v and "gsk_" in v:
            candidates.append(v)

    for raw in candidates:
        if not raw:
            continue
        compact = re.sub(r'[\s"\']+', '', raw)
        match = re.search(r'(gsk_[A-Za-z0-9_\-]{20,})', compact)
        if match:
            return match.group(1)
    return ""

AI_API_KEY = load_clean_groq_key()
RAW_SYMBOLS = os.getenv("TRADING_SYMBOLS", "XAUUSDm,GBPUSDm,GBPJPYm,USDCADm")
SYMBOLS = [s.strip() for s in RAW_SYMBOLS.split(",") if s.strip()]

# 2. Groq ගිණුමේ දැනට ක්‍රියාත්මක වන AI Models ස්වයංක්‍රීයව සොයාගැනීම
def discover_groq_chat_models(groq_client):
    try:
        res = groq_client.models.list()
        all_ids = [m.id for m in res.data if getattr(m, "active", True)]
        skip_keywords = ["whisper", "tts", "playai", "guard", "embed", "moderation", "allam", "orpheus"]
        chat_models = [
            m for m in all_ids
            if not any(w in m.lower() for w in skip_keywords)
        ]
        priority_keywords = ["llama-4", "llama-3.3", "deepseek", "qwen", "gpt", "kimi", "llama-3.1", "gemma"]
        def model_rank(m_id):
            m_low = m_id.lower()
            for idx, kw in enumerate(priority_keywords):
                if kw in m_low:
                    return idx
            return 999
        chat_models.sort(key=model_rank)
        return chat_models if chat_models else all_ids
    except Exception as e:
        print(f"⚠️ Models ලැයිස්තුව ලබාගැනීමේ දෝෂය: {e}")
        return []

YF_MAP = {
    "XAUUSD": "GC=F", "XAUUSDM": "GC=F", "GOLD": "GC=F",
    "GBPUSD": "GBPUSD=X", "GBPUSDM": "GBPUSD=X",
    "GBPJPY": "GBPJPY=X", "GBPJPYM": "GBPJPY=X",
    "USDCAD": "USDCAD=X", "USDCADM": "USDCAD=X",
    "BTCUSD": "BTC-USD", "BTCUSDM": "BTC-USD",
    "NAS100": "NQ=F", "USTEC": "NQ=F", "USTECM": "NQ=F"
}

def get_yf_ticker(symbol):
    clean = symbol.upper()
    return YF_MAP.get(clean, YF_MAP.get(clean.rstrip("M"), "GBPUSD=X"))

def get_pip_value(symbol):
    s = symbol.upper()
    if "JPY" in s:
        return 0.01
    if "XAU" in s or "GOLD" in s:
        return 0.10
    if "NAS" in s or "BTC" in s or "USTEC" in s:
        return 1.0
    return 0.0001

def get_price_decimals(symbol):
    s = symbol.upper()
    if "JPY" in s or "XAU" in s or "GOLD" in s or "NAS" in s or "BTC" in s or "USTEC" in s:
        return 2
    return 5

# 3. සජීවී Candles (1m, 5m, 15m) ලබාගැනීම
def fetch_live_candles(symbol, interval, period, count=30):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period=period, interval=interval)
    candles = []
    if df is None or df.empty:
        return candles
    for idx, row in df.tail(count).iterrows():
        candles.append({
            "time": str(idx)[11:16],
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5)
        })
    return candles

# 4. දැන් පවතින මිලට ඉතාම ආසන්න Fresh SNR Levels පමණක් තෝරාගැනීම
def scan_nearby_snr_levels(candles, timeframe, current_price, max_pips_away, pip_val):
    levels = []
    if not candles or len(candles) < 5:
        return levels

    for i in range(len(candles) - 2):
        c1, c2 = candles[i], candles[i + 1]
        c1_bull = c1['close'] > c1['open']
        c2_bull = c2['close'] > c2['open']

        lvl_type = "A_RESISTANCE" if (c1_bull and not c2_bull) else ("V_SUPPORT" if (not c1_bull and c2_bull) else "GAP_SNR")
        lvl_price = c1['close']
        status = "Fresh_1"

        for j in range(i + 2, len(candles)):
            cj = candles[j]
            body_high = max(cj['open'], cj['close'])
            body_low = min(cj['open'], cj['close'])
            wick_touched = (cj['low'] <= lvl_price <= cj['high']) and not (body_low < lvl_price < body_high)
            body_broken = body_low < lvl_price < body_high

            if status == "Fresh_1" and wick_touched:
                status = "Unfresh"
            elif status in ["Fresh_1", "Unfresh"] and body_broken:
                status = "Fresh_SBR_RBS"
            elif status == "Fresh_SBR_RBS" and wick_touched:
                status = "Dead"
                break

        dist_pips = abs(current_price - lvl_price) / pip_val
        if status != "Dead" and dist_pips <= max_pips_away:
            levels.append({
                "tf": timeframe,
                "type": lvl_type,
                "price": round(lvl_price, 5),
                "status": status,
                "dist_pips": round(dist_pips, 1)
            })

    levels.sort(key=lambda x: x["dist_pips"])
    return levels[:8]

# 5M Candles වලින් මේ මොහොතේ මාකට් වේගය (ATR Volatility in Pips) ගණනය කිරීම
def calculate_live_atr_pips(m5_candles, pip_val):
    if not m5_candles or len(m
