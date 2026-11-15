import os
import json
import time
import asyncio
import sqlite3
from datetime import datetime, timezone
import yfinance as yf
from google import genai
from google.genai import types

# 1. Cloud Environment Variables ලබාගැනීම
AI_API_KEY = os.getenv("AI_API_KEY")
SYMBOLS = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "XAUUSDm,GBPUSDm").split(",")]

ai_client = genai.Client(api_key=AI_API_KEY)

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

# 2. නොමිලේ සජීවී Candles ලබාගැනීම
def fetch_free_candles(symbol, interval, period, count=40):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period=period, interval=interval)
    candles = []
    for idx, row in df.tail(count).iterrows():
        candles.append({
            "time": str(idx),
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5)
        })
    return candles

# 3. Step 1: A-Shape, V-Shape සහ Gap Levels ගණනය
def scan_snr_levels(candles, timeframe):
    levels = []
    if not candles or len(candles) < 5:
        return levels

    for i in range(len(candles) - 3):
        c1, c2 = candles[i], candles[i + 1]
        c1_bull = c1['close'] > c1['open']
        c2_bull = c2['close'] > c2['open']

        lvl_type = "A_SHAPE" if (c1_bull and not c2_bull) else ("V_SHAPE" if (not c1_bull and c2_bull) else "GAP")
        lvl_price = c1['close']
        status = "Fresh_Use_1"

        for j in range(i + 2, len(candles)):
            cj = candles[j]
            body_high = max(cj['open'], cj['close'])
            body_low = min(cj['open'], cj['close'])
            wick_touched = (cj['low'] <= lvl_price <= cj['high']) and not (body_low < lvl_price < body_high)
            body_broken = body_low < lvl_price < body_high

            if status == "Fresh_Use_1" and wick_touched:
                status = "Unfresh_1"
            elif status in ["Fresh_Use_1", "Unfresh_1"] and body_broken:
                status = "Fresh_Use_2"
            elif status == "Fresh_Use_2" and wick_touched:
                status = "Dead"
                break

        if status != "Dead":
            levels.append({
                "timeframe": timeframe,
                "type": lvl_type,
                "price": round(lvl_price, 5),
                "status": status
            })

    return levels[-15:]

# 4. AI මගින් ඉදිරියට Predict කරන (Forward-Looking) විශ්ලේෂණය
async def analyze_with_ai(symbol, current_price, w1_levels, d1_levels, h4_candles, h1_candles, m5_candles):
    prompt = f"""
    You are an automated MALAYSIAN SNR (MSNR 1.0) Forex Trading Engine analyzing {symbol}.
    CURRENT LIVE PRICE: {current_price}

    CRITICAL TIMING & PREDICTION RULES:
    - DO NOT give late signals where the move has already happened!
    - Always look AHEAD of the current price ({current_price}) to give the trader time to place the order.
    - Prefer LIMIT ORDERS ("BUY_LIMIT" below {current_price}, or "SELL_LIMIT" above {current_price}) at upcoming valid MSNR HNS zones (APEX, QM, GAP, SBR/RBS) once the Storyline is confirmed.
    - Only give "BUY_MARKET" or "SELL_MARKET" if price is CURRENTLY right at the entry zone (within 5 pips) and just confirmed on 5M. If price already ran away from the zone, output "WAIT".

    MSNR 1.0 STRATEGY RULES:
    1. Storyline: Weekly/Daily Fresh Level touch -> Wick Rejection -> 4H/Daily External Level Body Break. (If Weekly & Daily conflict, follow Daily).
    2. HNS Zones: Identify upcoming APEX (SL 30 pips), QM, GAP, SBR/RBS (SL 20 pips).
    3. Confluences: 1H Refining (ABOVE 4H for SELL, UNDER 4H for BUY), Trendline, Asian Session High/Low sweep.

    Weekly Active Levels: {json.dumps(w1_levels)}
    Daily Active Levels: {json.dumps(d1_levels)}
    Last 15 4H Candles: {json.dumps(h4_candles[-15:])}
    Last 15 1H Candles: {json.dumps(h1_candles[-15:])}
    Last 10 5M Candles: {json.dumps(m5_candles[-10:])}

    Return ONLY valid JSON:
    {{
      "storyline": "BULLISH" | "BEARISH" | "NONE",
      "action": "BUY_LIMIT" | "SELL_LIMIT" | "BUY_MARKET" | "SELL_MARKET" | "WAIT",
      "entry_price": 0.0,
      "sl_pips": 30,
      "tp_pips": 200,
      "reason": "short explanation of upcoming setup"
    }}
    """
    def call_gemini():
        for attempt in range(3):
            try:
                res = ai_client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return json.loads(res.text)
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(5 * (attempt + 1))
                    continue
                raise e
        return {"storyline": "NONE", "action": "WAIT", "entry_price": 0.0, "reason": "Server busy"}

    return await asyncio.to_thread(call_gemini)

# 5. පරක්කු වූ සිග්නල් (Stale/Expired Signals) පරීක්ෂා කර ඉවත් කිරීම
def validate_signal_timing(symbol, decision, live_price, max_drift_pips=8):
    action = decision.get("action", "WAIT")
    entry = float(decision.get("entry_price", 0.0))
    if action == "WAIT" or entry <= 0:
        return False, "No active setup"

    pip = get_pip_value(symbol)

    # Limit Orders ඉස්සරහට තිබේදැයි පරීක්ෂාව
    if action == "BUY_LIMIT":
        if entry >= live_price:
            return False, f"EXPIRED: මිල ({live_price}) දැනටමත් Buy Limit ({entry}) පසුකර ඇත."
        return True, f"VALID UPCOMING BUY LIMIT (වත්මන් මිලට වඩා Pips {((live_price - entry)/pip):.1f}ක් පහළින්)"

    if action == "SELL_LIMIT":
        if entry <= live_price:
            return False, f"EXPIRED: මිල ({live_price}) දැනටමත් Sell Limit ({entry}) පසුකර ඇත."
        return True, f"VALID UPCOMING SELL LIMIT (වත්මන් මිලට වඩා Pips {((entry - live_price)/pip):.1f}ක් ඉහළින්)"

    # Market Orders පරක්කු වී ඇත්දැයි පරීක්ෂාව
    if action == "BUY_MARKET":
        drift = (live_price - entry) / pip
        if drift > max_drift_pips:
            return False, f"EXPIRED: මිල දැනටමත් Pips {drift:.1f}ක් ඉහළට ගොස් ඇත (පරක්කුයි)."
        return True, "VALID LIVE BUY MARKET"

    if action == "SELL_MARKET":
        drift = (entry - live_price) / pip
        if drift > max_drift_pips:
            return False, f"EXPIRED: මිල දැනටමත් Pips {drift:.1f}ක් පහළට ගොස් ඇත (පරක්කුයි)."
        return True, "VALID LIVE SELL MARKET"

    return False, "Unknown action"

# 6. ප්‍රධාන Cloud Loop එක (සෑම විනාඩි 2කට වරක් වේගවත් පරීක්ෂාව)
async def main():
    print("MSNR Predictive AI Bot (Anti-Delay Filter සහිතව) ආරම්භ විය!")
    last_scan_min = -1

    while True:
        try:
            now = datetime.now(timezone.utc)
            # සෑම විනාඩි 2කට වරක් පරීක්ෂා කිරීම (පෙර විනාඩි 5 වෙනුවට)
            if now.minute % 2 == 0 and now.minute != last_scan_min:
                last_scan_min = now.minute
                for symbol in SYMBOLS:
                    w1 = await asyncio.to_thread(fetch_free_candles, symbol, "1wk", "1y", 40)
                    d1 = await asyncio.to_thread(fetch_free_candles, symbol, "1d", "3mo", 40)
                    h1 = await asyncio.to_thread(fetch_free_candles, symbol, "1h", "1mo", 40)
                    m5 = await asyncio.to_thread(fetch_free_candles, symbol, "5m", "5d", 20)
                    m1 = await asyncio.to_thread(fetch_free_candles, symbol, "1m", "1d", 5)

                    if not m1 or not m5:
                        continue

                    h4 = h1[::4]
                    live_price = m1[-1]["close"]

                    w1_levels = scan_snr_levels(w1, "1W")
                    d1_levels = scan_snr_levels(d1, "1D")

                    decision = await analyze_with_ai(symbol, live_price, w1_levels, d1_levels, h4, h1, m5)

                    # AI එකෙන් පිළිතුර ආ පසු නැවතත් සජීවී මිල ගෙන පරක්කු වී ඇත්දැයි බැලීම
                    fresh_m1 = await asyncio.to_thread(fetch_free_candles, symbol, "1m", "1d", 2)
                    latest_price = fresh_m1[-1]["close"] if fresh_m1 else live_price

                    is_valid, status_msg = validate_signal_timing(symbol, decision, latest_price)

                    if is_valid:
                        print(f"✅ [{symbol}] {status_msg} | Entry: {decision['entry_price']} | SL: {decision['sl_pips']}p | TP: {decision['tp_pips']}p | Reason: {decision['reason']}")
                    else:
                        print(f"⏳ [{symbol}] මිල: {latest_price} | {status_msg}")

        except Exception as e:
            print("Scan Error:", e)

        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
