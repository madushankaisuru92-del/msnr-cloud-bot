import os
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
import yfinance as yf
from google import genai
from google.genai import types

# 1. Cloud Environment Variables ලබාගැනීම
# (API Keys එකකට වඩා ඇත්නම් කොමාවකින් වෙන් කර AI_API_KEY හි දැමිය හැක)
RAW_KEYS = os.getenv("AI_API_KEY", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]
SYMBOLS = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "XAUUSDm,GBPUSDm").split(",") if s.strip()]

# නොමිලේ ලැබෙන Quota එක අවසන් වූ විට මාරු වීමට Models ලැයිස්තුව
FREE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-3.8-flash",
    "gemini-1.5-flash"
]

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

# 2. නොමිලේ සජීවී Candles ලබාගැනීම
def fetch_free_candles(symbol, interval, period, count=35):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period=period, interval=interval)
    candles = []
    for idx, row in df.tail(count).iterrows():
        candles.append({
            "time": str(idx)[:16],
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5)
        })
    return candles

# 3. Step 1: A-Shape, V-Shape සහ Gap Levels ගණනය (2-Time Usage Rule)
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
                "tf": timeframe,
                "type": lvl_type,
                "price": round(lvl_price, 5),
                "status": status
            })

    return levels[-10:]

# 4. සියලුම Coins එකවර AI මගින් විශ්ලේෂණය කිරීම (Quota ඉතිරි කරන Batch ක්‍රමය)
async def analyze_all_symbols_with_ai(market_payload):
    prompt = f"""
    You are an automated MALAYSIAN SNR (MSNR 1.0) Forex Trading Engine.
    Analyze each symbol in the provided dataset and strictly apply MSNR rules:
    1. Storyline: Weekly/Daily Fresh Level touched -> Wick Rejection close -> 1 TF lower (Daily/4H) External Level broken by Candle Body. Follow Daily if Weekly conflicts.
    2. Identify 4H HNS zones: APEX, QM, GAP, SBR_RBS, plus 1H Refining level (ABOVE 4H for SELL, UNDER 4H for BUY).
    3. Even if waiting for confirmation, always identify the nearest projected/active Entry Price and the Expected Direction (UP for Bullish/Buy, DOWN for Bearish/Sell) based on the nearest HTF Fresh level rejection or storyline.
    4. Setup Type:
       - "APEX_LIMIT" -> SL is 30 pips.
       - "QM_GAP_SBR_MARKET" -> SL is 20 pips.

    Market Data:
    {json.dumps(market_payload)}

    Return ONLY valid JSON as an object where each key is the symbol name, matching this exact structure:
    {{
      "SYMBOL_NAME": {{
        "status": "ACTIVE_TRADE_SIGNAL" | "PENDING_SETUP_WAIT",
        "direction_up_down": "UP" | "DOWN",
        "position_type": "LONG (BUY)" | "SHORT (SELL)",
        "setup_zone": "APEX" | "QM" | "GAP" | "SBR_RBS" | "HTF_WAIT",
        "entry_price": 0.0,
        "opposite_fresh_level": 0.0,
        "reason": "short clear explanation"
      }}
    }}
    """

    def call_gemini_with_fallback():
        for api_key in API_KEYS:
            client = genai.Client(api_key=api_key)
            for model_name in FREE_MODELS:
                try:
                    res = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    return json.loads(res.text)
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str or "503" in err_str:
                        print(f"[AI Fallback] {model_name} සීමාව ඉක්මවා ඇත හෝ කාර්යබහුලයි. ඊළඟ Model එකට මාරු වේ...")
                        time.sleep(2)
                        continue
                    raise e
        raise Exception("සියලුම නොමිලේ දෙන AI Models වල අද දින සීමාව (Quota) අවසන් වී ඇත.")

    return await asyncio.to_thread(call_gemini_with_fallback)

# 5. ඔබ ඉල්ලූ පැහැදිලි ආකෘතියෙන් (Exact Prices සහිතව) විස්තර පෙන්වීම
def print_formatted_signal(symbol, current_price, data):
    sl_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
    pip = get_pip_value(symbol)
    decimals = get_price_decimals(symbol)

    direction = data.get("direction_up_down", "NONE")
    pos_type = data.get("position_type", "NONE")
    zone = data.get("setup_zone", "HTF_WAIT")
    status = data.get("status", "PENDING_SETUP_WAIT")

    entry = float(data.get("entry_price") or 0.0)
    if entry <= 0:
        entry = current_price

    # MSNR නීතියට අනුව APEX සඳහා 30 Pips ද, අනෙක් ඒවාට 20 Pips ද ලෙස SL මිල ගණනය කිරීම
    sl_pips = 30 if zone == "APEX" else 20
    tp_pips = 200

    if "UP" in direction or "LONG" in pos_type:
        sl_price = round(entry - (sl_pips * pip), decimals)
        tp_price = round(entry + (tp_pips * pip), decimals)
    else:
        sl_price = round(entry + (sl_pips * pip), decimals)
        tp_price = round(entry - (tp_pips * pip), decimals)

    # ප්‍රතිවිරුද්ධ Fresh Level එකක් තිබේ නම් සහ එය නිවැරදි පැත්තේ නම් TP ලෙස පෙන්වීම
    opp_lvl = float(data.get("opposite_fresh_level") or 0.0)

    print(
        f"\n========================================\n"
        f"🕒 TIME (SL)      : {sl_time}\n"
        f"🪙 COIN / PAIR    : {symbol} (දැන් මිල: {round(current_price, decimals)})\n"
        f"📊 STATUS         : {status} ({zone})\n"
        f"📈 DIRECTION      : {direction}\n"
        f"🎯 POSITION TYPE  : {pos_type}\n"
        f"🔵 ENTRY PRICE    : {round(entry, decimals)}\n"
        f"🛑 STOP LOSS (SL) : {sl_price}  ({sl_pips} Pips ආරක්ෂාව)\n"
        f"🟢 TAKE PROFIT(TP): {tp_price}  (200 Pips ඉලක්කය | Next HTF Level: {round(opp_lvl, decimals)})\n"
        f"📝 REASON         : {data.get('reason', '')}\n"
        f"========================================"
    )

# 6. ප්‍රධාන Cloud Loop එක
async def main():
    print("MSNR AI Cloud Bot (Multi-Model & Exact Price Version) ආරම්භ විය!")
    last_scan_slot = -1

    while True:
        try:
            now = datetime.now(timezone.utc)
            # විනාඩි 30කට වරක් පමණක් Scan කිරීම (Quota ආරක්ෂා කිරීමට)
            current_slot = (now.hour * 2) + (1 if now.minute >= 30 else 0)

            if current_slot != last_scan_slot:
                last_scan_slot = current_slot
                market_payload = {}
                current_prices = {}

                for symbol in SYMBOLS:
                    w1 = await asyncio.to_thread(fetch_free_candles, symbol, "1wk", "1y", 30)
                    d1 = await asyncio.to_thread(fetch_free_candles, symbol, "1d", "3mo", 30)
                    h1 = await asyncio.to_thread(fetch_free_candles, symbol, "1h", "1mo", 35)
                    m5 = await asyncio.to_thread(fetch_free_candles, symbol, "5m", "5d", 20)
                    h4 = h1[::4]

                    current_prices[symbol] = m5[-1]["close"] if m5 else 0.0
                    market_payload[symbol] = {
                        "current_price": current_prices[symbol],
                        "w1_levels": scan_snr_levels(w1, "1W"),
                        "d1_levels": scan_snr_levels(d1, "1D"),
                        "h4_candles": h4[-10:],
                        "h1_candles": h1[-10:],
                        "m5_candles": m5[-10:]
                    }

                decisions = await analyze_all_symbols_with_ai(market_payload)

                for symbol in SYMBOLS:
                    sym_data = decisions.get(symbol, {})
                    print_formatted_signal(symbol, current_prices.get(symbol, 0.0), sym_data)

        except Exception as e:
            print("Scan Error:", e)

        await asyncio.sleep(20)

if __name__ == "__main__":
    asyncio.run(main())
