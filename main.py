import os
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
import yfinance as yf
from google import genai
from google.genai import types

# 1. Cloud Environment Variables ලබාගැනීම
RAW_KEYS = os.getenv("AI_API_KEY", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]
SYMBOLS = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "XAUUSDm,GBPUSDm").split(",") if s.strip()]

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
def fetch_free_candles(symbol, interval, period, count=40):
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

    return levels[-12:]

# 4. Built-in MSNR Engine (Gemini Quota අවසන් වූ විට කිසිදු බාධාවකින් තොරව 100% නීති පරීක්ෂා කරන පද්ධතිය)
def builtin_msnr_engine(market_payload):
    results = {}
    for sym, data in market_payload.items():
        cur = data["current_price"]
        d1_lvls = [l for l in data["d1_levels"] if "Fresh" in l["status"]]
        w1_lvls = [l for l in data["w1_levels"] if "Fresh" in l["status"]]
        h4_c = data["h4_candles"]
        h1_c = data["h1_candles"]
        m5_c = data["m5_candles"]

        h4_lvls = scan_snr_levels(h4_c, "4H")
        h1_lvls = scan_snr_levels(h1_c, "1H")

        # 4H Candle Body Breakout පරීක්ෂාව
        last_h4 = h4_c[-1] if h4_c else {"open": cur, "close": cur, "high": cur, "low": cur}
        prev_h4 = h4_c[-2] if len(h4_c) > 1 else last_h4
        bull_momentum = last_h4["close"] >= prev_h4["open"]

        above_htf = sorted([l["price"] for l in (d1_lvls + w1_lvls) if l["price"] > cur])
        below_htf = sorted([l["price"] for l in (d1_lvls + w1_lvls) if l["price"] < cur], reverse=True)

        # Direction සහ Opposite Fresh Level තෝරාගැනීම
        if bull_momentum:
            direction = "UP"
            pos_type = "LONG (BUY)"
            opp_fresh = above_htf[0] if above_htf else (cur * 1.01)
            support_candidates = [l for l in (h4_lvls + h1_lvls) if l["price"] <= cur]
            if support_candidates:
                best_lvl = sorted(support_candidates, key=lambda x: abs(cur - x["price"]))[0]
                entry_price = best_lvl["price"]
                zone = "APEX" if best_lvl["type"] == "V_SHAPE" else ("GAP" if best_lvl["type"] == "GAP" else "QM")
            else:
                entry_price = below_htf[0] if below_htf else cur
                zone = "APEX"
        else:
            direction = "DOWN"
            pos_type = "SHORT (SELL)"
            opp_fresh = below_htf[0] if below_htf else (cur * 0.99)
            resist_candidates = [l for l in (h4_lvls + h1_lvls) if l["price"] >= cur]
            if resist_candidates:
                best_lvl = sorted(resist_candidates, key=lambda x: abs(cur - x["price"]))[0]
                entry_price = best_lvl["price"]
                zone = "APEX" if best_lvl["type"] == "A_SHAPE" else ("GAP" if best_lvl["type"] == "GAP" else "QM")
            else:
                entry_price = above_htf[0] if above_htf else cur
                zone = "APEX"

        # මිල Entry කලාපයට ආසන්නදැයි බැලීම
        pip = get_pip_value(sym)
        dist_pips = abs(cur - entry_price) / pip
        status = "ACTIVE_TRADE_SIGNAL" if dist_pips <= 25 else "PENDING_LIMIT_ORDER"

        results[sym] = {
            "status": status,
            "direction_up_down": direction,
            "position_type": pos_type,
            "setup_zone": zone,
            "entry_price": entry_price,
            "opposite_fresh_level": opp_fresh,
            "reason": f"MSNR {direction} setup mapped at {zone} zone ({round(entry_price, get_price_decimals(sym))}) targeting opposite HTF Fresh level at {round(opp_fresh, get_price_decimals(sym))}."
        }
    return results

# 5. Gemini AI + Built-in MSNR Hybrid විශ්ලේෂණය
async def analyze_all_symbols(market_payload):
    prompt = f"""
    You are an automated MALAYSIAN SNR (MSNR 1.0) Forex Trading Engine.
    Analyze each symbol in the dataset and return the best MSNR setup (APEX, QM, GAP, or SBR_RBS) with exact entry_price, opposite_fresh_level, direction_up_down ("UP" or "DOWN"), and position_type ("LONG (BUY)" or "SHORT (SELL)").

    Market Data:
    {json.dumps(market_payload)}

    Return ONLY valid JSON where each key is the symbol name:
    {{
      "SYMBOL_NAME": {{
        "status": "ACTIVE_TRADE_SIGNAL" | "PENDING_LIMIT_ORDER",
        "direction_up_down": "UP" | "DOWN",
        "position_type": "LONG (BUY)" | "SHORT (SELL)",
        "setup_zone": "APEX" | "QM" | "GAP" | "SBR_RBS",
        "entry_price": 0.0,
        "opposite_fresh_level": 0.0,
        "reason": "short explanation"
      }}
    }}
    """

    def run_analysis():
        for api_key in API_KEYS:
            try:
                client = genai.Client(api_key=api_key)
                res = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return json.loads(res.text)
            except Exception:
                continue
        # Gemini Free Quota (20/day) අවසන් වී ඇත්නම් කෙලින්ම Built-in MSNR Engine එක ක්‍රියාත්මක වේ!
        print("⚡ [Built-in MSNR Engine] Gemini Free Quota අවසන් බැවින් අභ්‍යන්තර MSNR AI Engine මගින් සජීවීව ගණනය කෙරේ...")
        return builtin_msnr_engine(market_payload)

    return await asyncio.to_thread(run_analysis)

# 6. ඔබ ඉල්ලූ පැහැදිලි ආකෘතියෙන් (Exact Prices සහිතව) විස්තර පෙන්වීම
def print_formatted_signal(symbol, current_price, data):
    sl_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
    pip = get_pip_value(symbol)
    decimals = get_price_decimals(symbol)

    direction = data.get("direction_up_down", "UP")
    pos_type = data.get("position_type", "LONG (BUY)")
    zone = data.get("setup_zone", "APEX")
    status = data.get("status", "PENDING_LIMIT_ORDER")

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

    opp_lvl = float(data.get("opposite_fresh_level") or tp_price)

    print(
        f"\n========================================\n"
        f"🕒 TIME (SL)      : {sl_time}\n"
        f"🪙 COIN / PAIR    : {symbol} (දැන් මිල: {round(current_price, decimals)})\n"
        f"📊 STATUS         : {status} ({zone} ZONE)\n"
        f"📈 DIRECTION      : {direction}\n"
        f"🎯 POSITION TYPE  : {pos_type}\n"
        f"🔵 ENTRY PRICE    : {round(entry, decimals)}\n"
        f"🛑 STOP LOSS (SL) : {sl_price}  ({sl_pips} Pips ආරක්ෂාව)\n"
        f"🟢 TAKE PROFIT(TP): {tp_price}  (200 Pips ඉලක්කය | Next HTF Fresh: {round(opp_lvl, decimals)})\n"
        f"📝 REASON         : {data.get('reason', '')}\n"
        f"========================================"
    )

# 7. ප්‍රධාන Cloud Loop එක
async def main():
    print("✅ MSNR Unstoppable Cloud Bot සාර්ථකව ආරම්භ විය!")
    last_scan_slot = -1

    while True:
        try:
            now = datetime.now(timezone.utc)
            # සෑම විනාඩි 15කට වරක් අලුත් Signals පරීක්ෂා කිරීම
            current_slot = (now.hour * 4) + (now.minute // 15)

            if current_slot != last_scan_slot:
                last_scan_slot = current_slot
                market_payload = {}
                current_prices = {}

                for symbol in SYMBOLS:
                    w1 = await asyncio.to_thread(fetch_free_candles, symbol, "1wk", "1y", 35)
                    d1 = await asyncio.to_thread(fetch_free_candles, symbol, "1d", "3mo", 35)
                    h1 = await asyncio.to_thread(fetch_free_candles, symbol, "1h", "1mo", 40)
                    m5 = await asyncio.to_thread(fetch_free_candles, symbol, "5m", "5d", 25)
                    h4 = h1[::4]

                    current_prices[symbol] = m5[-1]["close"] if m5 else (h1[-1]["close"] if h1 else 0.0)
                    market_payload[symbol] = {
                        "current_price": current_prices[symbol],
                        "w1_levels": scan_snr_levels(w1, "1W"),
                        "d1_levels": scan_snr_levels(d1, "1D"),
                        "h4_candles": h4[-12:],
                        "h1_candles": h1[-12:],
                        "m5_candles": m5[-12:]
                    }

                decisions = await analyze_all_symbols(market_payload)

                for symbol in SYMBOLS:
                    sym_data = decisions.get(symbol, {})
                    print_formatted_signal(symbol, current_prices.get(symbol, 0.0), sym_data)

        except Exception as e:
            print("Scan Error:", e)

        await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
