import os
import json
import asyncio
import sqlite3
from datetime import datetime, timezone
import yfinance as yf
from google import genai
from google.genai import types
from metaapi_cloud_sdk import MetaApi

# 1. Cloud Environment Variables ලබාගැනීම
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("MT5_ACCOUNT_ID")
AI_API_KEY = os.getenv("AI_API_KEY")
SYMBOLS = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "XAUUSDm,GBPUSDm").split(",")]

# නව Google GenAI SDK සැකසුම
ai_client = genai.Client(api_key=AI_API_KEY)

# නොමිලේ Cloud Data (Yahoo Finance) සඳහා Symbol ගැලපීම
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

# 2. SQLite Database සැකසීම (SNR Levels සහ Cloud Demo Trades)
def init_db():
    conn = sqlite3.connect("msnr_cloud.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS snr_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, timeframe TEXT, level_type TEXT,
            price REAL, status TEXT, touches INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS virtual_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, direction TEXT, entry_price REAL,
            sl REAL, tp REAL, opp_level REAL, status TEXT
        )
    """)
    conn.commit()
    conn.close()

# 3. නොමිලේ සජීවී Candles ලබාගැනීම (MetaApi නොමැති විට)
def fetch_free_candles(symbol, interval, period, count=40):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period=period, interval=interval)
    candles = []
    for idx, row in df.tail(count).iterrows():
        candles.append({
            "time": str(idx),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })
    return candles

# 4. Step 1: A-Shape, V-Shape සහ Gap Levels ගණනය (Fresh_Use_1, Unfresh_1, Fresh_Use_2, Dead)
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

# 5. Step 2, 3 & 4: AI මගින් Storyline, HNS සහ Entry තීරණය ගැනීම
async def analyze_with_ai(symbol, w1_levels, d1_levels, h4_candles, h1_candles, m5_candles):
    prompt = f"""
    You are an automated MALAYSIAN SNR (MSNR 1.0) Forex Trading Engine analyzing {symbol}.
    Strictly apply these 5 rules:
    1. Storyline requires: Price touches a Weekly/Daily Fresh Level (Fresh_Use_1 or Fresh_Use_2) -> Wick Rejection close -> 1 TF lower (Daily/4H) External Level broken by Candle Body. If Weekly and Daily conflict, follow Daily direction.
    2. Identify 4H HNS zones from the breakout: APEX, QM (Left Shoulder), GAP, and SBR_RBS.
    3. Check 1H Refining Level (Must be ABOVE 4H level for SELL, UNDER 4H level for BUY), 4H/1H Trendline, and Asian Session (00:00-09:00 UTC) High/Low sweep.
    4. Entry trigger rules:
       - If price is at APEX: action = "PLACE_APEX_LIMIT" (SL = 30 pips).
       - If price is touching QM, GAP, or SBR_RBS and 5M candles show a 5M Gap Body Break, QM retest, or Trendline Body Break: action = "EXECUTE_CONFIRMED_MARKET" (SL = 20 pips).
       - Otherwise: action = "WAIT".

    Weekly Active Levels: {json.dumps(w1_levels)}
    Daily Active Levels: {json.dumps(d1_levels)}
    Last 15 4H Candles: {json.dumps(h4_candles[-15:], default=str)}
    Last 15 1H Candles: {json.dumps(h1_candles[-15:], default=str)}
    Last 15 5M Candles: {json.dumps(m5_candles[-15:], default=str)}

    Return ONLY valid JSON:
    {{
      "storyline": "BULLISH" | "BEARISH" | "NONE",
      "action": "PLACE_APEX_LIMIT" | "EXECUTE_CONFIRMED_MARKET" | "WAIT",
      "direction": "BUY" | "SELL" | "NONE",
      "entry_price": 0.0,
      "sl_pips": 30,
      "tp_pips": 200,
      "opposite_htf_fresh_level": 0.0,
      "reason": "short explanation"
    }}
    """
    def call_gemini():
        res = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(res.text)

    return await asyncio.to_thread(call_gemini)

def get_pip_value(symbol):
    if "JPY" in symbol.upper():
        return 0.01
    if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
        return 0.10
    if "NAS" in symbol.upper() or "BTC" in symbol.upper() or "USTEC" in symbol.upper():
        return 1.0
    return 0.0001

# 6. ප්‍රධාන Cloud Loop එක
async def main():
    init_db()
    print("MSNR AI Cloud Bot ආරම්භ විය!")
    connection = None
    account = None

    # MetaApi සම්බන්ධතාවය පරීක්ෂා කිරීම (නොමැති නම් Free Cloud Mode වෙත මාරු වේ)
    if METAAPI_TOKEN and ACCOUNT_ID:
        try:
            api = MetaApi(token=METAAPI_TOKEN)
            account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
            if account.state != 'DEPLOYED':
                await account.deploy()
            await account.wait_connected()
            connection = account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()
            print("MT5 Cloud ගිණුම සාර්ථකව සම්බන්ධ විය!")
        except Exception as e:
            print(f"MetaApi ගෙවීම් සීමාව නිසා 100% නොමිලේ ක්‍රියාත්මක වන Free Cloud Demo Mode වෙත මාරු විය! ({e})")
            connection = None

    last_scan_min = -1
    daily_losses = 0
    current_day = datetime.now(timezone.utc).day

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.day != current_day:
                daily_losses = 0
                current_day = now.day

            if daily_losses >= 2:
                print("අද දින Losses 2 සම්පූර්ණයි (Step 5). හෙට වනතුරු විවේක ගනී...")
                await asyncio.sleep(300)
                continue

            # සෑම විනාඩි 5කට වරක් සියලුම Symbols පරීක්ෂා කිරීම
            if now.minute % 5 == 0 and now.minute != last_scan_min:
                last_scan_min = now.minute
                for symbol in SYMBOLS:
                    print(f"[{symbol}] Step 1-4 සඳහා Charts සහ SNR Levels පරීක්ෂා කරමින් පවතී...")
                    w1 = await asyncio.to_thread(fetch_free_candles, symbol, "1wk", "1y", 40)
                    d1 = await asyncio.to_thread(fetch_free_candles, symbol, "1d", "3mo", 40)
                    h1 = await asyncio.to_thread(fetch_free_candles, symbol, "1h", "1mo", 40)
                    m5 = await asyncio.to_thread(fetch_free_candles, symbol, "5m", "5d", 30)
                    h4 = h1[::4]  # 4H Candles සැකසීම

                    w1_levels = scan_snr_levels(w1, "1W")
                    d1_levels = scan_snr_levels(d1, "1D")

                    decision = await analyze_with_ai(symbol, w1_levels, d1_levels, h4, h1, m5)
                    cur_price = m5[-1]["close"] if m5 else 0.0
                    print(f"[{symbol}] මිල: {cur_price} | AI තීරණය: {json.dumps(decision)}")

        except Exception as e:
            print("Scan Error:", e)

        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
