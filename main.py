import os
import re
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
import yfinance as yf
from groq import Groq

# 1. Railway Variables තුළ ඇති Key එකේ සියලුම දෝෂ (Newlines, Spaces, Quotes) ස්වයංක්‍රීයව පිරිසිදු කර ලබාගැනීම
def load_clean_groq_key():
    candidates = [
        os.getenv("AI_API_KEY", ""),
        os.getenv("GROQ_API_KEY", "")
    ]
    # Railway හි වෙනත් නමකින් හෝ හිස්තැනක් සහිතව Variable එක තිබුණද එය ස්වයංක්‍රීයව සොයාගැනීම
    for k, v in os.environ.items():
        if v and "gsk_" in v:
            candidates.append(v)

    for raw in candidates:
        if not raw:
            continue
        # සියලුම හිස්තැන්, පේළි කැඩීම් (\n, \r) සහ උද්ධෘත ලකුණු ඉවත් කිරීම
        compact = re.sub(r'[\s"\']+', '', raw)
        match = re.search(r'(gsk_[A-Za-z0-9_\-]{20,})', compact)
        if match:
            return match.group(1)
    return ""

AI_API_KEY = load_clean_groq_key()
RAW_SYMBOLS = os.getenv("TRADING_SYMBOLS", "XAUUSDm,GBPUSDm,GBPJPYm,USDCADm")
SYMBOLS = [s.strip() for s in RAW_SYMBOLS.split(",") if s.strip()]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
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

# 2. සජීවී Candles නිවැරදිව ලබාගැනීම
def fetch_free_candles(symbol, interval, period, count=35):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period=period, interval=interval)
    candles = []
    if df is None or df.empty:
        return candles
    for idx, row in df.tail(count).iterrows():
        candles.append({
            "time": str(idx)[:16],
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5)
        })
    return candles

# 3. MSNR Step 1: A-Shape, V-Shape සහ Gap Levels ගණනය (2-Time Usage Rule)
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

# 4. Groq AI මගින් MSNR නීති 5ට අනුව ගැඹුරින් විශ්ලේෂණය කිරීම
async def analyze_with_groq_ai(groq_client, symbol, payload):
    cur_price = payload["current_price"]
    pip = get_pip_value(symbol)

    system_rules = """You are an institutional MALAYSIAN SNR (MSNR 1.0) Forex AI Trading Engine.
Strictly apply these MSNR rules:
1. Storyline (1+2+3 Rule): Price touches a Weekly/Daily Fresh Level (Fresh_Use_1 or Fresh_Use_2) -> Rejects with a wick close -> 1 TF lower (4H/Daily) External Level is broken by a full Candle Body. If Weekly and Daily conflict, follow Daily direction.
2. HNS Setup Zones: From the 4H breakout structure, identify the best entry zone: APEX (extreme tip), QM (left shoulder), GAP, or SBR_RBS (broken external level).
3. Refining: Check 1H Fresh levels (must be ABOVE 4H level for SELL, UNDER 4H level for BUY).
4. Strict Price & Direction Rules:
   - If Bearish: direction_up_down = "DOWN", position_type = "SHORT (SELL)", entry_price must be a valid resistance level, and target_price MUST be strictly LOWER than entry_price.
   - If Bullish: direction_up_down = "UP", position_type = "LONG (BUY)", entry_price must be a valid support level, and target_price MUST be strictly HIGHER than entry_price."""

    user_prompt = f"""Analyze Symbol: {symbol}
Current Live Price: {cur_price}
Pip Value: {pip}

Market Structure Data:
- Weekly Active SNR Levels: {json.dumps(payload['w1_levels'])}
- Daily Active SNR Levels: {json.dumps(payload['d1_levels'])}
- 4H Active SNR Levels: {json.dumps(payload['h4_levels'])}
- 1H Active SNR Levels: {json.dumps(payload['h1_levels'])}
- Last 10 4H Candles: {json.dumps(payload['h4_candles'][-10:])}
- Last 10 1H Candles: {json.dumps(payload['h1_candles'][-10:])}
- Last 10 5M Candles: {json.dumps(payload['m5_candles'][-10:])}

Return ONLY a valid JSON object with these exact keys:
{{
  "symbol": "{symbol}",
  "trade_status": "ACTIVE_ENTRY_NOW" | "WAITING_FOR_PULLBACK_LIMIT",
  "direction_up_down": "UP" | "DOWN",
  "position_type": "LONG (BUY)" | "SHORT (SELL)",
  "setup_zone": "APEX" | "QM" | "GAP" | "SBR_RBS",
  "entry_price": 0.0,
  "target_price": 0.0,
  "reason": "Clear explanation of Daily/Weekly rejection, 4H breakout, and selected HNS zone"
}}"""

    def call_groq():
        for model_name in GROQ_MODELS:
            try:
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_rules},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return json.loads(completion.choices[0].message.content)
            except Exception as e:
                print(f"[{symbol}] Groq Model ({model_name}) Error: {e}")
                time.sleep(2)
                continue
        raise Exception("Groq API පරීක්ෂාව අසාර්ථක විය.")

    return await asyncio.to_thread(call_groq)

# 5. තනි Entry, SL සහ TP මිලක් පමණක් පැහැදිලිව පෙන්වීම
def print_clean_signal(symbol, current_price, ai_data):
    sl_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
    pip = get_pip_value(symbol)
    decimals = get_price_decimals(symbol)

    direction_raw = str(ai_data.get("direction_up_down", "DOWN")).upper()
    is_up = "UP" in direction_raw
    direction_label = "UP (ඉහළට)" if is_up else "DOWN (පහළට)"
    pos_label = "LONG (BUY)" if is_up else "SHORT (SELL)"

    zone = str(ai_data.get("setup_zone", "APEX")).upper()
    status = str(ai_data.get("trade_status", "WAITING_FOR_PULLBACK_LIMIT"))

    entry = float(ai_data.get("entry_price") or 0.0)
    if entry <= 0:
        entry = current_price

    sl_pips = 30 if "APEX" in zone else 20
    default_tp_pips = 200
    ai_target = float(ai_data.get("target_price") or 0.0)

    if is_up:
        sl_price = round(entry - (sl_pips * pip), decimals)
        if ai_target > entry + (50 * pip):
            tp_price = round(ai_target, decimals)
        else:
            tp_price = round(entry + (default_tp_pips * pip), decimals)
    else:
        sl_price = round(entry + (sl_pips * pip), decimals)
        if 0 < ai_target < entry - (50 * pip):
            tp_price = round(ai_target, decimals)
        else:
            tp_price = round(entry - (default_tp_pips * pip), decimals)

    print(
        f"\n========================================\n"
        f"🕒 TIME           : {sl_time}\n"
        f"🪙 COIN / PAIR    : {symbol}\n"
        f"💵 CURRENT PRICE  : {round(current_price, decimals)}\n"
        f"📊 SETUP STATUS   : {status} ({zone})\n"
        f"📈 DIRECTION      : {direction_label}\n"
        f"🎯 TRADE TYPE     : {pos_label}\n"
        f"🔵 ENTRY PRICE    : {round(entry, decimals)}\n"
        f"🛑 STOP LOSS (SL) : {sl_price}\n"
        f"🟢 TAKE PROFIT(TP): {tp_price}\n"
        f"📝 AI REASON      : {ai_data.get('reason', 'N/A')}\n"
        f"========================================"
    )

# 6. ප්‍රධාන Cloud Loop එක
async def main():
    print("🚀 MSNR Groq-AI Cloud Bot ආරම්භ විය!")

    if not AI_API_KEY:
        print("❌ ERROR: Railway Variables තුළ 'gsk_' වලින් පටන් ගන්නා Groq Key එකක් හමු නොවීය!")
        print("දැනට පවතින Variable නම්:", list(os.environ.keys()))
        return

    masked = f"{AI_API_KEY[:8]}...{AI_API_KEY[-4:]}"
    print(f"🔑 Loaded Clean Groq Key: {masked} (Length: {len(AI_API_KEY)} chars)")

    groq_client = Groq(api_key=AI_API_KEY)
    last_scan_slot = -1

    while True:
        try:
            now = datetime.now(timezone.utc)
            current_slot = (now.hour * 4) + (now.minute // 15)

            if current_slot != last_scan_slot:
                last_scan_slot = current_slot

                for symbol in SYMBOLS:
                    try:
                        w1 = await asyncio.to_thread(fetch_free_candles, symbol, "1wk", "1y", 35)
                        d1 = await asyncio.to_thread(fetch_free_candles, symbol, "1d", "3mo", 35)
                        h1 = await asyncio.to_thread(fetch_free_candles, symbol, "1h", "1mo", 40)
                        m5 = await asyncio.to_thread(fetch_free_candles, symbol, "5m", "5d", 25)

                        if not h1 or not d1:
                            print(f"[{symbol}] Candle දත්ත ලබාගැනීමට නොහැකි විය.")
                            continue

                        h4 = h1[::4]
                        cur_price = m5[-1]["close"] if m5 else h1[-1]["close"]

                        payload = {
                            "current_price": cur_price,
                            "w1_levels": scan_snr_levels(w1, "1W"),
                            "d1_levels": scan_snr_levels(d1, "1D"),
                            "h4_levels": scan_snr_levels(h4, "4H"),
                            "h1_levels": scan_snr_levels(h1, "1H"),
                            "h4_candles": h4,
                            "h1_candles": h1,
                            "m5_candles": m5
                        }

                        ai_decision = await analyze_with_groq_ai(groq_client, symbol, payload)
                        print_clean_signal(symbol, cur_price, ai_decision)
                        await asyncio.sleep(4)

                    except Exception as sym_err:
                        print(f"[{symbol}] Analysis Error: {sym_err}")

        except Exception as e:
            print("Main Loop Error:", e)

        await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
