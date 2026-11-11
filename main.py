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

# 3. සජීවී Candles (1m, 5m, 15m, 1h) ලබාගැනීම
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

# 4. දැන් පවතින මිලට ඉතාම ආසන්න Fresh SNR Levels පමණක් තෝරාගැනීම (පරණ Levels ඉවත් කෙරේ)
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

        # Dead නොවන සහ දැන් මිලට ඉතා ආසන්නයේ ඇති Levels පමණක් තබාගැනීම
        dist_pips = abs(current_price - lvl_price) / pip_val
        if status != "Dead" and dist_pips <= max_pips_away:
            levels.append({
                "tf": timeframe,
                "type": lvl_type,
                "price": round(lvl_price, 5),
                "status": status,
                "dist_pips": round(dist_pips, 1)
            })

    # දැන් මිලට ළඟම Levels 8 පමණක් තෝරාගැනීම
    levels.sort(key=lambda x: x["dist_pips"])
    return levels[:8]

# 5M Candles වලින් මේ මොහොතේ මාකට් වේගය (ATR Volatility in Pips) ගණනය කිරීම
def calculate_live_atr_pips(m5_candles, pip_val):
    if not m5_candles or len(m5_candles) < 5:
        return 10.0
    ranges = [(c["high"] - c["low"]) / pip_val for c in m5_candles[-10:]]
    avg_5m_pips = sum(ranges) / len(ranges)
    return max(round(avg_5m_pips, 1), 3.0)

def extract_json_object(raw_text):
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        return json.loads(match.group(0))
    return json.loads(cleaned)

# 5. Real-Time Intraday/Scalping AI Analysis (1m, 5m, 15m මත පදනම්ව)
async def analyze_realtime_scalp_ai(groq_client, available_models, symbol, payload):
    cur_price = payload["current_price"]
    pip = get_pip_value(symbol)
    atr_5m = payload["atr_5m_pips"]
    decimals = get_price_decimals(symbol)

    # දැන් මිලේ සිට උපරිම Entry පරතරය (විනාඩි කිහිපයකින් වැදිය හැකි දුරක් පමණි)
    max_entry_offset = round(atr_5m * 2.5 * pip, decimals)

    system_rules = f"""You are a Real-Time High-Frequency Intraday & Scalping MSNR Forex AI Engine.
CRITICAL RULES FOR REAL-TIME TRADING (NO MULTI-DAY SWING TRADES):
1. Focus strictly on what the market is doing RIGHT NOW on the 1M, 5M, and 15M candles.
2. NEVER pick old or distant entry prices! Your `entry_price` MUST be either:
   - The exact `Current Live Price` ({cur_price}) for an immediate Market execution, OR
   - A micro-pullback level within {max_entry_offset} ({round(atr_5m * 2.5, 1)} pips) of {cur_price} that can be triggered within the next few minutes.
3. NEVER use a fixed 200-pip Take Profit! Calculate realistic intraday `sl_price` and `tp_price` based on immediate 5M/15M MSNR structure and current 5M volatility (5M candle average range is {atr_5m} pips):
   - Stop Loss (`sl_price`): Place just behind the immediate 5M/15M invalidation wick/SNR level (typically {round(atr_5m * 2, 1)} to {round(atr_5m * 4, 1)} pips away from entry).
   - Take Profit (`tp_price`): Place at the next opposing 5M/15M Fresh SNR level or liquidity high/low (typically {round(atr_5m * 4, 1)} to {round(atr_5m * 10, 1)} pips away, maintaining at least 1:1.5 to 1:3 Risk-to-Reward).
4. Direction alignment:
   - If BUY (UP): `sl_price` < `entry_price` < `tp_price`
   - If SELL (DOWN): `tp_price` < `entry_price` < `sl_price`
Respond ONLY with valid JSON."""

    user_prompt = f"""Symbol: {symbol}
Current Live Price: {cur_price}
Pip Value: {pip} | Current 5M Average Candle Volatility: {atr_5m} pips

Live Intraday Data (Only nearby active levels):
- Nearby 15M SNR Levels: {json.dumps(payload['m15_levels'])}
- Nearby 5M SNR Levels: {json.dumps(payload['m5_levels'])}
- Last 10 15M Candles: {json.dumps(payload['m15_candles'][-10:])}
- Last 12 5M Candles: {json.dumps(payload['m5_candles'][-12:])}
- Last 12 1M Candles (Immediate Momentum): {json.dumps(payload['m1_candles'][-12:])}

Return ONLY valid JSON with these exact keys:
{{
  "symbol": "{symbol}",
  "execution_type": "MARKET_NOW" | "NEAR_PULLBACK_LIMIT",
  "direction_up_down": "UP" | "DOWN",
  "position_type": "LONG (BUY)" | "SHORT (SELL)",
  "setup_zone": "5M_GAP_BREAK" | "5M_QM_APEX" | "15M_SBR_RBS" | "1M_MOMENTUM_BREAK",
  "entry_price": {cur_price},
  "sl_price": 0.0,
  "tp_price": 0.0,
  "reason": "Explain the immediate 1M/5M candle momentum, why entry is right here, and which nearby 5M/15M level is used for TP/SL"
}}"""

    def call_groq():
        for model_name in available_models:
            try:
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_rules},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                raw_content = completion.choices[0].message.content
                parsed = extract_json_object(raw_content)
                parsed["_model_used"] = model_name
                return parsed
            except Exception as e:
                print(f"[{symbol}] Model ({model_name}) Skip: {e}")
                time.sleep(1)
                continue
        raise Exception("Groq Models වලින් පිළිතුරක් ලබාගත නොහැකි විය.")

    return await asyncio.to_thread(call_groq)

# 6. පරණ Entry/TP වැරදි සම්පූර්ණයෙන්ම වළක්වන Real-Time Signal Output එක
def print_realtime_signal(symbol, current_price, atr_5m_pips, ai_data):
    sl_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
    pip = get_pip_value(symbol)
    decimals = get_price_decimals(symbol)

    direction_raw = str(ai_data.get("direction_up_down", "DOWN")).upper()
    is_up = "UP" in direction_raw
    direction_label = "UP (ඉහළට)" if is_up else "DOWN (පහළට)"
    pos_label = "LONG (BUY)" if is_up else "SHORT (SELL)"

    exec_type = str(ai_data.get("execution_type", "MARKET_NOW")).upper()
    zone = str(ai_data.get("setup_zone", "5M_MSNR")).upper()
    model_used = ai_data.get("_model_used", "Groq-AI")

    entry = float(ai_data.get("entry_price") or current_price)
    # ආරක්ෂක පරීක්ෂාව: AI දුන් Entry එක දැන් මිලට වඩා ගොඩක් දුර නම් (පරණ මිලක් නම්), එය දැන් මිලටම සකසන්න
    max_allowed_entry_dist = max(atr_5m_pips * 3.0, 15.0) * pip
    if entry <= 0 or abs(entry - current_price) > max_allowed_entry_dist:
        entry = current_price
        exec_type = "MARKET_NOW (LIVE PRICE)"

    raw_sl = float(ai_data.get("sl_price") or 0.0)
    raw_tp = float(ai_data.get("tp_price") or 0.0)

    # Dynamic Volatility SL සහ TP පරාසයන් (Pips 200 ස්ථිර අගය වෙනුවට එම වෙලාවේ මාකට් වේගය අනුව)
     realistic_sl_pips = max(round(atr_5m_pips * 2.5, 1), 8.0)
    realistic_tp_pips = max(round(atr_5m_pips * 5.5, 1), 18.0)

    if is_up:
        # BUY සඳහා SL පරීක්ෂාව
        sl_dist = (entry - raw_sl) / pip if raw_sl > 0 else 0
        if 4.0 <= sl_dist <= (atr_5m_pips * 6.0):
            sl_price = round(raw_sl, decimals)
        else:
            sl_price = round(entry - (realistic_sl_pips * pip), decimals)

        # BUY සඳහා TP පරීක්ෂාව
        tp_dist = (raw_tp - entry) / pip if raw_tp > 0 else 0
        if 8.0 <= tp_dist <= (atr_5m_pips * 12.0):
            tp_price = round(raw_tp, decimals)
        else:
            tp_price = round(entry + (realistic_tp_pips * pip), decimals)
    else:
        # SELL සඳහා SL පරීක්ෂාව
        sl_dist = (raw_sl - entry) / pip if raw_sl > 0 else 0
        if 4.0 <= sl_dist <= (atr_5m_pips * 6.0):
            sl_price = round(raw_sl, decimals)
        else:
            sl_price = round(entry + (realistic_sl_pips * pip), decimals)

        # SELL සඳහා TP පරීක්ෂාව
        tp_dist = (entry - raw_tp) / pip if raw_tp > 0 else 0
        if 8.0 <= tp_dist <= (atr_5m_pips * 12.0):
            tp_price = round(raw_tp, decimals)
        else:
            tp_price = round(entry - (realistic_tp_pips * pip), decimals)

    final_sl_pips = round(abs(entry - sl_price) / pip, 1)
    final_tp_pips = round(abs(tp_price - entry) / pip, 1)

    print(
        f"\n========================================\n"
        f"🕒 TIME (SL)      : {sl_time}\n"
        f"🤖 AI ENGINE      : {model_used}\n"
        f"🪙 COIN / PAIR    : {symbol}\n"
        f"💵 LIVE PRICE     : {round(current_price, decimals)}\n"
        f"⚡ EXECUTION      : {exec_type} ({zone})\n"
        f"📈 DIRECTION      : {direction_label}\n"
        f"🎯 TRADE TYPE     : {pos_label}\n"
        f"🔵 ENTRY PRICE    : {round(entry, decimals)}\n"
        f"🛑 STOP LOSS (SL) : {sl_price}  ({final_sl_pips} Pips)\n"
        f"🟢 TAKE PROFIT(TP): {tp_price}  ({final_tp_pips} Pips)\n"
        f"📝 LIVE REASON    : {ai_data.get('reason', 'N/A')}\n"
        f"========================================"
    )

# 7. ප්‍රධාන Real-Time Cloud Loop එක (සෑම විනාඩි 3කට වරක්)
async def main():
    print("⚡ MSNR Real-Time Scalping & Intraday AI Bot ආරම්භ විය!")

    if not AI_API_KEY:
        print("❌ ERROR: Railway Variables තුළ 'gsk_' වලින් පටන් ගන්නා Groq Key එකක් හමු නොවීය!")
        return

    groq_client = Groq(api_key=AI_API_KEY)
    available_models = discover_groq_chat_models(groq_client)

    if not available_models:
        print("❌ ERROR: Groq Models ලබාගත නොහැකි විය.")
        return

    print(f"✅ Active AI Engine: {available_models[0]} (Scanning every 3 minutes)")
    last_scan_slot = -1

    while True:
        try:
            now = datetime.now(timezone.utc)
            # සෑම විනාඩි 3කට වරක් Real-Time Scan කිරීම
            current_slot = (now.hour * 20) + (now.minute // 3)

            if current_slot != last_scan_slot:
                last_scan_slot = current_slot

                for symbol in SYMBOLS:
                    try:
                        m15 = await asyncio.to_thread(fetch_live_candles, symbol, "15m", "5d", 35)
                        m5 = await asyncio.to_thread(fetch_live_candles, symbol, "5m", "2d", 35)
                        m1 = await asyncio.to_thread(fetch_live_candles, symbol, "1m", "1d", 30)

                        if not m5 or not m15:
                            print(f"[{symbol}] Live Candle දත්ත ලබාගැනීමට නොහැකි විය.")
                            continue

                        cur_price = m1[-1]["close"] if m1 else m5[-1]["close"]
                        pip = get_pip_value(symbol)
                        atr_5m_pips = calculate_live_atr_pips(m5, pip)

                        # දැන් පවතින මිලේ සිට Pips කිහිපයක් ඇතුළත ඇති අලුත්ම Levels පමණක් ගැනීම
                        max_search_pips = max(atr_5m_pips * 12, 40.0)
                        m15_levels = scan_nearby_snr_levels(m15, "15M", cur_price, max_search_pips, pip)
                        m5_levels = scan_nearby_snr_levels(m5, "5M", cur_price, max_search_pips, pip)

                        payload = {
                            "current_price": cur_price,
                            "atr_5m_pips": atr_5m_pips,
                            "m15_levels": m15_levels,
                            "m5_levels": m5_levels,
                            "m15_candles": m15,
                            "m5_candles": m5,
                            "m1_candles": m1 if m1 else m5[-10:]
                        }

                        ai_decision = await analyze_realtime_scalp_ai(groq_client, available_models, symbol, payload)
                        print_realtime_signal(symbol, cur_price, atr_5m_pips, ai_decision)
                        await asyncio.sleep(3)

                    except Exception as sym_err:
                        print(f"[{symbol}] Analysis Error: {sym_err}")

        except Exception as e:
            print("Main Loop Error:", e)

        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
