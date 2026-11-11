import os
import re
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
import yfinance as yf
from groq import Groq

def load_clean_groq_key():
    candidates = [os.getenv("AI_API_KEY", ""), os.getenv("GROQ_API_KEY", "")]
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

# සජීවීව ක්‍රියාත්මක වන Signals සහ Win/Loss වාර්තා තබාගන්නා මතකය
ACTIVE_TRADES = {}
SIGNAL_COUNTER = 0
STATS = {"wins": 0, "losses": 0}

def discover_groq_chat_models(groq_client):
    try:
        res = groq_client.models.list()
        all_ids = [m.id for m in res.data if getattr(m, "active", True)]
        skip = ["whisper", "tts", "playai", "guard", "embed", "moderation", "allam", "orpheus"]
        chat_models = [m for m in all_ids if not any(w in m.lower() for w in skip)]
        priority = ["llama-4", "llama-3.3", "deepseek", "qwen", "gpt", "kimi", "llama-3.1", "gemma"]
        def rank(m_id):
            for idx, kw in enumerate(priority):
                if kw in m_id.lower():
                    return idx
            return 999
        chat_models.sort(key=rank)
        return chat_models if chat_models else all_ids
    except Exception as e:
        print(f"Models list error: {e}")
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
    if any(k in s for k in ["JPY", "XAU", "GOLD", "NAS", "BTC", "USTEC"]):
        return 2
    return 5

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

def scan_nearby_snr_levels(candles, timeframe, current_price, max_pips_away, pip_val):
    levels = []
    if not candles or len(candles) < 5:
        return levels
    for i in range(len(candles) - 2):
        c1, c2 = candles[i], candles[i + 1]
        c1_bull = c1["close"] > c1["open"]
        c2_bull = c2["close"] > c2["open"]
        lvl_type = "A_RESISTANCE" if (c1_bull and not c2_bull) else ("V_SUPPORT" if (not c1_bull and c2_bull) else "GAP_SNR")
        lvl_price = c1["close"]
        status = "Fresh_1"
        for j in range(i + 2, len(candles)):
            cj = candles[j]
            b_high = max(cj["open"], cj["close"])
            b_low = min(cj["open"], cj["close"])
            wick_touch = (cj["low"] <= lvl_price <= cj["high"]) and not (b_low < lvl_price < b_high)
            body_break = b_low < lvl_price < b_high
            if status == "Fresh_1" and wick_touch:
                status = "Unfresh"
            elif status in ["Fresh_1", "Unfresh"] and body_break:
                status = "Fresh_SBR_RBS"
            elif status == "Fresh_SBR_RBS" and wick_touch:
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

def calculate_live_atr_pips(m5_candles, pip_val):
    if not m5_candles or len(m5_candles) < 5:
        return 10.0
    ranges = [(c["high"] - c["low"]) / pip_val for c in m5_candles[-10:]]
    return max(round(sum(ranges) / len(ranges), 1), 3.0)

def extract_json_object(raw_text):
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        return json.loads(match.group(0))
    return json.loads(cleaned)

async def analyze_realtime_scalp_ai(groq_client, available_models, symbol, payload):
    cur_price = payload["current_price"]
    pip = get_pip_value(symbol)
    atr_5m = payload["atr_5m_pips"]
    decimals = get_price_decimals(symbol)
    max_offset = round(atr_5m * 2.0 * pip, decimals)

    system_rules = f"""You are a Real-Time Intraday & Scalping MSNR Forex AI Engine.
1. Focus strictly on current 1M, 5M, and 15M price action.
2. NEVER pick old or distant entry prices. entry_price MUST be Current Live Price ({cur_price}) or within {max_offset} of {cur_price}.
3. Calculate realistic fast scalping/intraday sl_price and tp_price based on nearby 5M/15M levels and 5M volatility ({atr_5m} pips) so the trade can complete quickly.
4. For BUY (UP): sl_price < entry_price < tp_price. For SELL (DOWN): tp_price < entry_price < sl_price.
Respond ONLY with valid JSON."""

    user_prompt = f"""Symbol: {symbol} | Live Price: {cur_price} | Pip: {pip} | 5M Volatility: {atr_5m} pips
Nearby 15M Levels: {json.dumps(payload['m15_levels'])}
Nearby 5M Levels: {json.dumps(payload['m5_levels'])}
Last 10 15M Candles: {json.dumps(payload['m15_candles'][-10:])}
Last 10 5M Candles: {json.dumps(payload['m5_candles'][-10:])}
Last 10 1M Candles: {json.dumps(payload['m1_candles'][-10:])}

Return ONLY valid JSON:
{{
  "symbol": "{symbol}",
  "execution_type": "MARKET_NOW",
  "direction_up_down": "UP" | "DOWN",
  "position_type": "LONG (BUY)" | "SHORT (SELL)",
  "setup_zone": "5M_GAP_BREAK" | "5M_QM_APEX" | "15M_SBR_RBS" | "1M_MOMENTUM_BREAK",
  "entry_price": {cur_price},
  "sl_price": 0.0,
  "tp_price": 0.0,
  "reason": "Brief explanation of immediate 1M/5M momentum and nearby target level"
}}"""

    def call_groq():
        for model_name in available_models:
            try:
                comp = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_rules},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                parsed = extract_json_object(comp.choices[0].message.content)
                parsed["_model_used"] = model_name
                return parsed
            except Exception as e:
                print(f"[{symbol}] Model ({model_name}) Skip: {e}")
                time.sleep(1)
                continue
        raise Exception("All Groq models failed.")

    return await asyncio.to_thread(call_groq)

# අලුත් Signal එකක් ලියාපදිංචි කර Console එකේ පෙන්වීම
def register_and_print_signal(symbol, current_price, atr_5m_pips, ai_data):
    global SIGNAL_COUNTER
    SIGNAL_COUNTER += 1
    sig_id = f"#{SIGNAL_COUNTER:03d}"

    sl_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
    pip = get_pip_value(symbol)
    decimals = get_price_decimals(symbol)

    direction_raw = str(ai_data.get("direction_up_down", "DOWN")).upper()
    is_up = "UP" in direction_raw
    direction_label = "UP (ඉහළට)" if is_up else "DOWN (පහළට)"
    pos_label = "LONG (BUY)" if is_up else "SHORT (SELL)"

    zone = str(ai_data.get("setup_zone", "5M_MSNR")).upper()
    model_used = ai_data.get("_model_used", "Groq-AI")

    # සජීවීව Win/Loss බැලීමට පහසු වන ලෙස දැන් පවතින මිලෙන්ම Trade එක ආරම්භ කිරීම
    entry = round(current_price, decimals)

    raw_sl = float(ai_data.get("sl_price") or 0.0)
    raw_tp = float(ai_data.get("tp_price") or 0.0)
    def_sl_pips = max(round(atr_5m_pips * 2.0, 1), 6.0)
    def_tp_pips = max(round(atr_5m_pips * 3.5, 1), 12.0)

    if is_up:
        sl_dist = (entry - raw_sl) / pip if raw_sl > 0 else 0
        sl_price = round(raw_sl, decimals) if (3.0 <= sl_dist <= atr_5m_pips * 5.0) else round(entry - (def_sl_pips * pip), decimals)
        tp_dist = (raw_tp - entry) / pip if raw_tp > 0 else 0
        tp_price = round(raw_tp, decimals) if (5.0 <= tp_dist <= atr_5m_pips * 8.0) else round(entry + (def_tp_pips * pip), decimals)
    else:
        sl_dist = (raw_sl - entry) / pip if raw_sl > 0 else 0
        sl_price = round(raw_sl, decimals) if (3.0 <= sl_dist <= atr_5m_pips * 5.0) else round(entry + (def_sl_pips * pip), decimals)
        tp_dist = (entry - raw_tp) / pip if raw_tp > 0 else 0
        tp_price = round(raw_tp, decimals) if (5.0 <= tp_dist <= atr_5m_pips * 8.0) else round(entry - (def_tp_pips * pip), decimals)

    final_sl_pips = round(abs(entry - sl_price) / pip, 1)
    final_tp_pips = round(abs(tp_price - entry) / pip, 1)

    ACTIVE_TRADES[symbol] = {
        "id": sig_id,
        "symbol": symbol,
        "is_up": is_up,
        "pos_label": pos_label,
        "entry": entry,
        "sl": sl_price,
        "tp": tp_price,
        "sl_pips": final_sl_pips,
        "tp_pips": final_tp_pips,
        "opened_at": sl_time
    }

    print(
        f"\n========================================\n"
        f"🆔 SIGNAL NUMBER  : {sig_id} (LIVE TRACKING STARTED)\n"
        f"🕒 TIME (SL)      : {sl_time}\n"
        f"🤖 AI ENGINE      : {model_used}\n"
        f"🪙 COIN / PAIR    : {symbol}\n"
        f"📈 DIRECTION      : {direction_label} ({zone})\n"
        f"🎯 TRADE TYPE     : {pos_label}\n"
        f"🔵 ENTRY PRICE    : {entry}\n"
        f"🛑 STOP LOSS (SL) : {sl_price}  ({final_sl_pips} Pips)\n"
        f"🟢 TAKE PROFIT(TP): {tp_price}  ({final_tp_pips} Pips)\n"
        f"📝 LIVE REASON    : {ai_data.get('reason', 'N/A')}\n"
        f"========================================"
    )

# ක්‍රියාත්මක වන Signals වල TP හෝ SL වැදී ඇත්දැයි පරීක්ෂා කර වෙනමම කැපී පෙනෙන ලෙස පෙන්වීම
async def check_active_trades_status():
    if not ACTIVE_TRADES:
        return

    closed_symbols = []
    for symbol, trade in list(ACTIVE_TRADES.items()):
        try:
            m1 = await asyncio.to_thread(fetch_live_candles, symbol, "1m", "1d", 3)
            if not m1:
                continue

            last_c = m1[-1]
            c_high = last_c["high"]
            c_low = last_c["low"]
            c_close = last_c["close"]

            outcome = None
            hit_price = c_close
            pips_result = 0.0

            if trade["is_up"]:
                if c_high >= trade["tp"] or c_close >= trade["tp"]:
                    outcome = "WIN"
                    hit_price = trade["tp"]
                    pips_result = trade["tp_pips"]
                elif c_low <= trade["sl"] or c_close <= trade["sl"]:
                    outcome = "LOSS"
                    hit_price = trade["sl"]
                    pips_result = trade["sl_pips"]
            else:
                if c_low <= trade["tp"] or c_close <= trade["tp"]:
                    outcome = "WIN"
                    hit_price = trade["tp"]
                    pips_result = trade["tp_pips"]
                elif c_high >= trade["sl"] or c_close >= trade["sl"]:
                    outcome = "LOSS"
                    hit_price = trade["sl"]
                    pips_result = trade["sl_pips"]

            if outcome:
                closed_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
                if outcome == "WIN":
                    STATS["wins"] += 1
                    header = "🏆✨  SIGNAL RESULT : TRADE WON (TP HIT!)  ✨🏆"
                    pips_str = f"+{pips_result} Pips (WIN ✅)"
                else:
                    STATS["losses"] += 1
                    header = "❌💔  SIGNAL RESULT : TRADE LOST (SL HIT)  💔❌"
                    pips_str = f"-{pips_result} Pips (LOSS ❌)"

                total = STATS["wins"] + STATS["losses"]
                win_rate = round((STATS["wins"] / total) * 100, 1) if total > 0 else 0.0

                # අනෙක් Signals වලින් පැහැදිලිව වෙන් වී පෙනීමට ඉහළින් සහ පහළින් හිස් පේළි 4 බැගින් තබා ඇත
                print(
                    f"\n\n\n\n"
                    f"************************************************************\n"
                    f"{header}\n"
                    f"************************************************************\n"
                    f"   🆔 SIGNAL NUMBER  : {trade['id']}\n"
                    f"   🪙 COIN / PAIR    : {symbol} ({trade['pos_label']})\n"
                    f"   🔵 ENTRY PRICE    : {trade['entry']}\n"
                    f"   🏁 EXIT PRICE     : {hit_price}\n"
                    f"   💰 PROFIT / LOSS  : {pips_str}\n"
                    f"   🕒 CLOSED TIME    : {closed_time}\n"
                    f"   📊 SCOREBOARD     : Wins: {STATS['wins']} | Losses: {STATS['losses']} | Win Rate: {win_rate}%\n"
                    f"************************************************************"
                    f"\n\n\n\n"
                )
                closed_symbols.append(symbol)

        except Exception as e:
            print(f"[{symbol}] Trade Check Error: {e}")

    for sym in closed_symbols:
        ACTIVE_TRADES.pop(sym, None)

async def main():
    print("⚡ MSNR Real-Time AI Bot (With Auto WIN/LOSS Tracker) ආරම්භ විය!")
    if not AI_API_KEY:
        print("❌ ERROR: Groq API Key not found!")
        return

    groq_client = Groq(api_key=AI_API_KEY)
    available_models = discover_groq_chat_models(groq_client)
    if not available_models:
        print("❌ ERROR: No active Groq models found.")
        return

    print(f"✅ Active AI Engine: {available_models[0]}")
    last_scan_slot = -1

    while True:
        try:
            # 1. සෑම තත්පර 30කට වරක් දැනට විවෘතව ඇති Signals WIN ද LOSS ද කියා පරීක්ෂා කිරීම
            await check_active_trades_status()

            # 2. විනාඩි 3කට වරක් (දැනට Active Signal එකක් නොමැති Coins සඳහා පමණක්) අලුත් Signal එකක් සෙවීම
            now = datetime.now(timezone.utc)
            current_slot = (now.hour * 20) + (now.minute // 3)
            if current_slot != last_scan_slot:
                last_scan_slot = current_slot
                for symbol in SYMBOLS:
                    # මෙම Coin එකට දැනටමත් Signal එකක් වැඩ කරමින් පවතී නම් එය අවසන් වන තුරු අලුත් එකක් නොදමයි
                    if symbol in ACTIVE_TRADES:
                        continue
                    try:
                        m15 = await asyncio.to_thread(fetch_live_candles, symbol, "15m", "5d", 35)
                        m5 = await asyncio.to_thread(fetch_live_candles, symbol, "5m", "2d", 35)
                        m1 = await asyncio.to_thread(fetch_live_candles, symbol, "1m", "1d", 30)
                        if not m5 or not m15:
                            continue
                        cur_price = m1[-1]["close"] if m1 else m5[-1]["close"]
                        pip = get_pip_value(symbol)
                        atr_5m_pips = calculate_live_atr_pips(m5, pip)
                        max_search = max(atr_5m_pips * 12, 40.0)
                        payload = {
                            "current_price": cur_price,
                            "atr_5m_pips": atr_5m_pips,
                            "m15_levels": scan_nearby_snr_levels(m15, "15M", cur_price, max_search, pip),
                            "m5_levels": scan_nearby_snr_levels(m5, "5M", cur_price, max_search, pip),
                            "m15_candles": m15,
                            "m5_candles": m5,
                            "m1_candles": m1 if m1 else m5[-10:]
                        }
                        ai_decision = await analyze_realtime_scalp_ai(groq_client, available_models, symbol, payload)
                        register_and_print_signal(symbol, cur_price, atr_5m_pips, ai_decision)
                        await asyncio.sleep(3)
                    except Exception as sym_err:
                        print(f"[{symbol}] Analysis Error: {sym_err}")
        except Exception as e:
            print("Main Loop Error:", e)

        await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
