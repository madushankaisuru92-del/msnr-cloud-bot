import os
import re
import sys
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
import yfinance as yf
from groq import Groq

# Railway Logs වල පේළි කලවම් නොවී 1 සිට 10 දක්වා හරියටම එක දිගට පෙන්වන විශේෂ Function එක
def print_lines_in_order(lines):
    for line in lines:
        print(line, flush=True)
        sys.stdout.flush()
        time.sleep(0.08)

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

ACTIVE_TRADES = {}
TRADE_COUNTER = 0
STATS = {"wins": 0, "losses": 0}

def discover_groq_chat_models(groq_client):
    try:
        res = groq_client.models.list()
        all_ids = [m.id for m in res.data if getattr(m, "active", True)]
        skip = ["whisper", "tts", "playai", "guard", "embed", "moderation", "allam", "orpheus", "8b"]
        chat_models = [m for m in all_ids if not any(w in m.lower() for w in skip)]
        priority = ["120b", "70b", "qwen3-32b", "deepseek", "kimi", "llama-3.3", "llama-4-maverick", "llama-4"]
        def rank(m_id):
            for idx, kw in enumerate(priority):
                if kw in m_id.lower():
                    return idx
            return 999
        chat_models.sort(key=rank)
        return chat_models if chat_models else all_ids
    except Exception as e:
        print(f"Models list error: {e}", flush=True)
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

def fetch_live_candles(symbol, interval, period, count=40):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period=period, interval=interval)
    candles = []
    if df is None or df.empty:
        return candles
    for idx, row in df.tail(count).iterrows():
        candles.append({
            "ts": int(idx.timestamp()),
            "time": str(idx)[11:16],
            "open": round(float(row["Open"]), 5),
            "high": round(float(row["High"]), 5),
            "low": round(float(row["Low"]), 5),
            "close": round(float(row["Close"]), 5)
        })
    return candles

def fetch_exact_live_price(symbol):
    ticker = yf.Ticker(get_yf_ticker(symbol))
    df = ticker.history(period="1d", interval="1m")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None

def compute_ema(prices, period):
    if not prices:
        return 0.0
    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = (p * k) + (ema * (1 - k))
    return ema

def analyze_mathematical_trend(m15_candles, m5_candles, m1_candles, pip_val):
    c5 = m5_candles[:-1] if len(m5_candles) > 5 else m5_candles
    c1 = m1_candles[:-1] if len(m1_candles) > 5 else m1_candles

    closes_5m = [c["close"] for c in c5]
    cur_price = m1_candles[-1]["close"] if m1_candles else closes_5m[-1]

    ema9_5m = compute_ema(closes_5m[-25:], 9)
    ema21_5m = compute_ema(closes_5m[-25:], 21)

    move_15m_pips = (cur_price - closes_5m[-4]) / pip_val if len(closes_5m) >= 4 else 0.0
    move_5m_pips = (cur_price - c1[-5]["open"]) / pip_val if len(c1) >= 5 else 0.0

    bearish_bodies = sum(1 for c in c5[-3:] if c["close"] < c["open"])
    bullish_bodies = sum(1 for c in c5[-3:] if c["close"] > c["open"])

    score = 0
    if cur_price > ema9_5m > ema21_5m:
        score += 2
    elif cur_price < ema9_5m < ema21_5m:
        score -= 2

    if move_15m_pips > 1.5:
        score += 1
    elif move_15m_pips < -1.5:
        score -= 1

    if move_5m_pips > 1.0:
        score += 1
    elif move_5m_pips < -1.0:
        score -= 1

    if bullish_bodies >= 2:
        score += 1
    elif bearish_bodies >= 2:
        score -= 1

    if score >= 2:
        locked_bias = "BULLISH_BUY_ONLY"
    elif score <= -2:
        locked_bias = "BEARISH_SELL_ONLY"
    else:
        locked_bias = "NEUTRAL_CHOPPY"

    return {
        "locked_bias": locked_bias,
        "score": score,
        "ema9_5m": round(ema9_5m, 5),
        "ema21_5m": round(ema21_5m, 5),
        "move_15m_pips": round(move_15m_pips, 1),
        "move_5m_pips": round(move_5m_pips, 1)
    }

def scan_nearby_snr_levels(candles, timeframe, current_price, max_pips_away, pip_val):
    levels = []
    closed_c = candles[:-1] if len(candles) > 5 else candles
    if len(closed_c) < 5:
        return levels

    for i in range(len(closed_c) - 2):
        c1, c2 = closed_c[i], closed_c[i + 1]
        c1_bull = c1["close"] > c1["open"]
        c2_bull = c2["close"] > c2["open"]
        lvl_type = "A_RESISTANCE" if (c1_bull and not c2_bull) else ("V_SUPPORT" if (not c1_bull and c2_bull) else "GAP_SNR")
        lvl_price = c1["close"]
        status = "Fresh_1"
        for j in range(i + 2, len(closed_c)):
            cj = closed_c[j]
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
    closed_c = m5_candles[:-1] if len(m5_candles) > 6 else m5_candles
    if len(closed_c) < 5:
        return 8.0
    ranges = [(c["high"] - c["low"]) / pip_val for c in closed_c[-10:]]
    ranges.sort()
    trimmed = ranges[1:-1] if len(ranges) > 4 else ranges
    return max(round(sum(trimmed) / len(trimmed), 1), 3.0)

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
    trend_info = payload["trend_info"]
    locked_bias = trend_info["locked_bias"]

    system_rules = f"""You are an institutional Real-Time MSNR & Momentum Forex AI Analyst.
STRICT RULES:
1. Quantitative pre-analysis shows `{locked_bias}` (15M move: {trend_info['move_15m_pips']}p, 5M move: {trend_info['move_5m_pips']}p).
   - If `BEARISH_SELL_ONLY`: Output `"direction_up_down": "DOWN"` and `"position_type": "SHORT (SELL)"`.
   - If `BULLISH_BUY_ONLY`: Output `"direction_up_down": "UP"` and `"position_type": "LONG (BUY)"`.
2. Entry Price is strictly {cur_price}.
3. Calculate realistic intraday `sl_price` and `tp_price` based on nearby 5M/15M MSNR levels and 5M volatility ({atr_5m} pips).
   - For SELL (DOWN): `tp_price` < {cur_price} < `sl_price`
   - For BUY (UP): `sl_price` < {cur_price} < `tp_price`
Respond ONLY with valid JSON."""

    user_prompt = f"""Symbol: {symbol} | Live Price: {cur_price} | Pip: {pip} | 5M Volatility: {atr_5m} pips
Trend Info: {json.dumps(trend_info)}
Nearby 15M Levels: {json.dumps(payload['m15_levels'])}
Nearby 5M Levels: {json.dumps(payload['m5_levels'])}
Last 8 Closed 5M Candles: {json.dumps(payload['m5_candles'][-9:-1])}
Last 8 Closed 1M Candles: {json.dumps(payload['m1_candles'][-9:-1])}

Return ONLY valid JSON:
{{
  "symbol": "{symbol}",
  "signal_valid": true,
  "direction_up_down": "UP" | "DOWN",
  "position_type": "LONG (BUY)" | "SHORT (SELL)",
  "setup_zone": "5M_GAP_BREAK" | "5M_QM_APEX" | "15M_SBR_RBS" | "1M_MOMENTUM_BREAK",
  "entry_price": {cur_price},
  "sl_price": 0.0,
  "tp_price": 0.0,
  "reason": "Short 1-sentence reason based on 5M/1M momentum and MSNR target"
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
                    temperature=0.0
                )
                parsed = extract_json_object(comp.choices[0].message.content)
                parsed["_model_used"] = model_name
                return parsed
            except Exception as e:
                print(f"[{symbol}] Model ({model_name}) Skip: {e}", flush=True)
                time.sleep(1)
                continue
        raise Exception("All Groq models failed.")

    return await asyncio.to_thread(call_groq)

def register_and_print_signal(symbol, current_price, atr_5m_pips, trend_info, ai_data):
    global TRADE_COUNTER

    if not ai_data.get("signal_valid", True) or trend_info["locked_bias"] == "NEUTRAL_CHOPPY":
        return

    if trend_info["locked_bias"] == "BEARISH_SELL_ONLY":
        is_up = False
    elif trend_info["locked_bias"] == "BULLISH_BUY_ONLY":
        is_up = True
    else:
        direction_raw = str(ai_data.get("direction_up_down", "DOWN")).upper()
        is_up = "UP" in direction_raw

    TRADE_COUNTER += 1
    t_num = TRADE_COUNTER
    tag = f"[Trade #{t_num}]"

    sl_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
    pip = get_pip_value(symbol)
    decimals = get_price_decimals(symbol)

    direction_label = "UP (ඉහළට)" if is_up else "DOWN (පහළට)"
    pos_label = "LONG (BUY)" if is_up else "SHORT (SELL)"
    zone = str(ai_data.get("setup_zone", "5M_MSNR")).upper()
    model_used = ai_data.get("_model_used", "Groq-AI")

    entry = round(current_price, decimals)
    raw_sl = float(ai_data.get("sl_price") or 0.0)
    raw_tp = float(ai_data.get("tp_price") or 0.0)

    def_sl_pips = max(round(atr_5m_pips * 2.2, 1), 6.0)
    def_tp_pips = max(round(atr_5m_pips * 3.8, 1), 10.0)

    if is_up:
        sl_dist = (entry - raw_sl) / pip if raw_sl > 0 else 0
        sl_price = round(raw_sl, decimals) if (4.0 <= sl_dist <= atr_5m_pips * 5.0) else round(entry - (def_sl_pips * pip), decimals)
        tp_dist = (raw_tp - entry) / pip if raw_tp > 0 else 0
        tp_price = round(raw_tp, decimals) if (6.0 <= tp_dist <= atr_5m_pips * 8.0) else round(entry + (def_tp_pips * pip), decimals)
    else:
        sl_dist = (raw_sl - entry) / pip if raw_sl > 0 else 0
        sl_price = round(raw_sl, decimals) if (4.0 <= sl_dist <= atr_5m_pips * 5.0) else round(entry + (def_sl_pips * pip), decimals)
        tp_dist = (entry - raw_tp) / pip if raw_tp > 0 else 0
        tp_price = round(raw_tp, decimals) if (6.0 <= tp_dist <= atr_5m_pips * 8.0) else round(entry - (def_tp_pips * pip), decimals)

    final_sl_pips = round(abs(entry - sl_price) / pip, 1)
    final_tp_pips = round(abs(tp_price - entry) / pip, 1)
    reason_text = str(ai_data.get("reason", "N/A")).replace("\n", " ")

    # පේළි 10 ලැයිස්තුවක් ලෙස සකස් කිරීම (එවිට 1 සිට 10 දක්වා පිළිවෙළටම මුද්‍රණය වේ)
    signal_lines = [
        f"{tag} 01. ============================================================",
        f"{tag} 02. 🔔 NEW SIGNAL GENERATED : TRADE NUMBER {t_num}",
        f"{tag} 03. 🕒 GIVEN TIME     : {sl_time} (AI: {model_used})",
        f"{tag} 04. 🪙 COIN / PAIR    : {symbol}",
        f"{tag} 05. 📈 DIRECTION      : {direction_label} ({zone})",
        f"{tag} 06. 🎯 TRADE TYPE     : {pos_label}",
        f"{tag} 07. 🔵 ENTRY PRICE    : {entry}",
        f"{tag} 08. 🛑 STOP LOSS (SL) : {sl_price}  ({final_sl_pips} Pips)",
        f"{tag} 09. 🟢 TAKE PROFIT(TP): {tp_price}  ({final_tp_pips} Pips)",
        f"{tag} 10. 📝 AI REASON      : {reason_text}",
        f"{tag} 11. ============================================================"
    ]

    ACTIVE_TRADES[symbol] = {
        "trade_num": t_num,
        "symbol": symbol,
        "is_up": is_up,
        "pos_label": pos_label,
        "entry": entry,
        "sl": sl_price,
        "tp": tp_price,
        "sl_pips": final_sl_pips,
        "tp_pips": final_tp_pips,
        "opened_at_str": sl_time,
        "opened_epoch": time.time(),
        "given_signal_lines": signal_lines[1:-1]
    }

    print_lines_in_order(signal_lines)

async def check_active_trades_status():
    if not ACTIVE_TRADES:
        return

    closed_symbols = []
    for symbol, trade in list(ACTIVE_TRADES.items()):
        try:
            if time.time() - trade["opened_epoch"] < 15:
                continue

            live_price = await asyncio.to_thread(fetch_exact_live_price, symbol)
            if live_price is None:
                continue

            decimals = get_price_decimals(symbol)
            live_price = round(live_price, decimals)
            pip = get_pip_value(symbol)

            outcome = None
            pips_moved = 0.0

            if trade["is_up"]:
                if live_price >= trade["tp"]:
                    outcome = "WIN"
                    pips_moved = round((live_price - trade["entry"]) / pip, 1)
                elif live_price <= trade["sl"]:
                    outcome = "LOSS"
                    pips_moved = round((trade["entry"] - live_price) / pip, 1)
            else:
                if live_price <= trade["tp"]:
                    outcome = "WIN"
                    pips_moved = round((trade["entry"] - live_price) / pip, 1)
                elif live_price >= trade["sl"]:
                    outcome = "LOSS"
                    pips_moved = round((live_price - trade["entry"]) / pip, 1)

            if outcome:
                closed_time = (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")
                t_num = trade["trade_num"]
                r_tag = f"[Result #{t_num}]"

                usd_010 = round(pips_moved * 1.0, 2)
                usd_001 = round(pips_moved * 0.10, 2)

                if outcome == "WIN":
                    STATS["wins"] += 1
                    title_banner = f"{r_tag} 🏆 TRADE NUMBER {t_num} : WIN ✅"
                    amount_line = f"{r_tag} 💰 WON AMOUNT     : +{pips_moved} Pips  (💵 +${usd_010:.2f} on 0.10 Lot | +${usd_001:.2f} on 0.01 Lot)"
                else:
                    STATS["losses"] += 1
                    title_banner = f"{r_tag} ❌ TRADE NUMBER {t_num} : LOSS ❌"
                    amount_line = f"{r_tag} 💸 LOST AMOUNT    : -{pips_moved} Pips  (💵 -${usd_010:.2f} on 0.10 Lot | -${usd_001:.2f} on 0.01 Lot)"

                total = STATS["wins"] + STATS["losses"]
                win_rate = round((STATS["wins"] / total) * 100, 1) if total > 0 else 0.0

                result_lines = [
                    "   ",
                    "   ",
                    f"{r_tag} ************************************************************",
                    title_banner,
                    f"{r_tag} ************************************************************",
                    amount_line,
                    f"{r_tag} 🏁 LIVE EXIT PRICE: {live_price}  (Closed at {closed_time})",
                    f"{r_tag} 📊 REAL SCOREBOARD: Wins: {STATS['wins']} | Losses: {STATS['losses']} | Win Rate: {win_rate}%",
                    f"{r_tag} ------------------------------------------------------------",
                    f"{r_tag} 📌 GIVEN SIGNAL (මුලින් ලබාදුන් සිග්නල් එක):"
                ]
                for gl in trade["given_signal_lines"]:
                    result_lines.append(f"{r_tag}    {gl}")
                result_lines.append(f"{r_tag} ************************************************************")
                result_lines.append("   ")
                result_lines.append("   ")

                await asyncio.to_thread(print_lines_in_order, result_lines)
                closed_symbols.append(symbol)

        except Exception as e:
            print(f"[{symbol}] Live Price Check Error: {e}", flush=True)

    for sym in closed_symbols:
        ACTIVE_TRADES.pop(sym, None)

async def main():
    print("⚡ MSNR Real-Time Ordered-Log Bot ආරම්භ විය!", flush=True)
    if not AI_API_KEY:
        print("❌ ERROR: Groq API Key not found!", flush=True)
        return

    groq_client = Groq(api_key=AI_API_KEY)
    available_models = discover_groq_chat_models(groq_client)
    if not available_models:
        print("❌ ERROR: No active Groq models found.", flush=True)
        return

    print(f"✅ Primary AI Engine: {available_models[0]}", flush=True)
    last_scan_slot = -1

    while True:
        try:
            await check_active_trades_status()

            now = datetime.now(timezone.utc)
            current_slot = (now.hour * 20) + (now.minute // 3)
            if current_slot != last_scan_slot:
                last_scan_slot = current_slot
                for symbol in SYMBOLS:
                    if symbol in ACTIVE_TRADES:
                        continue
                    try:
                        m15 = await asyncio.to_thread(fetch_live_candles, symbol, "15m", "5d", 35)
                        m5 = await asyncio.to_thread(fetch_live_candles, symbol, "5m", "2d", 35)
                        m1 = await asyncio.to_thread(fetch_live_candles, symbol, "1m", "1d", 30)
                        if not m5 or not m15 or not m1:
                            continue

                        cur_price = m1[-1]["close"]
                        pip = get_pip_value(symbol)
                        atr_5m_pips = calculate_live_atr_pips(m5, pip)
                        trend_info = analyze_mathematical_trend(m15, m5, m1, pip)

                        if trend_info["locked_bias"] == "NEUTRAL_CHOPPY":
                            print(f"⏳ [{symbol}] Live: {cur_price} | Market Flat/Choppy -> Waiting for clear breakout...", flush=True)
                            continue

                        max_search = max(atr_5m_pips * 10, 35.0)
                        payload = {
                            "current_price": cur_price,
                            "atr_5m_pips": atr_5m_pips,
                            "trend_info": trend_info,
                            "m15_levels": scan_nearby_snr_levels(m15, "15M", cur_price, max_search, pip),
                            "m5_levels": scan_nearby_snr_levels(m5, "5M", cur_price, max_search, pip),
                            "m5_candles": m5,
                            "m1_candles": m1
                        }
                        ai_decision = await analyze_realtime_scalp_ai(groq_client, available_models, symbol, payload)
                        await asyncio.to_thread(register_and_print_signal, symbol, cur_price, atr_5m_pips, trend_info, ai_decision)
                        await asyncio.sleep(2)
                    except Exception as sym_err:
                        print(f"[{symbol}] Analysis Error: {sym_err}", flush=True)
        except Exception as e:
            print("Main Loop Error:", e, flush=True)

        await asyncio.sleep(15)

if __name__ == "__main__":
    asyncio.run(main())
