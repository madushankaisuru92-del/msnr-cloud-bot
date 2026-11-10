import os
import json
import asyncio
import sqlite3
from datetime import datetime, timezone
import google.generativeai as genai
from metaapi_cloud_sdk import MetaApi

# 1. Cloud Environment Variables ලබාගැනීම
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("MT5_ACCOUNT_ID")
AI_API_KEY = os.getenv("AI_API_KEY")
SYMBOLS = [s.strip() for s in os.getenv("TRADING_SYMBOLS", "XAUUSD,GBPUSD").split(",")]

genai.configure(api_key=AI_API_KEY)
ai_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={"response_mime_type": "application/json"}
)

# 2. SQLite Database සැකසීම (Step 1: SNR Levels සහ Storylines)
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
        CREATE TABLE IF NOT EXISTS active_setups (
            symbol TEXT PRIMARY KEY, storyline TEXT,
            apex REAL, qm REAL, gap REAL, sbr_rbs REAL,
            refined_1h REAL, opp_fresh_htf REAL, updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

# 3. Step 1: A-Shape, V-Shape සහ Gap Levels ගණිතමයව හඳුනාගැනීම (Fresh_Use_1, Unfresh_1, Fresh_Use_2, Dead)
def scan_snr_levels(candles, timeframe):
    levels = []
    if not candles or len(candles) < 5:
        return levels

    for i in range(len(candles) - 3):
        c1 = candles[i]
        c2 = candles[i + 1]
        c1_bull = c1['close'] > c1['open']
        c2_bull = c2['close'] > c2['open']

        lvl_type = None
        lvl_price = c1['close']

        if c1_bull and not c2_bull:
            lvl_type = "A_SHAPE"
        elif not c1_bull and c2_bull:
            lvl_type = "V_SHAPE"
        elif c1_bull == c2_bull:
            lvl_type = "GAP"

        if not lvl_type:
            continue

        status = "Fresh_Use_1"
        touches = 0

        # පසුව ආ Candles මගින් Level එකේ තත්ත්වය පරීක්ෂා කිරීම (2-Time Usage Rule)
        for j in range(i + 2, len(candles)):
            cj = candles[j]
            body_high = max(cj['open'], cj['close'])
            body_low = min(cj['open'], cj['close'])

            # Wick Touch පරීක්ෂාව
            wick_touched = (cj['low'] <= lvl_price <= cj['high']) and not (body_low < lvl_price < body_high)
            # Candle Body Breakout පරීක්ෂාව
            body_broken = body_low < lvl_price < body_high

            if status == "Fresh_Use_1" and wick_touched:
                status = "Unfresh_1"
                touches = 1
            elif status in ["Fresh_Use_1", "Unfresh_1"] and body_broken:
                status = "Fresh_Use_2"
            elif status == "Fresh_Use_2" and wick_touched:
                status = "Dead"
                touches = 2
                break

        if status != "Dead":
            levels.append({
                "timeframe": timeframe,
                "type": lvl_type,
                "price": round(lvl_price, 5),
                "status": status
            })

    return levels[-15:]  # අවසාන වලංගු Levels 15 පමණක් තබාගැනීම

# 4. Step 2 & Step 3: AI මගින් Storyline, HNS, Refining, Trendline සහ Asian High/Low විශ්ලේෂණය
async def analyze_with_ai(symbol, w1_levels, d1_levels, h4_candles, h1_candles, m5_candles):
    prompt = f"""
    You are an automated MALAYSIAN SNR (MSNR 1.0) Forex Trading Engine analyzing {symbol}.
    Strictly apply these 5 rules:
    1. Storyline requires: Price touches a Weekly/Daily Fresh Level (Fresh_Use_1 or Fresh_Use_2) -> Wick Rejection close -> 1 TF lower (Daily/4H) External Level broken by Candle Body. If Weekly and Daily conflict, follow Daily direction.
    2. Identify 4H HNS zones from the breakout: APEX, QM (Left Shoulder), GAP, and SBR_RBS.
    3. Check 1H Refining Level (Must be ABOVE 4H level for SELL, UNDER 4H level for BUY), 4H/1H Trendline (Outer with Engulfing+BOS or Break & Retest), and Asian Session (00:00-09:00 UTC) High/Low sweep.
    4. Entry trigger rules:
       - If price is for APEX: action = "PLACE_APEX_LIMIT" (SL = 30 pips).
       - If price is touching QM, GAP, or SBR_RBS and 5M candles show a 5M Gap Body Break, QM retest, or Trendline Body Break: action = "EXECUTE_CONFIRMED_MARKET" (SL = 20 pips).
       - Otherwise: action = "WAIT".

    Data Provided:
    Weekly Active Levels: {json.dumps(w1_levels)}
    Daily Active Levels: {json.dumps(d1_levels)}
    Last 20 4H Candles: {json.dumps(h4_candles[-20:], default=str)}
    Last 20 1H Candles: {json.dumps(h1_candles[-20:], default=str)}
    Last 15 5M Candles: {json.dumps(m5_candles[-15:], default=str)}

    Return ONLY valid JSON in this exact format:
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
    response = await asyncio.to_thread(ai_model.generate_content, prompt)
    return json.loads(response.text)

# 5. Pip අගය ගණනය කිරීම
def get_pip_value(symbol):
    if "JPY" in symbol:
        return 0.01
    if "XAU" in symbol or "GOLD" in symbol:
        return 0.10
    if "NAS" in symbol or "BTC" in symbol:
        return 1.0
    return 0.0001

# 6. Step 5: Trade Management (30 Pips Breakeven & 200 Pips / Next Fresh Level TP)
async def manage_open_trades(connection, symbol, current_bid, current_ask, opp_fresh_level):
    positions = await connection.get_positions()
    pip = get_pip_value(symbol)

    for pos in positions:
        if pos['symbol'] != symbol:
            continue

        open_price = pos['openPrice']
        pos_id = pos['id']
        is_buy = pos['type'] == 'POSITION_TYPE_BUY'
        current_price = current_bid if is_buy else current_ask
        profit_pips = ((current_price - open_price) / pip) if is_buy else ((open_price - current_price) / pip)

        # +30 Pips වූ විට SL එක Breakeven කිරීම
        if profit_pips >= 30:
            sl_current = pos.get('stopLoss', 0)
            if (is_buy and sl_current < open_price) or (not is_buy and (sl_current == 0 or sl_current > open_price)):
                await connection.modify_position(pos_id, open_price, pos.get('takeProfit'))
                print(f"[{symbol}] +30 Pips ලාභයි! SL එක Breakeven ({open_price}) වෙත ගෙන ආවා.")

        # +200 Pips හෝ Opposite Fresh Level වැදුණු විට Close කිරීම
        hit_opp_level = False
        if opp_fresh_level and opp_fresh_level > 0:
            if (is_buy and current_price >= opp_fresh_level) or (not is_buy and current_price <= opp_fresh_level):
                hit_opp_level = True

        if profit_pips >= 200 or hit_opp_level:
            await connection.close_position(pos_id)
            print(f"[{symbol}] Target සම්පූර්ණයි ({profit_pips:.1f} pips). Trade එක Close කරන ලදී!")

# 7. ප්‍රධාන Cloud Loop එක
async def main():
    init_db()
    api = MetaApi(token=METAAPI_TOKEN)
    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)

    if account.state != 'DEPLOYED':
        await account.deploy()
    print("MT5 Cloud ගිණුම සම්බන්ධ වෙමින් පවතී...")
    await account.wait_connected()

    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    print("MT5 Cloud ගිණුම සාර්ථකව සම්බන්ධ විය!")

    last_ai_scan_minute = -1
    daily_losses = 0
    current_day = datetime.now(timezone.utc).day

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.day != current_day:
                daily_losses = 0
                current_day = now.day

            # දිනකට Losses 2ක් නම් පසුදින වනතුරු Sleep කිරීම
            if daily_losses >= 2:
                print("අද දින Losses 2 සම්පූර්ණයි. හෙට වනතුරු විවේක ගනී...")
                await asyncio.sleep(300)
                continue

            # London සහ New York Session පරීක්ෂාව (07:00 - 21:00 UTC)
            in_session = 7 <= now.hour <= 21

            for symbol in SYMBOLS:
                price_data = await connection.get_symbol_price(symbol)
                bid, ask = price_data['bid'], price_data['ask']

                # සෑම තත්පරයකම Open Trades පරීක්ෂා කිරීම (Step 5)
                await manage_open_trades(connection, symbol, bid, ask, 0)

                # සෑම විනාඩි 5කට වරක් AI මගින් Candles සහ Setups පරීක්ෂා කිරීම (Step 1, 2, 3, 4)
                if in_session and now.minute % 5 == 0 and now.minute != last_ai_scan_minute:
                    w1 = await account.get_historical_candles(symbol, '1w', now, 40)
                    d1 = await account.get_historical_candles(symbol, '1d', now, 40)
                    h4 = await account.get_historical_candles(symbol, '4h', now, 40)
                    h1 = await account.get_historical_candles(symbol, '1h', now, 40)
                    m5 = await account.get_historical_candles(symbol, '5m', now, 30)

                    w1_levels = scan_snr_levels(w1, '1W')
                    d1_levels = scan_snr_levels(d1, '1D')

                    decision = await analyze_with_ai(symbol, w1_levels, d1_levels, h4, h1, m5)
                    print(f"[{symbol}] AI Decision: {decision}")

                    pip = get_pip_value(symbol)
                    lot = 0.01  # Demo පරීක්ෂාව සඳහා ආරක්ෂිත Lot Size එක

                    if decision.get("action") == "PLACE_APEX_LIMIT":
                        entry = float(decision["entry_price"])
                        if decision["direction"] == "BUY":
                            sl = entry - (30 * pip)
                            tp = entry + (200 * pip)
                            await connection.create_limit_buy_order(symbol, lot, entry, sl, tp)
                        elif decision["direction"] == "SELL":
                            sl = entry + (30 * pip)
                            tp = entry - (200 * pip)
                            await connection.create_limit_sell_order(symbol, lot, entry, sl, tp)

                    elif decision.get("action") == "EXECUTE_CONFIRMED_MARKET":
                        if decision["direction"] == "BUY":
                            sl = ask - (20 * pip)
                            tp = ask + (200 * pip)
                            await connection.create_market_buy_order(symbol, lot, sl, tp)
                        elif decision["direction"] == "SELL":
                            sl = bid + (20 * pip)
                            tp = bid - (200 * pip)
                            await connection.create_market_sell_order(symbol, lot, sl, tp)

            if now.minute % 5 == 0:
                last_ai_scan_minute = now.minute

        except Exception as e:
            print("Error:", e)

        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
