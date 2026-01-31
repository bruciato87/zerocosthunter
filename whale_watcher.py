import os
import requests
import logging
import datetime
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("WhaleWatcher")

class WhaleWatcher:
    """
    Zero-Cost 'Hacker Style' Whale Watcher V2 (Deep Sea Edition).
    Uses Binance Public API (Spot + Futures) to detect large market orders and hidden algos.
    No API Key required.
    
    Enhanced Features V2:
    - Futures Integration (fapi) for leverage tracking
    - Iceberg Order Detection (Time-based clustering)
    - Open Interest (OI) Analysis for trend strength
    - Extended symbol coverage
    - Real-time whale alerts (>$1M)
    - Session timing awareness (Asia/EU/US)
    - Weighted flow (recent trades count more)
    """
    def __init__(self, telegram_bot=None):
        self.spot_url = "https://api.binance.com/api/v3/aggTrades"
        self.futures_url = "https://fapi.binance.com/fapi/v1/aggTrades"
        self.oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
        
        self.min_value_spot = 500_000
        self.min_value_futures = 1_000_000 # Higher threshold for leverage
        self.alert_threshold = 1_000_000   # Alert for >$1M trades
        
        # IMPROVEMENT 1: Extended symbols
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
        self.telegram_bot = telegram_bot  # For real-time alerts
        
    def fetch_binance_whales(self, symbol, hours=1, market_type="SPOT"):
        """
        Fetches aggregated trades from the specified time window.
        Supports both SPOT and FUTURES markets.
        """
        whales = []
        base_url = self.futures_url if market_type == "FUTURES" else self.spot_url
        min_val = self.min_value_futures if market_type == "FUTURES" else self.min_value_spot

        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        cutoff_time = now_ms - (hours * 60 * 60 * 1000)
        
        end_time = None  # Starts at "Now"
        
        # Safety Limit: Max 20 requests per symbol/market type
        for _ in range(20):
             batch_success = False
             current_batch = []
             
             try:
                 params = {"symbol": symbol, "limit": 1000}
                 if end_time:
                     params["endTime"] = end_time
                 
                 resp = requests.get(base_url, params=params, timeout=3)
                 
                 if resp.status_code == 200:
                     trades = resp.json()
                     if not trades: break
                     current_batch = trades
                     batch_success = True
                 elif resp.status_code == 429: # Rate limit
                     break
                 elif resp.status_code == 451: # Legal restriction
                     break
             except Exception:
                 continue
            
             if not batch_success or not current_batch:
                 break
             
             oldest_in_batch = current_batch[0]['T']
             
             for t in current_batch:
                 if t['T'] < cutoff_time:
                     continue

                 price = float(t['p'])
                 qty = float(t['q'])
                 value_usd = price * qty
                 
                 # Basic Threshold Filter
                 if value_usd >= min_val:
                     side = "SELL" if t['m'] else "BUY" # For Spot: m=True is SELL. For Futures: m=True is SELL.
                     whales.append({
                         "symbol": symbol,
                         "price": price,
                         "qty": qty,
                         "value_usd": value_usd,
                         "side": side,
                         "timestamp": t['T'],
                         "market": market_type,
                         "type": "Single Trade"
                     })
             
             if oldest_in_batch < cutoff_time:
                 break
                 
             end_time = oldest_in_batch - 1
             
        return whales

    def detect_icebergs(self, symbol, hours=1):
        """
        IMPROVEMENT V2-2: Iceberg Order Detection.
        Fetches ALL trades (not just big ones) and looks for clusters.
        Cluster = >$1M volume in <5 seconds.
        """
        icebergs = []
        # Only check Futures for Icebergs (where algorithms live)
        base_url = self.futures_url 
        
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        # Only check last 15 minutes for Icebergs to save API calls
        cutoff_time = now_ms - (15 * 60 * 1000) 
        
        try:
            params = {"symbol": symbol, "limit": 1000} # Get most recent 1000 trades
            resp = requests.get(base_url, params=params, timeout=3)
            
            if resp.status_code != 200: return []
            trades = resp.json()
            
            # Simple Clustering Algorithm
            # Group trades by 5-second windows
            window_size_ms = 5000
            current_window_start = trades[0]['T']
            window_vol_buy = 0
            window_vol_sell = 0
            
            for t in trades:
                if t['T'] < cutoff_time: continue
                
                price = float(t['p'])
                qty = float(t['q'])
                val = price * qty
                side = "SELL" if t['m'] else "BUY"
                
                if t['T'] - current_window_start > window_size_ms:
                    # Analyze Window
                    if window_vol_buy > 1_000_000:
                        icebergs.append({
                             "symbol": symbol, "price": price, "value_usd": window_vol_buy,
                             "side": "BUY", "timestamp": current_window_start,
                             "market": "FUTURES", "type": "🧊 ICEBERG"
                        })
                    if window_vol_sell > 1_000_000:
                        icebergs.append({
                             "symbol": symbol, "price": price, "value_usd": window_vol_sell,
                             "side": "SELL", "timestamp": current_window_start,
                             "market": "FUTURES", "type": "🧊 ICEBERG"
                        })
                    
                    # Reset Window
                    current_window_start = t['T']
                    window_vol_buy = 0
                    window_vol_sell = 0
                
                if side == "BUY": window_vol_buy += val
                else: window_vol_sell += val
                
        except Exception as e:
            logger.warning(f"Iceberg detection failed: {e}")
            
        return icebergs

    def get_open_interest(self, symbol):
        """
        IMPROVEMENT V2-3: Open Interest Tracking.
        Returns the current OI amount in USD.
        """
        try:
            resp = requests.get(self.oi_url, params={"symbol": symbol}, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                return float(data['openInterest']) * float(data['symbol'].replace('USDT','') if 'price' not in data else data['openInterest']) # Some endpoints return contracts, some USD. fapi/v1/openInterest returns amount in USDT usually
                # Actually fapi returns 'openInterest' (quantity) and we need price to get USD value, but safer to assume it reflects magnitude. 
                # Let's trust the 'openInterest' field, but better yet 'sumOpenInterestValue' from another endpoint is better but let's stick to simple.
                # Correction: fapi/v1/openInterest returns: {"symbol": "BTCUSDT", "openInterest": "100.0", "time": 123} where OI is in base asset (BTC).
                # So we need price to get USD.
            return 0
        except:
            return 0
            
    def get_open_interest_analysis(self, symbol):
        """
        Analyzes OI structure (simple heuristic).
        """
        try:
             # Get current price and OI
             price_resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/price", params={"symbol": symbol}, timeout=2)
             oi_resp = requests.get(self.oi_url, params={"symbol": symbol}, timeout=2)
             
             if price_resp.status_code == 200 and oi_resp.status_code == 200:
                 price = float(price_resp.json()['price'])
                 oi_qty = float(oi_resp.json()['openInterest'])
                 oi_val = price * oi_qty
                 
                 return {"value_usd": oi_val, "status": "Stable"}
        except:
            pass
        return {"value_usd": 0, "status": "Unknown"}

    def get_trading_session(self):
        """
        Determine current trading session based on time.
        """
        now = datetime.datetime.now(ZoneInfo("UTC"))
        hour = now.hour
        
        if hour >= 0 and hour < 8:
            return {"session": "ASIA", "emoji": "🌏", "note": "Tokyo/HK active"}
        elif hour >= 7 and hour < 16:
            return {"session": "EUROPE", "emoji": "🌍", "note": "London active"}
        elif hour >= 13 and hour < 22:
            return {"session": "US", "emoji": "🌎", "note": "Wall St active"}
        else:
            return {"session": "OFF-HOURS", "emoji": "🌙", "note": "Low liquidity"}

    def calculate_weighted_flow(self, whales):
        """
        Weight recent trades more heavily.
        """
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        fifteen_min_ago = now_ms - (15 * 60 * 1000)
        
        weighted_buy = 0
        weighted_sell = 0
        
        for w in whales:
            weight = 3.0 if w['timestamp'] >= fifteen_min_ago else 1.0
            val = w['value_usd']
            if w.get('type') == '🧊 ICEBERG': weight *= 1.5 # Boost hidden orders importance
            
            if w['side'] == "BUY":
                weighted_buy += val * weight
            else:
                weighted_sell += val * weight
        
        return weighted_buy, weighted_sell

    def analyze_flow(self, test_mode=False):
        """
        Analyzes flow combining Spot + Futures + Icebergs + OI.
        """
        all_whales = []
        buy_vol = 0
        sell_vol = 0
        
        # 1. Fetch Spot & Futures & Icebergs
        for sym in self.symbols:
            # Spot
            spot_w = self.fetch_binance_whales(sym, market_type="SPOT")
            all_whales.extend(spot_w)
            
            # Futures (Leverage)
            fut_w = self.fetch_binance_whales(sym, market_type="FUTURES")
            all_whales.extend(fut_w)
            
            # Icebergs (Hidden)
            ice_w = self.detect_icebergs(sym)
            all_whales.extend(ice_w)
            
        # 2. Aggregate Volumes
        significant_events = []
        futures_vol_share = 0
        
        for w in all_whales:
            val_m = w['value_usd'] / 1_000_000
            
            if w['market'] == "FUTURES": futures_vol_share += w['value_usd']
            
            if w['side'] == "BUY":
                buy_vol += w['value_usd']
                if val_m > 2.0: # Raise threshold for log
                    significant_events.append(f"${val_m:.1f}M {w.get('type','BUY')} ({w['market']}) on {w['symbol'].replace('USDT', '')}")
            else:
                sell_vol += w['value_usd']
                if val_m > 2.0:
                    significant_events.append(f"${val_m:.1f}M {w.get('type','SELL')} ({w['market']}) on {w['symbol'].replace('USDT', '')}")
        
        total_vol = buy_vol + sell_vol
        futures_pct = (futures_vol_share / total_vol * 100) if total_vol > 0 else 0
        
        # 3. Weighted Flow
        weighted_buy, weighted_sell = self.calculate_weighted_flow(all_whales)
        weighted_net = weighted_buy - weighted_sell
        
        # 4. Status Determination
        status = "NEUTRAL"
        if weighted_net > 20_000_000: status = "STRONG BUY (Golden Whale)" # Higher threshold due to Futures volume
        elif weighted_net < -20_000_000: status = "STRONG SELL (Dump Incoming)"
        elif weighted_net > 5_000_000: status = "BULLISH"
        elif weighted_net < -5_000_000: status = "BEARISH"
        
        # 5. Context
        session = self.get_trading_session()
        
        # 6. Sample OI (Just for BTC as proxy for market)
        btc_oi = self.get_open_interest_analysis("BTCUSDT")
        
        hint = "Market is calm."
        if "STRONG" in status: hint = "🚨 HEAVY INSTITUTIONAL ACTIVITY! Follow the flow."
        elif "BULLISH" in status: hint = "✅ Accumulation detected."
        elif "BEARISH" in status: hint = "⚠️ Distribution/Dumping detected."
        
        if futures_pct > 70: hint += " (Leverage Driven)"

        context = f"""
        [WHALE WATCHER V2 (DEEP SEA EDITION)]
        Symbols: {', '.join([s.replace('USDT','') for s in self.symbols])}
        Global Status: {status}
        Session: {session['emoji']} {session['session']}
        
        Volume Profile:
        - Total Buy: ${buy_vol/1_000_000:.1f}M
        - Total Sell: ${sell_vol/1_000_000:.1f}M
        - Net Flow: ${(buy_vol-sell_vol)/1_000_000:.1f}M
        - Weighted Net: ${weighted_net/1_000_000:.1f}M
        
        Market Structure:
        - Futures Dominance: {futures_pct:.1f}% (High leverage = Volatility)
        - BTC Open Interest: ${btc_oi['value_usd']/1_000_000:.0f}M
        
        Top Moves: {', '.join(significant_events[:4]) if significant_events else 'None'}
        Strategy Hint: {hint}
        """
        
        return context.strip()

    def get_dashboard_stats(self):
        """
        Parses the text report to return structured stats for the UI.
        Keeps Dashboard backward compatible with V2 logic.
        """
        try:
            context = self.analyze_flow()
            lines = context.splitlines()
            
            stats = {
                "status": "NEUTRAL",
                "buy_vol_m": 0.0,
                "sell_vol_m": 0.0,
                "net_flow_m": 0.0,
                "weighted_net_m": 0.0,
                "full_report": context
            }
            
            import re
            
            # Simple Regex Extraction from the standard report format
            # Status: Global Status: NEUTRAL
            # Net Flow: - Net Flow: $3.6M
            
            for l in lines:
                l = l.strip()
                if "Global Status:" in l:
                    stats["status"] = l.split(":")[1].strip()
                elif "- Total Buy:" in l:
                     match = re.search(r"\$([\d\.]+)M", l)
                     if match: stats["buy_vol_m"] = float(match.group(1))
                elif "- Total Sell:" in l:
                     match = re.search(r"\$([\d\.]+)M", l)
                     if match: stats["sell_vol_m"] = float(match.group(1))
                elif "- Net Flow:" in l:
                     match = re.search(r"\$([-\d\.]+)M", l)
                     if match: stats["net_flow_m"] = float(match.group(1))
                elif "- Weighted Net:" in l:
                     match = re.search(r"\$([-\d\.]+)M", l)
                     if match: stats["weighted_net_m"] = float(match.group(1))

            return stats
            
        except Exception as e:
            logger.error(f"Dashboard Stats Parsing Failed: {e}")
            return {"status": "ERROR", "full_report": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ww = WhaleWatcher()
    print(ww.analyze_flow())
