import os
import requests
import logging
import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("WhaleWatcher")

class WhaleWatcher:
    """
    Zero-Cost 'Hacker Style' Whale Watcher.
    Uses Binance Public API (aggTrades) to detect large market orders.
    No API Key required.
    """
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/aggTrades"
        self.min_value_usd = 500_000  # Lowered slightly to catch 'Sharks' too, not just Whales
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
    def fetch_binance_whales(self, symbol):
        """
        Fetches aggregated trades from the LAST 1 HOUR.
        Iterates backwards using 'endTime' until the cutoff time is reached.
        """
        whales = []
        endpoints = [
            "https://api.binance.com/api/v3/aggTrades",
            "https://api.binance.us/api/v3/aggTrades",
        ]

        # 1 Hour Window
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        cutoff_time = now_ms - (60 * 60 * 1000) # 1 Hour ago
        
        end_time = None  # Starts at "Now"
        
        # Safety Limit: Max 30 requests (~30k trades max) to prevent timeout
        for _ in range(30):
             batch_success = False
             current_batch = []
             
             for base_url in endpoints:
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
                         break
                     elif resp.status_code == 451:
                         continue
                 except Exception:
                     continue
            
             if not batch_success or not current_batch:
                 break
             
             # Check if we crossed the cutoff line
             # Batch is oldest -> newest
             oldest_in_batch = current_batch[0]['T']
             
             for t in current_batch:
                 # Only keep trades within the 1-hour window
                 if t['T'] < cutoff_time:
                     continue

                 price = float(t['p'])
                 qty = float(t['q'])
                 value_usd = price * qty
                 
                 if value_usd >= self.min_value_usd:
                     side = "SELL" if t['m'] else "BUY"
                     whales.append({
                         "symbol": symbol,
                         "price": price,
                         "qty": qty,
                         "value_usd": value_usd,
                         "side": side,
                         "timestamp": t['T']
                     })
             
             # Prepare next loop
             # If the oldest trade in this batch is ALREADY older than cutoff, stop.
             if oldest_in_batch < cutoff_time:
                 break
                 
             # Otherwise, go back further
             end_time = oldest_in_batch - 1
             
        return whales

    def analyze_flow(self, test_mode=False):
        """
        Analyzes recent market flow for Whales.
        test_mode: Not used anymore (Real Data is Free!), kept for signature compatibility.
        """
        all_whales = []
        buy_vol = 0
        sell_vol = 0
        
        for sym in self.symbols:
            w = self.fetch_binance_whales(sym)
            all_whales.extend(w)
            
        # Aggregate
        significant_events = []
        for w in all_whales:
            val_m = w['value_usd'] / 1_000_000
            if w['side'] == "BUY":
                buy_vol += w['value_usd']
                if val_m > 1.0: # Only log super-whales (>1M) individually
                    significant_events.append(f"🟢 ${val_m:.1f}M BUY on {w['symbol']}")
            else:
                sell_vol += w['value_usd']
                if val_m > 1.0:
                    significant_events.append(f"🔴 ${val_m:.1f}M SELL on {w['symbol']}")
        
        # Determine Context
        net_flow = buy_vol - sell_vol
        
        status = "NEUTRAL"
        if net_flow > 5_000_000: status = "BULLISH (Net Buying)"
        elif net_flow < -5_000_000: status = "BEARISH (Net Selling)"
        
        # If no data found, be explicit but keep format
        if not all_whales:
             status = "NEUTRAL (Quiet Market)"
             significant_events = ["None"]

        context = f"""
        [WHALE WATCHER (BINANCE REAL-TIME)]
        Status: {status}
        Whale Buy Vol: ${buy_vol/1_000_000:.1f}M
        Whale Sell Vol: ${sell_vol/1_000_000:.1f}M
        Net Flow: ${net_flow/1_000_000:.1f}M
        Largest Transactions: {', '.join(significant_events[:5])}
        strategy_hint: {'⚠️ DUMP DETECTED: Be cautious with Crypto' if status.startswith('BEARISH') else '✅ ACCUMULATION: Institutional interest visible' if status.startswith('BULLISH') else 'Market is calm. No whale manipulation detected.'}
        """
        
        return context.strip()

    def get_dashboard_stats(self):
        """
        Returns structured stats for the Dashboard UI.
        Fail-safe: Returns neutral stats on error instead of raising.
        """
        try:
            all_whales = []
            buy_vol = 0
            sell_vol = 0
            
            for sym in self.symbols:
                w = self.fetch_binance_whales(sym)
                all_whales.extend(w)
                
            for w in all_whales:
                if w['side'] == "BUY":
                    buy_vol += w['value_usd']
                else:
                    sell_vol += w['value_usd']
            
            net_flow = buy_vol - sell_vol
            status = "NEUTRAL"
            if net_flow > 5_000_000: status = "BULLISH"
            elif net_flow < -5_000_000: status = "BEARISH"

            return {
                "status": status,
                "buy_vol_m": round(buy_vol / 1_000_000, 1),
                "sell_vol_m": round(sell_vol / 1_000_000, 1),
                "net_flow_m": round(net_flow / 1_000_000, 1),
                "whale_count": len(all_whales)
            }
        except Exception as e:
            logger.error(f"Whale Stats Calculation Failed: {e}")
            return {
                "status": "NEUTRAL",
                "buy_vol_m": 0.0,
                "sell_vol_m": 0.0,
                "net_flow_m": 0.0,
                "whale_count": 0
            }

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    ww = WhaleWatcher()
    print(ww.analyze_flow())
