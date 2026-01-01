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
        Fetches the last 1000 aggregated trades for a symbol.
        Implements fallback logic for Geo-Blocked regions (US).
        Returns list of 'Whale' trades (value > min_value_usd).
        """
        whales = []
        endpoints = [
            "https://api.binance.com/api/v3/aggTrades",      # Global
            "https://api.binance.us/api/v3/aggTrades",       # US Fallback
             # Coinbase typically uses different format, so we stick to Binance family for compat 
             # or implement distinct logic. Let's start with Binance US.
        ]

        success = False
        for base_url in endpoints:
            try:
                # Adjust symbol for Binance US if needed (usually same for major pairs)
                # Binance US volume is lower, but still indicative of major moves.
                params = {"symbol": symbol, "limit": 1000}
                
                # Fast timeout to quick-fail to fallback
                resp = requests.get(base_url, params=params, timeout=3)
                
                if resp.status_code == 200:
                    trades = resp.json()
                    for t in trades:
                        # Binance aggTrade format is consistent across .com and .us
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
                    success = True
                    break # Success, stop trying endpoints
                elif resp.status_code == 451:
                    logger.warning(f"Binance Global Geo-Blocked (451). Trying US endpoint...")
                    continue # Try next endpoint
                else:
                    logger.warning(f"Binance API Error {resp.status_code} from {base_url}: {resp.text}")
            
            except Exception as e:
                logger.error(f"Binance Fetch Error ({base_url}): {e}")
        
        if not success:
             logger.error(f"All Binance endpoints failed for {symbol}.")
            
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
        
        # If no data found (e.g. quiet market or offset issue), handle gracefully
        if not all_whales:
             return "[WHALE CONTEXT] No significant Whale activity detected in last 1000 trades."

        context = f"""
        [WHALE WATCHER (BINANCE REAL-TIME)]
        Status: {status}
        Whale Buy Vol: ${buy_vol/1_000_000:.1f}M
        Whale Sell Vol: ${sell_vol/1_000_000:.1f}M
        Net Flow: ${net_flow/1_000_000:.1f}M
        Largest Transactions: {', '.join(significant_events[:5])}
        strategy_hint: {'⚠️ DUMP DETECTED: Be cautious with Crypto' if status.startswith('BEARISH') else '✅ ACCUMULATION: Institutional interest visible' if status.startswith('BULLISH') else 'Flow is balanced'}
        """
        
        return context.strip()

    def get_dashboard_stats(self):
        """
        Returns structured stats for the Dashboard UI.
        """
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

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    ww = WhaleWatcher()
    print(ww.analyze_flow())
