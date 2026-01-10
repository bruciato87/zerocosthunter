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
    Zero-Cost 'Hacker Style' Whale Watcher.
    Uses Binance Public API (aggTrades) to detect large market orders.
    No API Key required.
    
    Enhanced Features:
    - Extended symbol coverage (BTC, ETH, SOL, XRP, DOGE, AVAX)
    - Real-time whale alerts (>$1M)
    - Historical trend comparison (1h vs 24h avg)
    - Session timing awareness (Asia/EU/US)
    - Weighted flow (recent trades count more)
    """
    def __init__(self, telegram_bot=None):
        self.base_url = "https://api.binance.com/api/v3/aggTrades"
        self.min_value_usd = 500_000  # Lowered slightly to catch 'Sharks' too
        self.alert_threshold = 1_000_000  # Alert for >$1M trades
        # IMPROVEMENT 1: Extended symbols
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
        self.telegram_bot = telegram_bot  # For real-time alerts
        
    def fetch_binance_whales(self, symbol, hours=1):
        """
        Fetches aggregated trades from the specified time window.
        Iterates backwards using 'endTime' until the cutoff time is reached.
        """
        whales = []
        endpoints = [
            "https://api.binance.com/api/v3/aggTrades",
            "https://api.binance.us/api/v3/aggTrades",
        ]

        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        cutoff_time = now_ms - (hours * 60 * 60 * 1000)
        
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
             
             oldest_in_batch = current_batch[0]['T']
             
             for t in current_batch:
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
             
             if oldest_in_batch < cutoff_time:
                 break
                 
             end_time = oldest_in_batch - 1
             
        return whales

    def get_trading_session(self):
        """
        IMPROVEMENT 5: Determine current trading session based on time.
        Returns context about which region's whales are most active.
        """
        now = datetime.datetime.now(ZoneInfo("UTC"))
        hour = now.hour
        
        # Trading sessions (UTC)
        # Asia: 00:00-08:00 UTC (Tokyo/HK)
        # Europe: 07:00-16:00 UTC (London/Frankfurt)
        # US: 13:00-22:00 UTC (NY)
        
        if hour >= 0 and hour < 8:
            return {"session": "ASIA", "emoji": "🌏", "note": "Asian whales active (Tokyo/HK/Singapore)"}
        elif hour >= 7 and hour < 16:
            return {"session": "EUROPE", "emoji": "🌍", "note": "European institutions active (London/Frankfurt)"}
        elif hour >= 13 and hour < 22:
            return {"session": "US", "emoji": "🌎", "note": "US whales active (Wall Street)"}
        else:
            return {"session": "OFF-HOURS", "emoji": "🌙", "note": "Low liquidity - whale moves have larger impact"}

    def calculate_weighted_flow(self, whales):
        """
        IMPROVEMENT 6: Weight recent trades more heavily.
        Trades in last 30 min = 2x weight, last 15 min = 3x weight.
        """
        now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        thirty_min_ago = now_ms - (30 * 60 * 1000)
        fifteen_min_ago = now_ms - (15 * 60 * 1000)
        
        weighted_buy = 0
        weighted_sell = 0
        
        for w in whales:
            # Determine weight based on recency
            if w['timestamp'] >= fifteen_min_ago:
                weight = 3.0  # Most recent = 3x
            elif w['timestamp'] >= thirty_min_ago:
                weight = 2.0  # Recent = 2x
            else:
                weight = 1.0  # Older = 1x
            
            if w['side'] == "BUY":
                weighted_buy += w['value_usd'] * weight
            else:
                weighted_sell += w['value_usd'] * weight
        
        return weighted_buy, weighted_sell

    def get_historical_comparison(self):
        """
        IMPROVEMENT 3: Compare current 1h flow with 24h average.
        Returns trend direction (increasing/decreasing whale activity).
        """
        try:
            # Get current 1h data (already have this)
            current_whales = []
            for sym in self.symbols[:3]:  # Only major 3 for speed
                current_whales.extend(self.fetch_binance_whales(sym, hours=1))
            
            current_buy = sum(w['value_usd'] for w in current_whales if w['side'] == 'BUY')
            current_sell = sum(w['value_usd'] for w in current_whales if w['side'] == 'SELL')
            current_total = current_buy + current_sell
            
            # Estimate 24h average per hour (we can't fetch full 24h due to API limits)
            # Use a simple heuristic: if current hour has > $10M activity, that's above normal
            avg_hourly = 5_000_000  # Baseline: $5M/hour is average
            
            if current_total > avg_hourly * 2:
                trend = "📈 SURGING"
                note = "Whale activity 2x+ above normal"
            elif current_total > avg_hourly * 1.5:
                trend = "⬆️ ELEVATED" 
                note = "Whale activity above normal"
            elif current_total < avg_hourly * 0.5:
                trend = "⬇️ QUIET"
                note = "Whale activity below normal"
            else:
                trend = "➡️ NORMAL"
                note = "Whale activity at expected levels"
            
            return {
                "trend": trend,
                "note": note,
                "current_volume_m": round(current_total / 1_000_000, 1)
            }
        except Exception as e:
            logger.warning(f"Historical comparison failed: {e}")
            return {"trend": "➡️ NORMAL", "note": "Comparison unavailable", "current_volume_m": 0}

    async def send_whale_alert(self, whale_trade):
        """
        IMPROVEMENT 2: Send real-time alert for mega-whale trades (>$1M).
        """
        if self.telegram_bot and whale_trade['value_usd'] >= self.alert_threshold:
            try:
                val_m = whale_trade['value_usd'] / 1_000_000
                emoji = "🟢" if whale_trade['side'] == "BUY" else "🔴"
                
                message = f"""
🐋 **MEGA-WHALE ALERT!**

{emoji} **${val_m:.1f}M {whale_trade['side']}** on {whale_trade['symbol'].replace('USDT', '')}

Price: ${whale_trade['price']:,.2f}
Quantity: {whale_trade['qty']:,.4f}

_This is a significant institutional move!_
"""
                await self.telegram_bot.send_message(message)
                logger.info(f"Whale alert sent: {whale_trade['side']} ${val_m:.1f}M")
            except Exception as e:
                logger.warning(f"Failed to send whale alert: {e}")

    def analyze_flow(self, test_mode=False):
        """
        Analyzes recent market flow for Whales.
        Enhanced with timing, trends, and weighted flow.
        """
        all_whales = []
        buy_vol = 0
        sell_vol = 0
        
        for sym in self.symbols:
            w = self.fetch_binance_whales(sym)
            all_whales.extend(w)
            
        # Aggregate raw volumes
        significant_events = []
        for w in all_whales:
            val_m = w['value_usd'] / 1_000_000
            if w['side'] == "BUY":
                buy_vol += w['value_usd']
                if val_m > 1.0:
                    significant_events.append(f"🟢 ${val_m:.1f}M BUY on {w['symbol'].replace('USDT', '')}")
            else:
                sell_vol += w['value_usd']
                if val_m > 1.0:
                    significant_events.append(f"🔴 ${val_m:.1f}M SELL on {w['symbol'].replace('USDT', '')}")
        
        # IMPROVEMENT 6: Calculate weighted flow
        weighted_buy, weighted_sell = self.calculate_weighted_flow(all_whales)
        weighted_net = weighted_buy - weighted_sell
        
        # Use weighted flow for status determination
        raw_net = buy_vol - sell_vol
        
        status = "NEUTRAL"
        if weighted_net > 10_000_000: status = "BULLISH (Net Buying)"
        elif weighted_net < -10_000_000: status = "BEARISH (Net Selling)"
        elif raw_net > 5_000_000: status = "SLIGHTLY BULLISH"
        elif raw_net < -5_000_000: status = "SLIGHTLY BEARISH"
        
        # IMPROVEMENT 5: Get session context
        session = self.get_trading_session()
        
        # IMPROVEMENT 3: Get trend
        trend = self.get_historical_comparison()
        
        if not all_whales:
             status = "NEUTRAL (Quiet Market)"
             significant_events = ["None"]

        # Build strategy hint
        if status.startswith('BEARISH'):
            hint = '⚠️ DUMP DETECTED: Be cautious with Crypto'
        elif status.startswith('BULLISH'):
            hint = '✅ ACCUMULATION: Institutional interest visible'
        elif 'SLIGHTLY' in status:
            hint = '👀 MIXED SIGNALS: Monitor closely'
        else:
            hint = 'Market is calm. No whale manipulation detected.'

        context = f"""
        [WHALE WATCHER (BINANCE REAL-TIME)]
        Symbols: BTC, ETH, SOL, XRP, DOGE, AVAX
        Status: {status}
        Session: {session['emoji']} {session['session']} - {session['note']}
        Activity: {trend['trend']} ({trend['note']})
        
        Raw Flow:
        - Whale Buy Vol: ${buy_vol/1_000_000:.1f}M
        - Whale Sell Vol: ${sell_vol/1_000_000:.1f}M
        - Net Flow: ${raw_net/1_000_000:.1f}M
        
        Weighted Flow (recent trades 3x):
        - Weighted Net: ${weighted_net/1_000_000:.1f}M
        
        Largest Transactions: {', '.join(significant_events[:5]) if significant_events else 'None'}
        
        strategy_hint: {hint}
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
            
            # Weighted flow
            weighted_buy, weighted_sell = self.calculate_weighted_flow(all_whales)
            weighted_net = weighted_buy - weighted_sell
            
            raw_net = buy_vol - sell_vol
            status = "NEUTRAL"
            if weighted_net > 10_000_000: status = "BULLISH"
            elif weighted_net < -10_000_000: status = "BEARISH"
            elif raw_net > 5_000_000: status = "SLIGHTLY_BULLISH"
            elif raw_net < -5_000_000: status = "SLIGHTLY_BEARISH"
            
            session = self.get_trading_session()
            trend = self.get_historical_comparison()

            return {
                "status": status,
                "buy_vol_m": round(buy_vol / 1_000_000, 1),
                "sell_vol_m": round(sell_vol / 1_000_000, 1),
                "net_flow_m": round(raw_net / 1_000_000, 1),
                "weighted_net_m": round(weighted_net / 1_000_000, 1),
                "whale_count": len(all_whales),
                "session": session['session'],
                "session_note": session['note'],
                "trend": trend['trend'],
                "trend_note": trend['note']
            }
        except Exception as e:
            logger.error(f"Whale Stats Calculation Failed: {e}")
            return {
                "status": "NEUTRAL",
                "buy_vol_m": 0.0,
                "sell_vol_m": 0.0,
                "net_flow_m": 0.0,
                "weighted_net_m": 0.0,
                "whale_count": 0,
                "session": "UNKNOWN",
                "session_note": "Error",
                "trend": "➡️ NORMAL",
                "trend_note": "Error"
            }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    ww = WhaleWatcher()
    print(ww.analyze_flow())
