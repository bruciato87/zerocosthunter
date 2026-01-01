import os
import requests
import logging
import random
import datetime

logger = logging.getLogger("WhaleWatcher")

class WhaleWatcher:
    def __init__(self):
        self.api_key = os.environ.get("WHALE_ALERT_API_KEY")
        self.base_url = "https://api.whale-alert.io/v1"
        self.min_value_usd = 10_000_000 # Minimum $10M to be considered a Whale

    def fetch_latest_whales(self, test_mode=False):
        """
        Fetches large transactions from the last 24h.
        Args:
            test_mode (bool): If True, returns realistic mock data for UI demo.
        """
        transfers = []
        
        # 1. MOCK MODE (Explicit Request Only)
        if test_mode:
            return self._generate_mock_data()

        # 2. REAL MODE (API Key Present)
        if self.api_key and self.api_key != "your_whale_key_here":
            try:
                # Fetch last hour for responsiveness
                now_ts = int(datetime.datetime.now().timestamp())
                start_ts = now_ts - 3600 
                url = f"{self.base_url}/transactions?api_key={self.api_key}&min_value={self.min_value_usd}&start={start_ts}"
                
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    transfers = data.get('transactions', [])
                else:
                    logger.warning(f"Whale API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Whale Fetch Failed: {e}")
            
        return transfers

    def analyze_flow(self, test_mode=False):
        """
        Analyzes transfers to detect Net Flow Pressure.
        Returns: (status, context_string)
        """
        transfers = self.fetch_latest_whales(test_mode=test_mode)
        
        if not transfers:
            return "[WHALE CONTEXT]\nStatus: NO DATA (No API Key)\nAction: Using Technical Volume Analysis instead."
        
        buy_pressure_usd = 0
        sell_pressure_usd = 0
        significant_events = []
        
        # Known Stablecoins
        stables = ['usdt', 'usdc', 'busd', 'dai']
        
        for tx in transfers:
            symbol = tx.get('symbol', '').lower()
            amount_usd = tx.get('amount_usd', 0)
            from_wallet = tx.get('from', {}).get('owner_type', 'unknown')
            to_wallet = tx.get('to', {}).get('owner_type', 'unknown')
            
            # Heuristic: Exchange Inflow
            if to_wallet == 'exchange':
                if symbol in stables:
                    # Stablecoin -> Exchange = Buying Power
                    buy_pressure_usd += amount_usd
                    if amount_usd > 50_000_000:
                        significant_events.append(f"🟢 BUY LOAD: ${amount_usd/1_000_000:.1f}M {symbol.upper()} to Exchange")
                elif symbol in ['btc', 'eth']:
                    # Crypto -> Exchange = Potential Dump
                    sell_pressure_usd += amount_usd
                    if amount_usd > 50_000_000:
                        significant_events.append(f"🔴 DUMP RISK: ${amount_usd/1_000_000:.1f}M {symbol.upper()} to Exchange")

        # Determine Context
        net_flow = buy_pressure_usd - sell_pressure_usd
        
        status = "NEUTRAL"
        if net_flow > 100_000_000: status = "BULLISH (Inflow)"
        elif net_flow < -100_000_000: status = "BEARISH (Outflow)"
        
        # Format Context for AI
        context = f"""
        [WHALE WATCHER CONTEXT]
        Net Flow Status: {status}
        Buy Pressure (Stables IN): ${buy_pressure_usd/1_000_000:.1f}M
        Sell Pressure (Coins IN): ${sell_pressure_usd/1_000_000:.1f}M
        Major Events: {', '.join(significant_events) if significant_events else 'None'}
        strategy_hint: {'⚠️ CAUTION: High Dump Risk Detected' if status.startswith('BEARISH') else '✅ SUPPORT: Institutional Buying Detected' if status.startswith('BULLISH') else 'Neutral Flow'}
        """
        
        return context.strip()

    def _generate_mock_data(self):
        """Generates realistic whale transactions for demo."""
        mock_txs = []
        # Simulate 1 big Bitcoin move (Sell Pressure)
        if random.random() > 0.5:
            mock_txs.append({
                "symbol": "btc",
                "amount_usd": random.randint(50, 200) * 1_000_000,
                "from": {"owner_type": "unknown"},
                "to": {"owner_type": "exchange"}
            })
        
        # Simulate 2 big USDT moves (Buy Pressure)
        if random.random() > 0.3:
            mock_txs.append({
                "symbol": "usdt",
                "amount_usd": random.randint(50, 300) * 1_000_000,
                "from": {"owner_type": "unknown"},
                "to": {"owner_type": "exchange"}
            })
            
        return mock_txs

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    ww = WhaleWatcher()
    print(ww.analyze_flow())
