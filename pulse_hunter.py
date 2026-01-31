import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from market_data import MarketData
from db_handler import DBHandler
from ticker_resolver import resolve_ticker

logger = logging.getLogger("PulseHunter")

class PulseHunter:
    """
    Market Pulse Engine: Quantitative Radar for Early Detection.
    Scans for volume anomalies, RSI divergences, and breakout patterns.
    """
    
    # Symbols to track if not in portfolio (Market Leaders)
    GLOBAL_WATCHLIST = [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "RENDER-USD", # Top Crypto
        "NVDA", "AAPL", "MSFT", "META", "GOOGL", "AMZN", "TSLA", # Magnificient 7
        "ASML", "AMD", "TSM" # Semis
    ]

    def __init__(self, market_instance: Optional[MarketData] = None):
        self.market = market_instance or MarketData()
        self.db = DBHandler()

    def _get_watchlist(self) -> List[str]:
        """Combine portfolio assets with global watchlist."""
        try:
            portfolio = self.db.get_portfolio()
            portfolio_tickers = [resolve_ticker(p['ticker']) for p in portfolio]
            
            # Combine and de-duplicate
            combined = list(set(portfolio_tickers + self.GLOBAL_WATCHLIST))
            return combined
        except Exception as e:
            logger.error(f"Failed to fetch portfolio for watchlist: {e}")
            return self.GLOBAL_WATCHLIST

    def detect_anomalies(self, ticker: str) -> Optional[Dict]:
        """
        Analyze a single ticker for quantitative anomalies.
        Returns a dict with findings or None.
        """
        try:
            # Fetch 30 days of data for baseline
            df = self.market.get_historical_data(ticker, days=30)
            if df is None or len(df) < 20:
                return None

            # 1. VOLUME ANOMALY (Predictive of Institutional Interest)
            # Current Vol vs 20-day Average
            avg_vol = df['Volume'].iloc[:-1].tail(20).mean()
            curr_vol = df['Volume'].iloc[-1]
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

            # 2. RSI DIVERGENCE (Leading indicator of reversals)
            # We use tail(14) for current RSI
            from ta.momentum import RSIIndicator
            rsi_series = RSIIndicator(close=df['Close'], window=14).rsi()
            curr_rsi = rsi_series.iloc[-1]
            prev_rsi = rsi_series.iloc[-5] # Simple lookback for slope
            
            # 3. BREAKOUT POTENTIAL
            # Close vs 20-day High
            max_20d = df['High'].iloc[:-1].tail(20).max()
            price = df['Close'].iloc[-1]
            dist_to_high = (max_20d - price) / price * 100

            # --- DETECTION LOGIC ---
            findings = []
            confidence_boost = 0
            
            # A. Volume Spike (The "Insider Move")
            if vol_ratio > 2.5:
                findings.append(f"📊 Volume anomalo: {vol_ratio:.1f}x la media (Accumulo sospetto)")
                confidence_boost += 0.2
            elif vol_ratio > 1.8:
                findings.append(f"🔍 Volume in crescita: {vol_ratio:.1f}x (Interesse istituzionale)")
                confidence_boost += 0.1

            # B. RSI Extreme (Mean Reversion)
            if curr_rsi < 30:
                findings.append(f"📉 Oversold: RSI {curr_rsi:.1f} (Potenziale rimbalzo tecnico)")
                confidence_boost += 0.15
            elif curr_rsi > 75:
                findings.append(f"📈 Overbought: RSI {curr_rsi:.1f} (Rischio correzione/euforia)")
                confidence_boost -= 0.1 # This is a warning, not a boost for BUY

            # C. Breakout Threshold
            if abs(dist_to_high) < 1.5:
                findings.append(f"⚡ Prossimità Breakout: Prezzo vicino ai massimi a 20 giorni")
                confidence_boost += 0.1
            
            if not findings:
                return None

            return {
                "ticker": ticker,
                "findings": findings,
                "confidence_modifier": confidence_boost,
                "metrics": {
                    "vol_ratio": round(vol_ratio, 2),
                    "rsi": round(curr_rsi, 2),
                    "dist_to_high": round(dist_to_high, 2)
                }
            }

        except Exception as e:
            logger.warning(f"Pulse analysis failed for {ticker}: {e}")
            return None

    def scan(self) -> List[Dict]:
        """Scan the entire watchlist and return actionable pulse triggers."""
        watchlist = self._get_watchlist()
        logger.info(f"PulseHunter: Scanning {len(watchlist)} tickers...")
        
        results = []
        for ticker in watchlist:
            anomaly = self.detect_anomalies(ticker)
            if anomaly:
                results.append(anomaly)
        
        logger.info(f"⚡ Market Pulse: Found {len(results)} technical anomalies.")
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pulse = PulseHunter()
    print(pulse.scan())
