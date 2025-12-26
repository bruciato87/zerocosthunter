import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import logging
import os

# Set YFinance cache to /tmp for read-only filesystems (Vercel)
try:
    yf.set_tz_cache_location("/tmp/py-yfinance")
except Exception:
    pass

# Configure logging
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        pass

    def get_technical_summary(self, ticker: str) -> str:
        """
        Fetches price history and calculates RSI and SMA.
        Returns a human-readable summary string.
        """
        try:
            # 1. Fetch Data (3 months to ensure enough data for SMA50)
            ticker = ticker.upper().strip()
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo") # 6 months to be safe for SMA calculations

            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return "No Market Data Available"

            # 2. Calculate Indicators using 'ta' library
            # RSI 14
            rsi_ind = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi_ind.rsi()
            
            # SMA 50 & 200
            sma50_ind = SMAIndicator(close=df['Close'], window=50)
            df['SMA_50'] = sma50_ind.sma_indicator()
            
            sma200_ind = SMAIndicator(close=df['Close'], window=200)
            df['SMA_200'] = sma200_ind.sma_indicator()

            # Get latest values (last row)
            latest = df.iloc[-1]
            price = latest['Close']
            rsi = latest['RSI']
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            
            # Calculate 24h Change
            if len(df) > 1:
                prev_close = df.iloc[-2]['Close']
                change_pct = ((price - prev_close) / prev_close) * 100
            else:
                change_pct = 0.0

            # 3. Interpret Data
            trend = "Neutral"
            if price > sma_50 and price > sma_200:
                trend = "BULLISH (Above SMA 50/200)"
            elif price < sma_50 and price < sma_200:
                trend = "BEARISH (Below SMA 50/200)"
            elif price > sma_200:
                trend = "Recovering (Above SMA 200)"

            rsi_status = "Neutral"
            if rsi > 70:
                rsi_status = "OVERBOUGHT (>70)"
            elif rsi < 30:
                rsi_status = "OVERSOLD (<30)"

            # 4. Format Summary
            summary = (
                f"Price: ${price:.2f} ({change_pct:+.2f}%), "
                f"RSI: {rsi:.1f} ({rsi_status}), "
                f"Trend: {trend}"
            )
            return summary

        except Exception as e:
            logger.error(f"Error fetching technicals for {ticker}: {e}")
            return "Technical Analysis Failed"

if __name__ == "__main__":
    # Test
    md = MarketData()
    print(md.get_technical_summary("AAPL"))
    print(md.get_technical_summary("BTC-USD"))
