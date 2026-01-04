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
import requests # Added for CoinGecko

# Configure logging
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        self.COINGECKO_MAP = {
            "BTC-USD": "bitcoin",
            "ETH-USD": "ethereum",
            "SOL-USD": "solana",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "SOL": "solana",
            "XRP": "ripple", "XRP-USD": "ripple",
            "ADA": "cardano", "ADA-USD": "cardano",
            "DOGE": "dogecoin", "DOGE-USD": "dogecoin",
            "DOT": "polkadot", "DOT-USD": "polkadot",
            "LINK": "chainlink", "LINK-USD": "chainlink",
            "AVAX": "avalanche-2", "AVAX-USD": "avalanche-2",
            "MATIC": "matic-network", "MATIC-USD": "matic-network",
            "SHIB": "shiba-inu", "SHIB-USD": "shiba-inu",
            "PEPE": "pepe", "PEPE-USD": "pepe",
            "RENDER": "render-token", "RENDER-USD": "render-token"
        }
        
        self.KNOWN_CRYPTO = list(self.COINGECKO_MAP.keys())
        
        # Manual overrides for problematic tickers (User Request)
        self.TICKER_ALIASES = {
            "ICGA.FRA": "ICGA.DE",   # Yahoo uses .DE for Xetra/Frankfurt often
            "3CP": "3CP.F",          # Xiaomi on Frankfurt
            "RNDR-USD": "RENDER-USD", # Rebranding fallback
            "RENDER": "RENDER-USD",   # Naked ticker support
            "BYD": "BY6.F",           # User owns BYD EV (Frankfurt), not Boyd Gaming (NYSE)
            "TCT": "NNnD.F",          # Tencent Frankfurt
            "3XC": "3CP.F",           # Xiaomi Frankfurt
            "NUKL": "NUKL.DE"         # Global X Uranium
        }

    def get_crypto_data_coingecko(self, ticker):
        """
        Fetch real-time price and 24h change from CoinGecko.
        Returns (price, change_pct) or (None, None) if failed.
        """
        try:
            coin_id = self.COINGECKO_MAP.get(ticker.upper())
            if not coin_id:
                return None, None
            
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
            # Add User-Agent to avoid 403 blocks often typically enforced by CG
            headers = {'User-Agent': 'Mozilla/5.0'} 
            
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            
            if coin_id in data:
                price = data[coin_id]['usd']
                change_pct = data[coin_id]['usd_24h_change']
                return price, change_pct
            return None, None
        except Exception as e:
            logger.warning(f"CoinGecko API failed for {ticker}: {e}")
            return None, None

    def get_market_price(self, ticker):
        """
        Fetches the current market price for a ticker (Crypto or Stock).
        Prioritizes CoinGecko for crypto, then Yahoo.
        Handles aliases.
        """
        ticker_u = ticker.upper()
        
        # 0. Check Aliases
        search_ticker = self.TICKER_ALIASES.get(ticker_u, ticker_u)
        
        # 1. Try CoinGecko
        price, _ = self.get_crypto_data_coingecko(search_ticker)
        if price:
            return price
        
        # 2. Try Yahoo Finance
        try:
            # AUTO-FIX: If known crypto but not in aliases, append -USD for Yahoo
            # (Matches logic in webhook.py fetch_price_smart)
            if search_ticker in ["XRP", "ADA", "DOGE", "DOT", "LINK", "AVAX", "MATIC", "SHIB", "PEPE", "BTC", "ETH", "SOL"] and '-' not in search_ticker:
                 search_ticker = f"{search_ticker}-USD"
            
            t = yf.Ticker(search_ticker)
            # Try fast info first
            if 'currentPrice' in t.info:
                return t.info['currentPrice']
            if 'regularMarketPrice' in t.info:
                return t.info['regularMarketPrice']
                
            # Fallback to history
            hist = t.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            logger.warning(f"Yahoo Price fetch failed for {search_ticker} (orig: {ticker}): {e}")
            
        return None

    def get_technical_summary(self, ticker: str) -> str:
        """
        Fetches price history and calculates RSI and SMA.
        Returns a human-readable summary string.
        """
        try:
            ticker = ticker.upper().strip()
            # Apply Alias Mapping for Technical Analysis
            ticker = self.TICKER_ALIASES.get(ticker, ticker)
            
            # A. Try CoinGecko for Crypto Real-Time Data first
            cg_price, cg_change = self.get_crypto_data_coingecko(ticker)
            
            # 1. Fetch Data (3 months to ensure enough data for SMA50)
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

            # Get latest values (last row) to use ONLY if CoinGecko failed
            latest = df.iloc[-1]
            
            # Use CoinGecko price if available, else Yahoo
            price = cg_price if cg_price else latest['Close']
            
            rsi = latest['RSI']
            sma_50 = latest['SMA_50']
            sma_200 = latest['SMA_200']
            
            # Calculate 24h Change
            if cg_change:
                change_pct = cg_change
            elif len(df) > 1:
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

            # RSI Status
            rsi_status = "Neutral"
            if rsi > 70:
                rsi_status = "OVERBOUGHT (>70)"
            elif rsi < 30:
                rsi_status = "OVERSOLD (<30)"

            # ATH (All-Time High) Check - approximate from loaded period or max
            # Since we only fetch 6mo of data above, we might miss true ATH.
            # Let's fetch max history for strictly ATH check is expensive? 
            # Compromise: High of last 6mo (52w High approx)
            period_high = df['High'].max()
            dist_from_high = ((price - period_high) / period_high) * 100
            
            # 4. Format Summary
            summary = (
                f"Price: ${price:.2f} ({change_pct:+.2f}%), "
                f"RSI: {rsi:.1f} ({rsi_status}), "
                f"Trend: {trend}, "
                f"Diff from 6m High: {dist_from_high:.1f}%"
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
