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

import logging
import requests # Added for CoinGecko

# Configure logging
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        logger.info("MarketData Class Loaded (v2.1 - Smart Price Enabled)")
        
        # [PERFORMANCE] Session-level cache to avoid redundant API calls
        self._price_cache = {}      # ticker -> (price, source, change_pct, timestamp)
        self._technical_cache = {}  # ticker -> (summary, timestamp)
        self._cache_ttl = 60        # Cache TTL in seconds (1 minute)
        
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
            "BYDDF": "BY6.F",         # BYD OTC -> Frankfurt (faster)
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

    def get_smart_price_eur(self, candidate_ticker, include_change=False):
        """
        Centralized Smart Pricing Logic (Single Source of Truth).
        Attempts to find a price for the ticker trying multiple suffixes.
        Prioritizes EUR markets (DE, MI, F, PA) for Stocks.
        Handles Crypto (-USD) separately and converts to EUR.
        Returns: 
           - If include_change=False (default): (price_in_eur, found_ticker_suffix_or_alias)
           - If include_change=True: (price_in_eur, found_ticker_suffix_or_alias, change_percent_24h)
        """
        import time
        cache_key = candidate_ticker.upper()
        
        # [PERFORMANCE] Check cache first
        if cache_key in self._price_cache:
            cached = self._price_cache[cache_key]
            if time.time() - cached['ts'] < self._cache_ttl:
                if include_change:
                    return (cached['price'], cached['source'], cached['change'])
                return (cached['price'], cached['source'])
        
        change_pct = 0.0
        
        # Helper to extract change
        def extract_change(ticker_obj):
            try:
                # regularMarketChangePercent is already a percentage (e.g. -5.5 for -5.5%)
                # Do NOT multiply by 100!
                if hasattr(ticker_obj, 'info') and 'regularMarketChangePercent' in ticker_obj.info:
                    change = ticker_obj.info['regularMarketChangePercent']
                    # Some APIs return as decimal (0.05), some as percent (5.0)
                    # Yahoo returns as percent already, so just return it
                    if change is not None:
                        return float(change)
            except Exception as e:
                logger.warning(f"Could not extract change: {e}")
            return 0.0
        # Note: Fetch EUR/USD rate dynamically or fallback
        eur_usd_rate = 1.09
        try:
             hist = yf.Ticker("EURUSD=X").history(period="1d")
             if not hist.empty: eur_usd_rate = hist['Close'].iloc[-1]
        except: pass

        ticker_u = candidate_ticker.upper()
        # 0. Check Aliases first
        # Usually aliases map to the EXACT ticker intended (e.g. BYD -> BY6.F)
        # If alias is found, trust it 100% and fetch that.
        if ticker_u in self.TICKER_ALIASES:
             start_ticker = self.TICKER_ALIASES[ticker_u]
        else:
             start_ticker = ticker_u

        # OPTIMIZATION: If ticker has '-' or is known crypto
        # This list should match known crypto list in COINGECKO_MAP roughly
        updated_crypto_list = ['RENDER', 'SOL', 'BTC', 'ETH', 'XRP', 'ADA', 'DOGE', 'DOT', 'LINK', 'AVAX', 'MATIC', 'SHIB', 'PEPE']
        
        if '-' in start_ticker or start_ticker in updated_crypto_list:
            base = start_ticker.split('-')[0] if '-' in start_ticker else start_ticker
            
            # 1. Try CoinGecko First (Most reliable for Crypto)
            cg_price, _ = self.get_crypto_data_coingecko(base + "-USD")
            if cg_price and cg_price > 0:
                # CG doesn't give change here easily without another call. 
                # Let's rely on YF for change if CG provides price, OR just return 0 change for speed.
                # Actually, YF is better for Change%.
                if include_change:
                     try:
                         # Quick check YF for change
                         yt = yf.Ticker(base + "-USD")
                         change_pct = extract_change(yt) 
                     except: pass
                
                result = (cg_price / eur_usd_rate, base + "-USD")
                return (*result, change_pct) if include_change else result

            # 2. Try EUR Pair first (e.g. BTC-EUR) - Most accurate for EU users
            # SKIP for cryptos without EUR pair on Yahoo (causes errors)
            CRYPTO_NO_EUR = {'RENDER', 'SHIB', 'PEPE', 'MATIC'}  # Add more as discovered
            if base.upper() not in CRYPTO_NO_EUR:
                try:
                    t_obj = yf.Ticker(f"{base}-EUR")
                    hist = t_obj.history(period="1d")
                    if not hist.empty:
                        if include_change: change_pct = extract_change(t_obj)
                        result = (hist['Close'].iloc[-1], f"{base}-EUR")
                        return (*result, change_pct) if include_change else result
                except: pass

            # 3. Try USD Pair (e.g. BTC-USD) - Fallback
            try:
                t_obj = yf.Ticker(f"{base}-USD")
                hist = t_obj.history(period="1d")
                if not hist.empty:
                     price = hist['Close'].iloc[-1]
                     if include_change: change_pct = extract_change(t_obj)
                     result = (price / eur_usd_rate, f"{base}-USD")
                     return (*result, change_pct) if include_change else result
            except: pass
            
            return (0.0, None, 0.0) if include_change else (0.0, None)

        # 1. Try Explicit EUR Suffixes first (Trade Republic common markets)
        suffixes_eur = ['.DE', '.F', '.MI', '.PA', '.MC', '.AS']
        
        # OPTIMIZATION: Known US Stocks - skip EU suffix attempts
        US_STOCKS = {
            'NVDA', 'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 
            'AMD', 'INTC', 'SPY', 'QQQ', 'NFLX', 'UBER', 'COIN', 'HOOD',
            'BA', 'JPM', 'GS', 'V', 'MA', 'DIS', 'WMT', 'JNJ', 'PG',
            'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'HD', 'LOW', 'TGT',
            'CVX', 'XOM', 'COP', 'PYPL', 'SQ', 'SHOP', 'PLTR', 'SNOW',
            'MU', 'AVGO', 'QCOM', 'COST', 'ADBE', 'CRM', 'ORCL', 'CSCO'
        }
        
        # If start_ticker already has a suffix, try it directly first
        initial_candidates = [start_ticker]
        
        if '.' not in start_ticker:
            # For US stocks, skip EU suffixes entirely
            if start_ticker in US_STOCKS:
                initial_candidates = [start_ticker]  # Only try US ticker
            else:
                initial_candidates = [start_ticker + s for s in suffixes_eur] + [start_ticker]
        
        for t_test in initial_candidates:
            try:
                t_obj = yf.Ticker(t_test)
                hist = t_obj.history(period="1d")
                if not hist.empty:
                    # If it's a raw ticker (no dot) and not crypto, assumed USD?
                    # Usually stocks without suffix on Yahoo are US Stocks (USD).
                    # If ticker had suffix (e.g. .DE), price is EUR.
                    price = hist['Close'].iloc[-1]
                    if include_change: change_pct = extract_change(t_obj)
                    
                    if '.' in t_test: 
                         result = (price, t_test) # EUR
                    else:
                         result = (price / eur_usd_rate, t_test) # Convert USD (heuristic)
                    
                    # [PERFORMANCE] Store in cache
                    self._price_cache[cache_key] = {
                        'price': result[0], 'source': result[1], 
                        'change': change_pct, 'ts': time.time()
                    }
                    return (*result, change_pct) if include_change else result
            except: pass
            
        return (0.0, None, 0.0) if include_change else (0.0, None)

    # =========================================================================
    # MULTI-TIMEFRAME ANALYSIS (NEW - Predictive System L1)
    # =========================================================================
    
    def get_multi_timeframe_trend(self, ticker: str) -> dict:
        """
        Analyze trend across multiple timeframes (Daily, Weekly, Monthly).
        
        Returns:
            {
                "alignment": 0-3 (how many timeframes agree),
                "direction": "bullish" / "bearish" / "mixed",
                "timeframes": {"1d": "bullish", "1w": "neutral", "1mo": "bearish"},
                "confidence_boost": 0.85-1.15
            }
        """
        try:
            yf_ticker = self.TICKER_ALIASES.get(ticker, ticker)
            
            # Handle crypto
            crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'RENDER', 'DOGE']
            base = ticker.replace('-USD', '').replace('-EUR', '')
            if base in crypto_list and not yf_ticker.endswith('-USD'):
                yf_ticker = f"{base}-USD"
            
            trends = {}
            bullish_count = 0
            bearish_count = 0
            
            # Analyze each timeframe
            for period, label in [("5d", "1d"), ("1mo", "1w"), ("3mo", "1mo")]:
                try:
                    data = yf.download(yf_ticker, period=period, progress=False, auto_adjust=True)
                    
                    if data.empty or len(data) < 3:
                        trends[label] = "unknown"
                        continue
                    
                    # Handle MultiIndex
                    if hasattr(data.columns, 'levels'):
                        close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
                    else:
                        close = data['Close']
                    
                    # Simple trend: compare first third to last third
                    n = len(close)
                    first_avg = float(close.iloc[:n//3].mean())
                    last_avg = float(close.iloc[-n//3:].mean())
                    
                    change_pct = ((last_avg - first_avg) / first_avg) * 100
                    
                    if change_pct > 2:
                        trends[label] = "bullish"
                        bullish_count += 1
                    elif change_pct < -2:
                        trends[label] = "bearish"
                        bearish_count += 1
                    else:
                        trends[label] = "neutral"
                        
                except Exception as e:
                    logger.warning(f"MTF analysis failed for {ticker} {label}: {e}")
                    trends[label] = "unknown"
            
            # Determine overall direction
            if bullish_count >= 2:
                direction = "bullish"
                alignment = bullish_count
            elif bearish_count >= 2:
                direction = "bearish"
                alignment = bearish_count
            else:
                direction = "mixed"
                alignment = max(bullish_count, bearish_count)
            
            # Confidence boost based on alignment
            if alignment == 3:
                confidence_boost = 1.15  # All timeframes agree
            elif alignment == 2:
                confidence_boost = 1.05
            else:
                confidence_boost = 0.90  # Mixed signals, reduce confidence
            
            return {
                "alignment": alignment,
                "direction": direction,
                "timeframes": trends,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "confidence_boost": confidence_boost
            }
            
        except Exception as e:
            logger.warning(f"Multi-timeframe analysis failed for {ticker}: {e}")
            return {"alignment": 0, "direction": "unknown", "confidence_boost": 1.0, "error": str(e)}

    def get_technical_summary(self, ticker: str) -> str:
        """
        Fetches price history and calculates RSI and SMA.
        Returns a human-readable summary string.
        """
        import time
        cache_key = f"tech_{ticker.upper()}"
        
        # [PERFORMANCE] Check cache first
        if cache_key in self._technical_cache:
            cached = self._technical_cache[cache_key]
            if time.time() - cached['ts'] < self._cache_ttl:
                return cached['summary']
        
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

            # ATH (All-Time High) Check
            period_high = df['High'].max()
            dist_from_high = ((price - period_high) / period_high) * 100
            
            # [CRITICAL UPDATE] Force Price to EUR using Smart Logic
            # This ensures AI sees the same price as the user's portfolio
            price_eur, _, smart_change = self.get_smart_price_eur(ticker, include_change=True)
            
            if price_eur > 0:
                display_price = price_eur
                display_sym = "€"
                # If we have a smart change, use it (it might be more accurate/recent)
                if smart_change != 0:
                    change_pct = smart_change
            else:
                display_price = price
                display_sym = "$" # Fallback
            
            # 4. Format Summary
            summary = (
                f"Price: {display_sym}{display_price:.4f} ({change_pct:+.2f}%), "
                f"RSI: {rsi:.1f} ({rsi_status}), "
                f"Trend: {trend}, "
                f"Diff from 6m High: {dist_from_high:.1f}%"
            )
            
            # [PERFORMANCE] Store in cache
            self._technical_cache[cache_key] = {'summary': summary, 'ts': time.time()}
            return summary
        
        except Exception as e:
            logger.warning(f"Technical summary failed for {ticker}: {e}")
            error_summary = f"Technical: Unknown (Error for {ticker})"
            # Cache error too to avoid retrying immediately
            self._technical_cache[cache_key] = {'summary': error_summary, 'ts': time.time()}
            return error_summary

if __name__ == "__main__":
    # Test
    md = MarketData()
    print(md.get_technical_summary("AAPL"))
    print(md.get_technical_summary("BTC-USD"))
