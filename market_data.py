"""
Market Data & Analysis (L5)
==========================
Handles technical analysis, data fetching, and macro analysis.
Now includes:
- SectorAnalyst: Multi-timeframe momentum rotation signals.
- Technical Indicators: RSI, SMA, MACD, etc.
"""

import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import logging
import os
from typing import Union, Dict

# Set YFinance cache to /tmp for read-only filesystems (Vercel)
try:
    yf.set_tz_cache_location("/tmp/py-yfinance")
except Exception:
    pass

import datetime
import time
import logging
import requests

# Configure logging
logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self):
        logger.info("MarketData: Initializing Phase 3 (V4.0 - Smart Sourcing)...")
        self.dry_run = os.environ.get("DRY_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
        
        # [PERFORMANCE] Session-level cache to avoid redundant API calls
        self._price_cache = {}      # ticker -> (price, source, change_pct, timestamp)
        self._technical_cache = {}  # ticker -> (summary, timestamp)
        self._cache_ttl = 60        # Cache TTL in seconds (1 minute)
        
        # Shared across instances to avoid redundant FX fetches
        if not hasattr(MarketData, '_eur_usd_rate_shared'):
            MarketData._eur_usd_rate_shared = 1.05  # Initial fallback
            MarketData._last_fx_update = 0
            
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
            "RENDER": "render-token", "RENDER-USD": "render-token",
            "XMR": "monero", "XMR-USD": "monero"
        }
        
        self.KNOWN_CRYPTO = list(self.COINGECKO_MAP.keys())
        
        # Manual overrides for problematic tickers (Trade Republic / EU optimized)
        self.TICKER_ALIASES = {
            "AAPL": "AAPL",          # Keep primary listing as safe default
            "META": "MEW2",          # Meta (Base EU)
            "NVDA": "NVD.DE",        # Nvidia (Suffix required to avoid Short ETF)
            "MSFT": "MSF",           # Microsoft (Base EU)
            "GOOGL": "ABE",          # Alphabet (Base EU)
            "AMZN": "AMZ",           # Amazon (Base EU)
            "TSLA": "TL0",           # Tesla (Base EU)
            "JAZZ": "JAZZ",          # Jazz Pharma on NASDAQ
            "BYD": "BY6.F",          # BYD on Frankfurt
            "BYDDF": "BY6.F",
            "TCT": "NNnD.F",         # Tencent on Frankfurt
            "3XC": "3CP.F",          # Xiaomi on Frankfurt
            "3CP": "3CP.F",
            "ICGA.FRA": "ICGA.DE",   
            "RNDR-USD": "RENDER-USD", 
            "RENDER": "RENDER-USD",
            "NUKL": "U3O8.DE",
            "BTC": "BTC-EUR",        
            "ETH": "ETH-EUR",
            "SOL": "SOL-EUR",
            "BTC-USD": "BTC-EUR",
            "ETH-USD": "ETH-EUR",
            "SOL-USD": "SOL-EUR",
            "VAVX": "VAVX.DE",        # VanEck Avalanche ETN
            "VAVX.DE": "VAVX.DE",
            "AVAX": "AVAX-USD",       # Primary Crypto
            "DOGE": "DOGE-USD",
            "XRP": "XRP-USD",
            "XMR": "XMR-USD",
        }

        # Suppression list
        # Suppression list (Avoid common non-ticker hallucinations)
        self.SUPPRESSED_TICKERS = {'SPACEX', 'CLARITY', 'IRA', 'NASDAQ', 'NYSE', 'SEC', 'FED', 'USA', 'US'}

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
            # [OPTIMIZATION] Use fast_info (lighter) instead of .info (heavy JSON)
            if hasattr(t, 'fast_info'):
                try:
                    price = t.fast_info.get('last_price')
                    if price: return price
                except:
                    pass # Fallback to history
            
            # Fallback for older yfinance versions
            if hasattr(t, 'info') and 'currentPrice' in t.info:
                return t.info['currentPrice']
            if hasattr(t, 'info') and 'regularMarketPrice' in t.info:
                return t.info['regularMarketPrice']
                
            # Fallback to history
            hist = t.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            logger.warning(f"Yahoo Price fetch failed for {search_ticker} (orig: {ticker}): {e}")
            
        return None

    async def get_smart_price_eur_async(self, candidate_ticker, include_change=False):
        """Async wrapper for smart price fetching to allow parallelization."""
        import asyncio
        return await asyncio.to_thread(self.get_smart_price_eur, candidate_ticker, include_change)

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
        # [PERFORMANCE] Refresh EUR/USD rate once per hour
        if time.time() - MarketData._last_fx_update > 3600 or MarketData._eur_usd_rate_shared == 1.05:
            try:
                 # Use 5d to avoid weekend empty data
                 hist = yf.Ticker("EURUSD=X").history(period="5d")
                 if not hist.empty: 
                     rate = hist['Close'].iloc[-1]
                     if 0.8 < rate < 1.5:
                         MarketData._eur_usd_rate_shared = float(rate)
                         MarketData._last_fx_update = time.time()
                         logger.info(f"FX Rate Updated: {MarketData._eur_usd_rate_shared}")
            except Exception as fx_err:
                 logger.warning(f"FX Update failed: {fx_err}")
        
        eur_usd_rate = MarketData._eur_usd_rate_shared
        change_pct = 0.0

        ticker_u = candidate_ticker.upper()
        
        # 0. Check Aliases first (Manual overrides) - PRIORITIZE OVER CACHE
        if ticker_u in self.TICKER_ALIASES:
             start_ticker = self.TICKER_ALIASES[ticker_u]
        else:
             start_ticker = ticker_u

        # [V11] Check DB ticker_cache
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # First, try to get a persistent fresh price
            persistent_cached = db.get_cached_price(ticker_u, max_age_minutes=15)
            if persistent_cached:
                raw_price = persistent_cached["price"]
                is_crypto = persistent_cached["is_crypto"]
                currency = persistent_cached["currency"]
                
                price_eur = raw_price
                if currency == "USD":
                    price_eur = raw_price / eur_usd_rate
                elif currency == "HKD":
                    price_eur = raw_price / (eur_usd_rate * 7.8)
                
                logger.info(f"Persistent Cache HIT: {ticker_u} -> €{price_eur:.2f} (Source: {currency})")
                self._price_cache[cache_key] = {'price': price_eur, 'source': 'db_cache', 'change': 0.0, 'ts': time.time()}
                return (price_eur, 'db_cache', 0.0) if include_change else (price_eur, 'db_cache')

            # Fallback to Ticker Resolution Cache (Old logic)
            cached_ticker = db.get_ticker_cache(ticker_u)
            if cached_ticker:
                resolved = cached_ticker.get("resolved_ticker")
                is_crypto = cached_ticker.get("is_crypto", False)
                currency = cached_ticker.get("currency", "USD")
                logger.debug(f"Ticker resolution cache HIT: {ticker_u} -> {resolved}")
                
                try:
                    t_obj = yf.Ticker(resolved)
                    hist = t_obj.history(period="1d")
                    if not hist.empty:
                        raw_price = hist['Close'].iloc[-1]
                        if include_change: change_pct = extract_change(t_obj)
                        
                        # Cache in persistent DB using NATIVE currency
                        db.save_ticker_price(ticker_u, raw_price, is_crypto=is_crypto, currency=currency, resolved_ticker=resolved)
                        
                        # Convert to EUR for immediate return
                        price_eur = raw_price
                        if currency == "USD":
                            price_eur = raw_price / eur_usd_rate
                        elif currency == "HKD":
                            price_eur = raw_price / (eur_usd_rate * 7.8)
                        
                        # Cache in memory
                        self._price_cache[cache_key] = {'price': price_eur, 'source': resolved, 'change': change_pct, 'ts': time.time()}
                        return (*[price_eur, resolved, change_pct], ) if include_change else (price_eur, resolved)
                except Exception as e:
                    db.increment_ticker_fail(ticker_u)
                    logger.debug(f"Ticker resolution cache MISS (fetch failed): {ticker_u} -> {resolved}")
        except:
            pass  # DB not available, continue without cache
        
        # start_ticker is already resolved via aliases above

        # OPTIMIZATION: If ticker has '-' or is known crypto
        # This list should match known crypto list in COINGECKO_MAP roughly
        updated_crypto_list = ['RENDER', 'SOL', 'BTC', 'ETH', 'XRP', 'ADA', 'DOGE', 'DOT', 'LINK', 'AVAX', 'MATIC', 'SHIB', 'PEPE', 'XMR']
        
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
                
                # [FIX] Save to DB (Native USD) with correct resolved ticker
                try:
                    from db_handler import DBHandler
                    # CoinGecko is "USD" by default here. Ensure we save the RESOLVED ticker so Yahoo doesn't hit ETF later.
                    DBHandler().save_ticker_price(ticker_u, cg_price, is_crypto=True, currency="USD", resolved_ticker=base + "-USD")
                except: pass
                
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
                        
                        # [FIX] Save to DB (Native EUR)
                        try:
                            from db_handler import DBHandler
                            DBHandler().save_ticker_price(ticker_u, result[0], is_crypto=True, currency="EUR")
                        except: pass

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
                     
                     # [FIX] Save to DB (Native USD)
                     try:
                         from db_handler import DBHandler
                         DBHandler().save_ticker_price(ticker_u, price, is_crypto=True, currency="USD")
                     except: pass
                     
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
            'MU', 'AVGO', 'QCOM', 'COST', 'ADBE', 'CRM', 'ORCL', 'CSCO',
            # Healthcare & Pharma
            'JAZZ', 'PFE', 'MRK', 'LLY', 'ABBV', 'BMY', 'GILD', 'BIIB', 'MRNA', 'VRTX',
            # Additional Tech
            'PANW', 'ZS', 'CRWD', 'NET', 'DDOG', 'ZM', 'OKTA', 'TWLO',
            # Finance & Other
            'BLK', 'MS', 'C', 'WFC', 'BAC', 'AXP', 'SCHW',
        }
        
        # Try suffixes
        initial_candidates = [start_ticker]
        if '.' not in start_ticker:
            # During early morning (08:00 - 10:00 CET), Frankfurt (.F) often leads Xetra (.DE) for US Stocks
            now_cet = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=1)))
            if 8 <= now_cet.hour <= 10:
                # Early birds: try Frankfurt first
                suffixes_eur = ['.F', '.DE', '.MI', '.PA', '.MC', '.AS']
            else:
                suffixes_eur = ['.DE', '.F', '.MI', '.PA', '.MC', '.AS']
            
            initial_candidates = [start_ticker + s for s in suffixes_eur] + [start_ticker]

        # Safety fallback: if alias/cached ticker fails, also try original input ticker.
        if ticker_u != start_ticker:
            if '.' not in ticker_u:
                initial_candidates += [ticker_u + s for s in suffixes_eur] + [ticker_u]
            else:
                initial_candidates += [ticker_u]

        # Deduplicate while preserving order
        initial_candidates = list(dict.fromkeys(initial_candidates))
        
        best_result = None
        best_change = 0.0
        best_currency = "USD"
        
        for t_test in initial_candidates:
            try:
                t_obj = yf.Ticker(t_test)
                hist = t_obj.history(period="1d")
                if not hist.empty:
                    last_row = hist.iloc[-1]
                    price = last_row['Close']
                    change_pct = 0.0
                    if include_change: change_pct = extract_change(t_obj)
                    
                    # Currency detection based on suffix
                    if t_test.endswith('.HK'):
                        hkd_eur = eur_usd_rate * 7.8 
                        converted_price = price / hkd_eur
                        currency = "HKD"
                    elif '.' in t_test:
                        converted_price = price
                        currency = "EUR"
                    else:
                        converted_price = price / eur_usd_rate
                        currency = "USD"
                    
                    result = (converted_price, t_test)
                    
                    # [FRESHNESS CHECK]
                    row_date = hist.index[-1].date()
                    is_today = (row_date == datetime.date.today())
                    
                    if is_today:
                        self._price_cache[cache_key] = {
                            'price': result[0], 'source': result[1], 
                            'change': change_pct, 'ts': time.time()
                        }
                        try:
                            from db_handler import DBHandler
                            db = DBHandler()
                            db.save_ticker_price(ticker_u, price, is_crypto=False, currency=currency, resolved_ticker=t_test)
                        except: pass
                        
                        return (*result, change_pct) if include_change else result
                    
                    if not best_result:
                        best_result = result
                        best_change = change_pct
                        best_currency = currency
                
            except Exception as e:
                logger.debug(f"Failed to fetch {t_test}: {e}")
                continue
        
        # [V12] FINAL SECURITY: If it was a known crypto, DO NOT return a stock price
        # This prevents "XRP" (Bitwise ETF) from poisoning "XRP" (Ripple)
        if best_result and ticker_u in updated_crypto_list:
            # If the source doesn't look like a crypto pair, reject it
            source = best_result[1]
            if "-USD" not in source and "-EUR" not in source:
                logger.warning(f"Rejecting stock resolution for known crypto {ticker_u}: {source}")
                return (0.0, None, 0.0) if include_change else (0.0, None)

        if best_result:
             self._price_cache[cache_key] = {
                'price': best_result[0], 'source': best_result[1], 
                'change': best_change, 'ts': time.time()
             }
             return (*best_result, best_change) if include_change else best_result
             
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
                    if ticker.upper() not in self.SUPPRESSED_TICKERS:
                        logger.warning(f"MTF analysis fail for {ticker} {label}: {e}")
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

    def get_technical_summary(self, ticker: str, return_dict: bool = False) -> Union[str, Dict]:
        """
        Fetches price history and calculates RSI and SMA.
        Returns a human-readable summary string OR a dictionary if return_dict=True.
        """
        import time
        cache_key = f"tech_{ticker.upper()}"
        
        # [PERFORMANCE] Check cache first
        if cache_key in self._technical_cache:
            cached = self._technical_cache[cache_key]
            if time.time() - cached['ts'] < self._cache_ttl:
                if return_dict:
                    return cached.get('data', {})
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
                return {} if return_dict else "No Market Data Available"

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
            
            rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50.0
            sma_50 = float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else price
            sma_200 = float(latest['SMA_200']) if not pd.isna(latest['SMA_200']) else price
            
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
            price_eur, found_ticker, smart_change = self.get_smart_price_eur(ticker, include_change=True)
            
            if price_eur > 0:
                display_price = price_eur
                display_sym = "€"
                if smart_change != 0:
                    change_pct = smart_change
            else:
                display_price = price
                display_sym = "$" 
            
            # 4. Format Summary
            source_info = f" (via {found_ticker})" if 'found_ticker' in locals() and found_ticker else ""
            summary = (
                f"CURRENT_PRICE: {display_sym}{display_price:.2f} {change_pct:+.2f}% [Market: {display_sym}]{source_info}, "
                f"RSI: {rsi:.1f} ({rsi_status}), "
                f"Trend: {trend}, "
                f"Diff from 6m High: {dist_from_high:.1f}%"
            )
            
            # [PHASE 4] Data Object
            data = {
                "price": display_price,
                "currency": "EUR" if price_eur > 0 else "USD",
                "rsi": rsi,
                "rsi_status": rsi_status,
                "trend": trend,
                "change_24h": change_pct,
                "dist_from_high": dist_from_high,
                "sma_50": sma_50,
                "sma_200": sma_200
            }
            
            # Cache both
            self._technical_cache[cache_key] = {
                'ts': time.time(),
                'summary': summary,
                'data': data
            }
            
            return data if return_dict else summary
            
            # [PERFORMANCE] Store in cache
            self._technical_cache[cache_key] = {'summary': summary, 'ts': time.time()}
            return summary
        
        except Exception as e:
            logger.warning(f"Technical summary failed for {ticker}: {e}")
            error_summary = f"Technical: Unknown (Error for {ticker})"
            # Cache error too to avoid retrying immediately
            self._technical_cache[cache_key] = {'summary': error_summary, 'ts': time.time()}
            return error_summary

    # =========================================================================
    # ATR (Average True Range) - Level 11: Dynamic Stop-Loss
    # =========================================================================
    
    def calculate_atr(self, ticker: str, period: int = 14) -> dict:
        """
        Calculate Average True Range for dynamic stop-loss levels.
        
        Args:
            ticker: Asset ticker
            period: ATR period (default 14 days)
        
        Returns:
            {
                "atr": 5.23,           # ATR value in asset currency
                "atr_pct": 2.5,        # ATR as % of current price
                "suggested_stop": 5.0, # Suggested stop-loss % (2x ATR)
                "volatility": "medium" # low/medium/high classification
            }
        """
        try:
            yf_ticker = self.TICKER_ALIASES.get(ticker.upper(), ticker.upper())
            
            # Handle crypto
            crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'RENDER', 'DOGE']
            base = ticker.replace('-USD', '').replace('-EUR', '')
            if base.upper() in crypto_list and not yf_ticker.endswith('-USD'):
                yf_ticker = f"{base}-USD"
            
            # Fetch data
            data = yf.download(yf_ticker, period="60d", progress=False, auto_adjust=True)
            
            if data.empty or len(data) < 5: # At least 5 days for a rough estimate
                return {"atr": 0, "atr_pct": 0, "suggested_stop": 10.0, "volatility": "unknown"}
            
            # If not enough for 14-day SMA, use what we have
            actual_period = min(period, len(data) - 1)
            
            # Handle MultiIndex
            if hasattr(data.columns, 'levels'):
                high = data['High'].iloc[:, 0] if data['High'].ndim > 1 else data['High']
                low = data['Low'].iloc[:, 0] if data['Low'].ndim > 1 else data['Low']
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                high = data['High']
                low = data['Low']
                close = data['Close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR (Simple Moving Average of TR)
            atr = tr.rolling(window=actual_period).mean().iloc[-1]
            current_price = close.iloc[-1]
            
            # ATR as percentage of price
            atr_pct = (atr / current_price) * 100
            
            # Suggested stop-loss = 2x ATR (common practice)
            suggested_stop = atr_pct * 2
            
            # Volatility classification
            if atr_pct < 2:
                volatility = "low"
            elif atr_pct < 5:
                volatility = "medium"
            else:
                volatility = "high"
            
            return {
                "atr": round(float(atr), 4),
                "atr_pct": round(float(atr_pct), 2),
                "suggested_stop": round(float(suggested_stop), 1),
                "volatility": volatility
            }
            
        except Exception as e:
            logger.warning(f"ATR calculation failed for {ticker}: {e}")
            return {"atr": 0, "atr_pct": 0, "suggested_stop": 10.0, "volatility": "unknown"}

    # =========================================================================
    # Correlation Matrix - Level 11: Diversification Analysis
    # =========================================================================
    
    def calculate_correlation_matrix(self, tickers: list, period: str = "90d") -> dict:
        """
        Calculate correlation matrix between assets for diversification analysis.
        
        Args:
            tickers: List of ticker symbols
            period: Historical period for correlation (default 90 days)
        
        Returns:
            {
                "matrix": {
                    "BTC-USD": {"ETH-USD": 0.92, "AAPL": 0.45},
                    "ETH-USD": {"BTC-USD": 0.92, "AAPL": 0.38}
                },
                "high_correlation_pairs": [("BTC-USD", "ETH-USD", 0.92)],
                "diversification_score": 65  # 0-100, higher is better
            }
        """
        try:
            if len(tickers) < 2:
                return {"matrix": {}, "high_correlation_pairs": [], "diversification_score": 100}
            
            # Normalize tickers
            normalized = []
            for t in tickers:
                t_upper = t.upper()
                t_resolved = self.TICKER_ALIASES.get(t_upper, t_upper)
                
                # Handle crypto
                crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'RENDER', 'DOGE']
                base = t_resolved.replace('-USD', '').replace('-EUR', '')
                if base in crypto_list and not t_resolved.endswith('-USD'):
                    t_resolved = f"{base}-USD"
                
                normalized.append(t_resolved)
            
            # Download all data
            data = yf.download(normalized, period=period, progress=False, auto_adjust=True)
            
            if data.empty:
                return {"matrix": {}, "high_correlation_pairs": [], "diversification_score": 50}
            
            # Extract Close prices
            if 'Close' in data.columns:
                if hasattr(data['Close'], 'columns'):
                    close_prices = data['Close']
                else:
                    close_prices = data[['Close']]
            else:
                close_prices = data
            
            # Calculate daily returns
            returns = close_prices.pct_change(fill_method=None).dropna()
            
            if returns.empty or len(returns) < 10:
                return {"matrix": {}, "high_correlation_pairs": [], "diversification_score": 50}
            
            # Calculate correlation matrix
            corr_matrix = returns.corr()
            
            # Convert to dictionary format
            matrix_dict = {}
            high_corr_pairs = []
            
            for i, t1 in enumerate(corr_matrix.columns):
                matrix_dict[t1] = {}
                for j, t2 in enumerate(corr_matrix.columns):
                    if t1 != t2:
                        corr_val = corr_matrix.loc[t1, t2]
                        matrix_dict[t1][t2] = round(corr_val, 3)
                        
                        # Track high correlation pairs (avoid duplicates)
                        if i < j and abs(corr_val) > 0.7:
                            high_corr_pairs.append((t1, t2, round(corr_val, 2)))
            
            # Calculate diversification score
            # Lower avg correlation = better diversification
            import numpy as np
            avg_corr = corr_matrix.values[~np.eye(len(corr_matrix), dtype=bool)].mean()
            diversification_score = max(0, min(100, int((1 - abs(avg_corr)) * 100)))
            
            return {
                "matrix": matrix_dict,
                "high_correlation_pairs": sorted(high_corr_pairs, key=lambda x: -abs(x[2])),
                "diversification_score": diversification_score
            }
            
        except Exception as e:
            logger.warning(f"Correlation matrix calculation failed: {e}")
            return {"matrix": {}, "high_correlation_pairs": [], "diversification_score": 50}


    # =========================================================================
    # VALUATION CONTEXT - Quick Win #2: P/E, PEG, Sector Comparison
    # =========================================================================
    
    def get_valuation_context(self, ticker: str) -> dict:
        """
        Get valuation metrics for a ticker to help AI assess fair value.
        
        Returns:
            {
                "pe_ratio": 25.3,
                "forward_pe": 22.1,
                "peg_ratio": 1.5,
                "price_to_book": 4.2,
                "sector": "Technology",
                "sector_pe_avg": 28.0,
                "valuation_vs_sector": "undervalued",  # undervalued/fair/overvalued
                "summary": "NVDA trades at P/E 25.3 (sector avg 28), PEG 1.5 - fairly valued with growth premium"
            }
        """
        try:
            import yfinance as yf
            
            yf_ticker = self.TICKER_ALIASES.get(ticker.upper(), ticker.upper())
            
            # Crypto doesn't have traditional valuation metrics
            crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'RENDER', 'DOGE']
            base = ticker.replace('-USD', '').replace('-EUR', '')
            if base.upper() in crypto_list:
                return {
                    "pe_ratio": None,
                    "forward_pe": None,
                    "peg_ratio": None,
                    "price_to_book": None,
                    "sector": "Cryptocurrency",
                    "sector_pe_avg": None,
                    "valuation_vs_sector": "N/A",
                    "summary": f"{ticker} is a cryptocurrency - traditional valuation metrics don't apply"
                }
            
            stock = yf.Ticker(yf_ticker)
            info = stock.info
            
            pe_ratio = info.get('trailingPE')
            forward_pe = info.get('forwardPE')
            peg_ratio = info.get('pegRatio')
            price_to_book = info.get('priceToBook')
            sector = info.get('sector', 'Unknown')
            
            # Sector average P/E estimates (rough benchmarks)
            sector_pe_benchmarks = {
                'Technology': 28,
                'Healthcare': 22,
                'Financial Services': 12,
                'Consumer Cyclical': 18,
                'Consumer Defensive': 20,
                'Energy': 10,
                'Industrials': 18,
                'Basic Materials': 14,
                'Utilities': 16,
                'Real Estate': 35,
                'Communication Services': 20,
            }
            sector_pe_avg = sector_pe_benchmarks.get(sector, 20)
            
            # Determine valuation status
            valuation_vs_sector = "fair"
            if pe_ratio:
                if pe_ratio < sector_pe_avg * 0.8:
                    valuation_vs_sector = "undervalued"
                elif pe_ratio > sector_pe_avg * 1.25:
                    valuation_vs_sector = "overvalued"
            
            # Build summary
            summary_parts = []
            if pe_ratio:
                summary_parts.append(f"P/E {pe_ratio:.1f}")
                summary_parts.append(f"sector avg ~{sector_pe_avg}")
            if peg_ratio:
                peg_comment = "attractive" if peg_ratio < 1.0 else ("reasonable" if peg_ratio < 2.0 else "expensive")
                summary_parts.append(f"PEG {peg_ratio:.2f} ({peg_comment})")
            
            summary = f"{ticker} ({sector}): " + ", ".join(summary_parts) if summary_parts else f"{ticker}: No valuation data available"
            summary += f" - {valuation_vs_sector}"
            
            return {
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "peg_ratio": round(peg_ratio, 2) if peg_ratio else None,
                "price_to_book": round(price_to_book, 2) if price_to_book else None,
                "sector": sector,
                "sector_pe_avg": sector_pe_avg,
                "valuation_vs_sector": valuation_vs_sector,
                "summary": summary
            }
            
        except Exception as e:
            logger.warning(f"Valuation context failed for {ticker}: {e}")
            return {
                "pe_ratio": None,
                "forward_pe": None, 
                "peg_ratio": None,
                "price_to_book": None,
                "sector": "Unknown",
                "sector_pe_avg": None,
                "valuation_vs_sector": "unknown",
                "summary": f"{ticker}: Valuation data unavailable"
            }

    # =========================================================================
    # VOLATILITY SCALAR - Quick Win #4: Position Size Adjustment
    # =========================================================================
    
    def get_volatility_scalar(self, ticker: str) -> dict:
        """
        Calculate position sizing scalar based on volatility.
        High volatility assets should have smaller position sizes.
        
        Returns:
            {
                "atr_pct": 3.5,           # ATR as % of price
                "volatility_class": "medium",  # low/medium/high/extreme
                "position_scalar": 0.75,  # Multiply suggested position by this
                "max_position_pct": 8.0,  # Maximum position % for this asset
                "reason": "Medium volatility - standard position sizing applies"
            }
        """
        try:
            atr_data = self.calculate_atr(ticker)
            atr_pct = atr_data.get('atr_pct', 3.0)
            
            # Position sizing based on ATR
            # Idea: Target same $ risk per position regardless of volatility
            # Low vol (< 2%): can take larger positions
            # High vol (> 5%): smaller positions to control risk
            
            if atr_pct < 1.5:
                volatility_class = "low"
                position_scalar = 1.25  # Can size up 25%
                max_position_pct = 15.0
                reason = "Low volatility - can take larger position"
            elif atr_pct < 3.0:
                volatility_class = "medium"
                position_scalar = 1.0  # Standard sizing
                max_position_pct = 10.0
                reason = "Medium volatility - standard position sizing"
            elif atr_pct < 5.0:
                volatility_class = "high"
                position_scalar = 0.75  # Reduce 25%
                max_position_pct = 7.0
                reason = "High volatility - reduce position size"
            else:
                volatility_class = "extreme"
                position_scalar = 0.5  # Cut in half
                max_position_pct = 5.0
                reason = "Extreme volatility - minimal position recommended"
            
            return {
                "atr_pct": round(atr_pct, 2),
                "volatility_class": volatility_class,
                "position_scalar": position_scalar,
                "max_position_pct": max_position_pct,
                "reason": reason
            }
            
        except Exception as e:
            logger.warning(f"Volatility scalar failed for {ticker}: {e}")
            return {
                "atr_pct": 3.0,
                "volatility_class": "medium",
                "position_scalar": 1.0,
                "max_position_pct": 10.0,
                "reason": "Default sizing (volatility data unavailable)"
            }

    # =========================================================================
    # BACKTESTING SUPPORT - Historical Data with Disk Cache
    # =========================================================================

    def get_historical_data(self, ticker: str, days: int = 365, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for backtesting.
        Uses local disk cache (.cache/history_{ticker}.json) to save API calls.
        """
        import os
        import json
        from datetime import datetime, timedelta

        cache_dir = ".cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        ticker_u = ticker.upper()
        # Resolve aliases
        ticker_search = self.TICKER_ALIASES.get(ticker_u, ticker_u)
        
        # Crypto Alias Handling
        crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'RENDER', 'DOGE', 'ADA', 'DOT', 'LINK']
        base = ticker_search.replace('-USD', '').replace('-EUR', '')
        if base in crypto_list and not ticker_search.endswith('-USD'):
            ticker_search = f"{base}-USD"

        safe_ticker = ticker_search.replace('=', '').replace('^', '')
        cache_file = os.path.join(cache_dir, f"history_{safe_ticker}.json")
        
        # 1. Check Cache
        if not force_refresh and os.path.exists(cache_file):
            try:
                # Check file age (expire after 24h)
                file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
                if file_age < 86400: # 1 day
                    with open(cache_file, 'r') as f:
                        raw_data = json.load(f)
                    
                    # Convert back to DataFrame
                    df = pd.DataFrame(raw_data)
                    if not df.empty:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        logger.info(f"Loaded {len(df)} rows from cache for {ticker_search}")
                        return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")

        # 2. Fetch from API
        try:
            logger.info(f"Fetching {days}d history for {ticker_search} from API...")
            # Calculate start date string
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            data = yf.download(ticker_search, start=start_date, progress=False, auto_adjust=True)
            
            if data.empty:
                logger.warning(f"No data found for {ticker_search}")
                return pd.DataFrame()

            # Handle MultiIndex
            if hasattr(data.columns, 'levels'):
                df = data.copy()
                df.columns = df.columns.get_level_values(0)
            else:
                df = data.copy()

            # reset index to make Date a column for JSON serialization
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to Cache
            with open(cache_file, 'w') as f:
                f.write(df.to_json(orient='records', date_format='iso'))
            
            # Restore Index for return
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Failed to fetch history for {ticker}: {e}")
            return pd.DataFrame()


class SectorAnalyst:
    """
    Phase 3: Sector Rotation Signal.
    Analyzes relative strength of major US Sector ETFs to determine macro flows.
    """
    
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Energy": "XLE",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Industrials": "XLI",
        "Staples": "XLP",
        "Discretionary": "XLY",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "IYR",
        "Comm Services": "XLC",
        # "Semiconductors": "SMH" # Specific industry, maybe exclude from broad rotation
    }
    
    def __init__(self, market_instance=None):
        self.market = market_instance if market_instance else MarketData()
        
    def get_sector_ranking(self, limit=3):
        """
        Ranks sectors by momentum score (avg of 1M, 3M, 6M performance).
        Returns list of dicts: {'sector': 'Technology', 'ticker': 'XLK', 'momentum_score': 15.5, ...}
        """
        results = []
        
        # We need historical data for calculating returns
        # Ideally calculate efficiently (batch fetch? yfinance is one by one via Ticker)
        # We'll use 6mo history
        
        for sector, ticker in self.SECTOR_ETFS.items():
            try:
                # Use cached fetch if possible, but calculating locally is easier
                hist = self.market.get_historical_data(ticker, days=200)
                
                if hist.empty or len(hist) < 130: # Need ~6 months (126 trading days)
                    continue
                    
                current_price = hist['Close'].iloc[-1]
                
                # Helper to safe get past price
                def get_pct_change(days_ago):
                    if len(hist) > days_ago:
                        past_price = hist['Close'].iloc[-days_ago]
                        return ((current_price - past_price) / past_price) * 100
                    return 0.0
                
                pct_1m = get_pct_change(21)
                pct_3m = get_pct_change(63)
                pct_6m = get_pct_change(126)
                
                # Momentum Score: Weighted average favoring recent trend
                # Score = 40% 1M + 40% 3M + 20% 6M
                score = (pct_1m * 0.4) + (pct_3m * 0.4) + (pct_6m * 0.2)
                
                results.append({
                    "sector": sector,
                    "ticker": ticker,
                    "momentum_score": score,
                    "1m_pct": pct_1m,
                    "3m_pct": pct_3m,
                    "6m_pct": pct_6m,
                    "current_price": current_price
                })
                
            except Exception as e:
                logger.warning(f"SectorAnalyst: Failed to analyze {sector} ({ticker}): {e}")
                
        # Sort by score descending
        results.sort(key=lambda x: x['momentum_score'], reverse=True)
        return results
        
    def get_rotation_signals(self):
        """
        Generates actionable rotation advice.
        """
        ranking = self.get_sector_ranking(limit=len(self.SECTOR_ETFS))
        if not ranking:
            return []
            
        leaders = ranking[:3]
        laggards = ranking[-3:]
        
        signals = []
        signals.append(f"🔄 **Top Sectors (Accumulate):** {', '.join([x['sector'] for x in leaders])}")
        signals.append(f"📉 **Weakest Sectors (Avoid/Trim):** {', '.join([x['sector'] for x in laggards])}")
        
        # Specific rotation tip
        top = leaders[0]
        bottom = laggards[-1]
        
        # Only suggest rotation if the gap is significant (>10% momentum gap)
        if top['momentum_score'] - bottom['momentum_score'] > 10:
             signals.append(f"💡 **Rotation Idea:** Shift exposure from **{bottom['sector']}** to **{top['sector']}** (Momentum Gap: {top['momentum_score'] - bottom['momentum_score']:.1f}%)")
             
        return signals


if __name__ == "__main__":
    # Test
    md = MarketData()
    # history = md.get_historical_data("BTC", days=30) 
    # print(history.tail())
    # print(md.get_technical_summary("AAPL"))
    # print(md.get_technical_summary("BTC-USD"))
    
    # Test ATR
    # print("\nATR Test:")
    # print(md.calculate_atr("BTC-USD"))
    # print(md.calculate_atr("AAPL"))
    
    # Test Correlation
    # print("\nCorrelation Test:")
    # corr = md.calculate_correlation_matrix(["BTC-USD", "ETH-USD", "AAPL"])
    # print(f"High Correlation Pairs: {corr['high_correlation_pairs']}")
    # print(f"Diversification Score: {corr['diversification_score']}")
    
    # Test Sector Analyst
    print("\nSector Analyst Test:")
    sa = SectorAnalyst()
    print(sa.get_rotation_signals())
