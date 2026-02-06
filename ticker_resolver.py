"""
Ticker Resolver Utility
=======================
Central utility for resolving ticker symbols across all modules.
Uses the self-learning ticker_cache in Supabase.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Local session cache to avoid redundant DB lookups
_RESOLVER_CACHE = {}

# Fallback aliases when DB is unavailable
TICKER_ALIASES = {
    "RENDER": "RENDER-USD",
    "XMR": "XMR-USD",
    "XMR-USD": "XMR-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "DOT": "DOT-USD",
    "LINK": "LINK-USD",
    "AVAX": "AVAX-USD",
    "MATIC": "MATIC-USD",
    "TCT": "0700.HK",
    "3XC": "1810.HK",
    "NUKL": "U3O8.DE",
    "JAZZ": "JAZZ",
    "BYD": "BY6.F",
    "BYDDF": "BY6.F",
}

# Known crypto tickers (always append -USD if no suffix)
CRYPTO_TICKERS = {
    'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'LINK', 
    'AVAX', 'MATIC', 'SHIB', 'PEPE', 'RENDER', 'XMR', 'FET', 'NEAR',
    'UNI', 'AAVE', 'LTC', 'BCH', 'XLM', 'ALGO', 'ATOM', 'VET'
}

# Major assets that should NEVER be rejected by fail_count (safety net)
PROTECTED_TICKERS = {
    'BTC', 'BTC-USD', 'ETH', 'ETH-USD', 'SOL', 'SOL-USD', 'XRP', 'XRP-USD',
    'RENDER', 'RENDER-USD', 'XMR', 'XMR-USD', 'NVDA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT'
}

_TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,8}(?:[-.][A-Z0-9]{1,8})?$")
_TICKER_STOPWORDS = {
    "THE", "THIS", "THAT", "FROM", "WITH", "NEWS", "MARKET", "ALERT", "CHECK",
    "OWNED", "ENTRY", "FULL", "TEXT", "PORTFOLIO", "SIGNAL", "TRADING",
    "DCA", "BUY", "SELL", "HOLD", "MODEL", "QUOTE", "PAGE", "SITE", "HOME",
    "HTTP", "HTTPS", "WWW"
}


def is_probable_ticker(value: str) -> bool:
    """
    Lightweight ticker sanity check to avoid sending obvious noise to yfinance.
    """
    if not isinstance(value, str):
        return False

    ticker = value.strip().upper()
    if not ticker:
        return False

    if ticker in _TICKER_STOPWORDS:
        return False

    if not _TICKER_PATTERN.match(ticker):
        return False

    # Reject pure numeric tokens unless they are exchange-qualified (e.g. 0700.HK).
    if ticker.isdigit():
        return False

    return True

def resolve_ticker(ticker: str) -> str:
    """
    Resolve a user-provided ticker to its Yahoo Finance compatible format.
    
    Uses:
    1. DB ticker_cache (if available) - self-learning
    2. Local TICKER_ALIASES (fallback)
    3. Crypto detection (append -USD)
    
    Args:
        ticker: User-provided ticker symbol (e.g., "TCT", "BTC", "RENDER")
        
    Returns:
        Yahoo Finance compatible ticker (e.g., "0700.HK", "BTC-USD", "RENDER-USD")
    """
    if not isinstance(ticker, str) or not ticker.strip():
        return None

    ticker_u = ticker.upper()
    preferred_alias = TICKER_ALIASES.get(ticker_u)
    
    # 0. Check local session cache first
    if ticker_u in _RESOLVER_CACHE:
        return _RESOLVER_CACHE[ticker_u]

    # 1. Check DB cache first (self-learning)
    try:
        from db_handler import DBHandler
        db = DBHandler()
        cached = db.get_ticker_cache(ticker_u)
        if cached:
            # Check for known failures
            fail_count = cached.get("fail_count", 0) or 0
            if fail_count > 3 and ticker_u not in PROTECTED_TICKERS:
                logger.debug(f"Ticker cache REJECT (Too many failures): {ticker_u}")
                _RESOLVER_CACHE[ticker_u] = None
                return None # Explicitly reject

            resolved = cached.get("resolved_ticker", ticker_u)
            # Guard against stale cache entries for crypto tickers (e.g. XMR -> XMR)
            base = ticker_u.replace('-USD', '').replace('-EUR', '')
            if base in CRYPTO_TICKERS and '-' not in resolved:
                resolved = f"{base}-USD"
            # If we have an explicit local alias and cache returns raw ticker, prefer alias.
            if preferred_alias and resolved == ticker_u:
                resolved = preferred_alias
            logger.debug(f"Ticker cache HIT: {ticker_u} -> {resolved}")
            _RESOLVER_CACHE[ticker_u] = resolved
            return resolved
    except Exception as e:
        logger.debug(f"Ticker cache unavailable: {e}")
    
    # 2. Check local aliases
    resolved = ticker_u
    if preferred_alias:
        resolved = preferred_alias
    else:
        # 3. Crypto detection
        base = ticker_u.replace('-USD', '').replace('-EUR', '')
        if base in CRYPTO_TICKERS and '-' not in ticker_u:
            resolved = f"{base}-USD"
    
    _RESOLVER_CACHE[ticker_u] = resolved
    return resolved


def resolve_tickers(tickers: list) -> list:
    """
    Resolve multiple tickers at once.
    
    Args:
        tickers: List of user-provided ticker symbols
        
    Returns:
        List of Yahoo Finance compatible tickers
    """
    # Optimized batch resolution
    from db_handler import DBHandler
    db = DBHandler()
    
    # 1. Fetch from DB in one go
    cache_map = db.get_ticker_cache_batch(tickers)
    
    results = {}
    
    for t in tickers:
        if not isinstance(t, str) or not t.strip():
            results[t] = None
            continue
        t_u = t.upper()
        if not is_probable_ticker(t_u):
            results[t] = None
            continue
        # A. Check DB Cache
        if t_u in cache_map:
            record = cache_map[t_u]
            fail_count = record.get("fail_count", 0) or 0
            if fail_count > 3 and t_u not in PROTECTED_TICKERS:
                results[t] = None # Reject
            else:
                resolved = record.get("resolved_ticker", t_u)
                base = t_u.replace('-USD', '').replace('-EUR', '')
                if base in CRYPTO_TICKERS and '-' not in resolved:
                    resolved = f"{base}-USD"
                if t_u in TICKER_ALIASES and resolved == t_u:
                    resolved = TICKER_ALIASES[t_u]
                results[t] = resolved
            continue
            
        # B. Check Local Aliases
        if t_u in TICKER_ALIASES:
            results[t] = TICKER_ALIASES[t_u]
            continue
            
        # C. Crypto Check
        base = t_u.replace('-USD', '').replace('-EUR', '')
        if base in CRYPTO_TICKERS and '-' not in t_u:
            results[t] = f"{base}-USD"
            continue
            
        # D. Default
        results[t] = t_u
        
    return results


def get_ticker_currency(ticker: str) -> str:
    """
    Determine the currency of a ticker based on its suffix.
    
    Returns: "EUR", "USD", "HKD", etc.
    """
    if ticker.endswith('.HK'):
        return "HKD"
    elif ticker.endswith('.DE') or ticker.endswith('.F') or ticker.endswith('.MI'):
        return "EUR"
    elif ticker.endswith('.PA') or ticker.endswith('.MC') or ticker.endswith('.AS'):
        return "EUR"
    elif '-USD' in ticker or '-EUR' in ticker:
        return "USD" if '-USD' in ticker else "EUR"
    else:
        return "USD"  # Default to USD for US stocks


if __name__ == "__main__":
    # Test
    test_tickers = ['TCT', '3XC', 'NUKL', 'BTC', 'RENDER', 'NVDA', 'EUNL.DE']
    for t in test_tickers:
        resolved = resolve_ticker(t)
        currency = get_ticker_currency(resolved)
        print(f"{t} -> {resolved} ({currency})")
