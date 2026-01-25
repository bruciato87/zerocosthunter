"""
Ticker Resolver Utility
=======================
Central utility for resolving ticker symbols across all modules.
Uses the self-learning ticker_cache in Supabase.
"""

import logging

logger = logging.getLogger(__name__)

# Fallback aliases when DB is unavailable
TICKER_ALIASES = {
    "RENDER": "RENDER-USD",
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
    'AVAX', 'MATIC', 'SHIB', 'PEPE', 'RENDER', 'FET', 'NEAR',
    'UNI', 'AAVE', 'LTC', 'BCH', 'XLM', 'ALGO', 'ATOM', 'VET'
}

# Major assets that should NEVER be rejected by fail_count (safety net)
PROTECTED_TICKERS = {
    'BTC', 'BTC-USD', 'ETH', 'ETH-USD', 'SOL', 'SOL-USD', 'XRP', 'XRP-USD',
    'RENDER', 'RENDER-USD', 'NVDA', 'AAPL', 'GOOGL', 'AMZN', 'MSFT'
}

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
    ticker_u = ticker.upper()
    
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
                return None # Explicitly reject

            resolved = cached.get("resolved_ticker", ticker_u)
            logger.debug(f"Ticker cache HIT: {ticker_u} -> {resolved}")
            return resolved
    except Exception as e:
        logger.debug(f"Ticker cache unavailable: {e}")
    
    # 2. Check local aliases
    if ticker_u in TICKER_ALIASES:
        return TICKER_ALIASES[ticker_u]
    
    # 3. Crypto detection
    base = ticker_u.replace('-USD', '').replace('-EUR', '')
    if base in CRYPTO_TICKERS and '-' not in ticker_u:
        return f"{base}-USD"
    
    # 4. Return as-is
    return ticker_u


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
        t_u = t.upper()
        # A. Check DB Cache
        if t_u in cache_map:
            record = cache_map[t_u]
            fail_count = record.get("fail_count", 0) or 0
            if fail_count > 3 and t_u not in PROTECTED_TICKERS:
                results[t] = None # Reject
            else:
                results[t] = record.get("resolved_ticker", t_u)
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
