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
    return [resolve_ticker(t) for t in tickers]


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
