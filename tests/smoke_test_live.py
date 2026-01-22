
import sys
import unittest.mock
import logging

# Mock DBHandler to prevent database connection attempts
mock_db = unittest.mock.MagicMock()
sys.modules['db_handler'] = mock_db

# Mock ticker_resolver for main logic re-use (we want to test regex mostly)
# But we can import the real one if we want, assuming it doesn't hit DB hard or fail.
# ticker_resolver imports DBHandler too. So mocking db_handler first is good.
sys.modules['ticker_resolver'] = unittest.mock.MagicMock()

# Mock trafilatura and curl_cffi to avoid installation issues or complexity
sys.modules['trafilatura'] = unittest.mock.MagicMock()
sys.modules['curl_cffi'] = unittest.mock.MagicMock()
sys.modules['curl_cffi.requests'] = unittest.mock.MagicMock()

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup Logging
logging.basicConfig(level=logging.WARNING)

from hunter import NewsHunter
import re

# COPY OF THE LOGIC FROM MAIN.PY (To verify identical behavior)
IGNORE_LIST = {
    'THE', 'AND', 'FOR', 'NEW', 'CEO', 'IPO', 'AI', 'US', 'EU', 'UK', 'HK',
    'API', 'APP', 'BIG', 'BUY', 'NOW', 'TOP', 'HOT', 'SOS', 'RUN', 'SET',
    'EFT', 'ETF', 'CRYPTO', 'BITCOIN', 'ETHEREUM', 'SOLANA', 'RIPPLE', 
    'CARDANO', 'DOGECOIN', 'POLKADOT', 'CHAINLINK', 'AVALANCHE', 'POLYGON',
    'THIS', 'THAT', 'WITH', 'FROM', 'INTO', 'OVER', 'MORE', 'LESS', 'BEST',
    'REAL', 'TIME', 'YEAR', 'WEEK', 'DAY', 'HOUR', 'LIFE', 'GOOD', 'BAD',
    'LOW', 'HIGH', 'MAX', 'MIN', 'ONE', 'TWO', 'SIX', 'TEN', 'ALL', 'ANY',
    'CAN', 'GET', 'HAS', 'HAD', 'NOT', 'BUT', 'WHY', 'HOW', 'WHO', 'ITS',
    'WAR', 'TAX', 'LAW', 'JOB', 'PAY', 'FEE', 'WIN', 'LOSE', 'NET', 'GRO',
    'GDP', 'CPI', 'PPI', 'FED', 'ECB', 'SEC', 'DOJ', 'FTX', 'SBF', 'KYC',
    'AML', 'NFT', 'DAO', 'DEX', 'CEX', 'PUMP', 'DUMP', 'FOMO', 'FUD', 'ATH',
    'ATL', 'ROI', 'APR', 'APY', 'TVL', 'MCAP', 'VOL', 'PNL', 'YTD', 'QTD',
    'LTD', 'INC', 'CORP', 'LLC', 'PLC', 'AG', 'GMBH', 'SA', 'SPA', 'NV', 'BV',
    'UP', 'DOWN', 'LEFT', 'RIGHT', 'NORTH', 'SOUTH', 'EAST', 'WEST',
    'HITS', 'MISS', 'BEAT', 'DROP', 'FALL', 'RISE', 'JUMP', 'DIVE', 'SOAR',
    'SURGE', 'TANK', 'CRASH', 'BOOM', 'BUST', 'HOLD', 'SELL', 'SWAP', 'LONG',
    'SHORT', 'CALL', 'PUT', 'ASK', 'BID', 'AVG', 'EST', 'EPS', 'REV', 
    'SAYS', 'SAID', 'WILL', 'WENT', 'GONE', 'SEEN', 'DONE', 'MADE', 'MAKE',
    'KEEP', 'HELD', 'SOLD', 'BOUGHT', 'PAID', 'OWED', 'LENT', 'SENT', 'TOOK',
    'GAVE', 'GOT', 'MET', 'SUES', 'SUE', 'WON', 'LOST', 'COST', 'VALUE', 
    'PRICE', 'RATE', 'YIELD', 'BOND', 'NOTE', 'BILL', 'CASH', 'GOLD', 'OIL',
    'GAS', 'DATA', 'TECH', 'SOFT', 'HARD', 'FIRM', 'BANK', 'FUND', 'USER',
    'ZERO', 'COST', 'HUNT',
    # HTML/CSS Garbage
    'HTML', 'CSS', 'SRC', 'HREF', 'IMG', 'JPG', 'PNG', 'DIV', 'SPAN', 
    'CLASS', 'STYLE', 'WIDTH', 'HEIGHT', 'MARGIN', 'PADDING', 'FLOAT', 
    'ALT', 'TYPE', 'COM', 'HTTP', 'HTTPS', 'WWW', 'NET', 'ORG', 'GOV',
    # Common Prepositions/Verbs (Short Uppercase risks)
    'THE', 'AND', 'FOR', 'BUT', 'NOT', 'YOU', 'ARE', 'WAS', 'ITS', 'HAS',
    'HAD', 'CAN', 'GET', 'DID', 'WAY', 'TOO', 'USE', 'SEE', 'OWN', 'GOT',
    'MET', 'WON', 'LOST', 'RUN', 'SET', 'PUT', 'SAY', 'LET', 'BIG', 'OLD'
}

CANONICAL_MAP = {
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD", "SOLANA": "SOL-USD",
    "RNDR": "RENDER-USD", "RENDER": "RENDER-USD",
    "BYD": "BYDDF", 
    "BYD COMPANY": "BYDDF"
}

def extract_tickers_from_text(text):
    # Finds 3-8 letter UPPERCASE keys (Avoids IS, AT, TO, MY errors)
    candidates = set(re.findall(r'\b[A-Z0-9]{3,8}\b', text))
    
    # Explicitly look for major Crypto Names (Title Case often used)
    common_names = ["Bitcoin", "Ethereum", "Solana", "Ripple", "Cardano", "Dogecoin", "Polkadot", "Avalanche"]
    for name in common_names:
        if name in text: 
            for k, v in CANONICAL_MAP.items():
                if k.upper() == name.upper() or v.replace('-USD','').upper() == name.upper():
                    if k in CANONICAL_MAP: 
                        candidates.add(k)
                    else:
                        candidates.add(name.upper())
                    break

    found = set()
    for cand in candidates:
        if cand in IGNORE_LIST: continue
        
        # NOISE FILTERS (MATCHING MAIN.PY):
        if cand.isdigit(): continue
        if re.match(r'^\d+[KMBXG]$', cand): continue
        
        found.add(cand)
    return list(found)

def run_smoke_test():
    print("Initializing NewsHunter (Mock Mode)...")
    hunter = NewsHunter()
    
    # We intentionally restrict feeds to avoid waiting forever
    hunter.rss_feeds = [
        "https://finance.yahoo.com/news/rssindex",
        "https://cointelegraph.com/rss"
    ]
    
    print("Fetching LIVE news (Wait 5s)...")
    try:
        news = hunter.fetch_news()
    except Exception as e:
        print(f"Fetch failed (network?): {e}")
        return

    print(f"Fetched {len(news)} items.")
    print("-" * 30)
    
    all_found = []
    
    for item in news[:10]: # Check first 10
        title = item.get('title', '')
        summary = item.get('summary', '')
        # USE RAW TEXT (Maintain Case) like in main.py
        text = f"{title} {summary}"
        
        extracted = extract_tickers_from_text(text)
        if extracted:
            print(f"DEBUG: '{title[:50]}...' -> {extracted}")
            all_found.extend(extracted)
        else:
            # print(f"DEBUG: '{title[:50]}...' -> NO TICKERS")
            pass
            
    print("-" * 30)
    print("Total Unique Candidates Found:", set(all_found))

if __name__ == "__main__":
    run_smoke_test()
