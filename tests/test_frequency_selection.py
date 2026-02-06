
from collections import Counter

# MOCK Data
MAX_NEW_DISCOVERIES = 5
portfolio_map = {"BTC-USD": {}, "ETH-USD": {}, "NVDA": {}}

news_items = [
    {"title": "PLTR soars", "summary": "PLTR is great"},
    {"title": "PLTR again", "summary": "PLTR moving up"},
    {"title": "COIN news", "summary": "COIN is cool"},
    {"title": "COIN updates", "summary": "COIN regulatory things"},
    {"title": "COIN jumping", "summary": "COIN COIN COIN"},
    {"title": "Random ticker XYZ", "summary": "XYZ exists"},
    {"title": "Another random ABC", "summary": "ABC exists"},
    {"title": "Bitcoin is king", "summary": "BTC hitting new highs"},
    {"title": "Apple releases thing", "summary": "AAPL new product"},
]

def extract_tickers_from_text(text):
    # Simple mock extraction
    import re
    return re.findall(r'\b(PLTR|COIN|XYZ|ABC|BTC|AAPL)\b', text)

CANONICAL_MAP = {"BTC": "BTC-USD"}

def run_simulation():
    unique_tickers = set()
    discovery_counter = Counter()
    portfolio_found = set()
    
    # First Pass
    for item in news_items:
        text_content = (item.get('title', '') + " " + item.get('summary', '')).upper()
        extracted = extract_tickers_from_text(text_content)
        
        for t in extracted:
            norm_ticker = CANONICAL_MAP.get(t, t)
            
            if f"{norm_ticker}-USD" in portfolio_map:
                norm_ticker = f"{norm_ticker}-USD"
            
            if norm_ticker in portfolio_map:
                portfolio_found.add(norm_ticker)
            else:
                discovery_counter[norm_ticker] += 1

    # Step 2: Prioritize Selection
    print("Portfolio Found (Always Add):", portfolio_found)
    for p_ticker in portfolio_found:
        unique_tickers.add(p_ticker)
        
    # Select Top N New Discoveries by Frequency
    if discovery_counter:
        top_discoveries = discovery_counter.most_common(MAX_NEW_DISCOVERIES)
        print(f"Top 5 Candidates by Frequency: {top_discoveries}")
        
        for ticker, count in top_discoveries:
            unique_tickers.add(ticker)
    
    print("Final Unique Tickers to Hunt:", unique_tickers)

if __name__ == "__main__":
    run_simulation()
