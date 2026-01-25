import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_handler import DBHandler
from ticker_resolver import PROTECTED_TICKERS

db = DBHandler()

print(f"Starting reset for {len(PROTECTED_TICKERS)} protected assets...")

for ticker in PROTECTED_TICKERS:
    try:
        # Check if exists and reset
        res = db.supabase.table("ticker_cache") \
            .update({"fail_count": 0}) \
            .eq("user_ticker", ticker.upper()) \
            .execute()
        
        if res.data:
            print(f"✓ Reset successful for {ticker}")
        else:
            # print(f"ℹ {ticker} not found in cache, skipping reset.")
            pass
            
    except Exception as e:
        print(f"✗ Failed to reset {ticker}: {e}")

print("Reset complete.")
