import requests
import re
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional

logger = logging.getLogger("SocialScraper")

class SocialScraper:
    """
    The Oracle: Social Intelligence Module.
    Scrapes public social data to detect early ticker hype.
    """
    
    SUBREDDITS = ["wallstreetbets", "cryptocurrency", "stocks", "pennystocks"]
    TICKER_PATTERN = re.compile(r'\b[A-Z]{2,5}\b')
    # Filter out common words and noise that look like tickers
    BLACKLIST = {
        "THE", "AND", "ARE", "FOR", "NOT", "BUT", "HAS", "ANY", "ALL", "NEW", "NOW", "ONE",
        "CEO", "IPO", "ETF", "SEC", "FED", "CPI", "USA", "EMA", "RSI", "MACD", "ATH", "ATL",
        "GDP", "PMI", "STK", "OPT", "CALL", "PUT", "BUY", "SELL", "HOLD", "DD", "MOON", "LAMBO",
        "HODL", "DYOR", "FOMO", "FUD", "IYKYK", "LFG", "NFA", "WAGMI", "YOLO", "BTW", "FYI", "IMO",
        "IMHO", "TLDR", "WTF", "LOL", "AFK", "BRB", "GG", "GL", "HF", "IDK", "IKR", "NP", "OMG",
        "PE", "EPS", "FY", "Q1", "Q2", "Q3", "Q4", "AI", "EU", "UAE", "UK", "US", "UTF", "FT", "MP", 
        "DAP", "TYO", "DBS", "BULL", "BEAR", "PUMP", "DUMP", "DEX", "CEX", "NFT", "TA", "FA", 
        "ATH", "FOMO", "SAUCE", "BAGS", "WHALE", "ALPHA", "BETA", "CASH", "GOLD", "SLV", "GLD",
        "IP", "AT", "ER", "IN", "UP", "GMT", "UTC", "AMA", "DCA", "JUST", "RATES", "INFO", "PLAY", "REAL", "POST"
    }

    def __init__(self):
        # We'll use curl_cffi's requests directly in the method for better impersonation
        pass

    def get_reddit_trending(self) -> Dict[str, int]:
        """
        [PHASE 13 REFINED]
        Multi-source Social intelligence:
        1. CoinGecko Trending (Crypto)
        2. Google News RSS for "Reddit Stock Trends" (Meta-Scraper for Stocks)
        3. Reddit direct (Stealth/Silent fallback)
        """
        from curl_cffi import requests as c_requests
        import time
        import random
        
        trending = {}

        # --- SOURCE 1: CoinGecko Trending (Reliable Crypto) ---
        try:
            cg_url = "https://api.coingecko.com/api/v3/search/trending"
            resp = requests.get(cg_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for coin in data.get("coins", []):
                    item = coin.get("item", {})
                    symbol = item.get("symbol", "").upper()
                    if symbol and len(symbol) > 1 and symbol not in self.BLACKLIST:
                        trending[symbol] = trending.get(symbol, 0) + 5 # Weight Coingecko high (Top 7 only)
        except Exception as e:
            logger.debug(f"Social: CoinGecko Trending failed: {e}")

        # --- SOURCE 2: Google News Meta-Scraper (Resilient Stocks/Reddit) ---
        try:
            # Query aimed at finding trending stocks discussed on Reddit
            queries = ["wallstreetbets+trending+stocks", "reddit+stocks+hot"]
            for q in queries:
                gn_url = f"https://news.google.com/rss/search?q={q}"
                resp = requests.get(gn_url, timeout=10)
                if resp.status_code == 200:
                    content = resp.text
                    found = self.TICKER_PATTERN.findall(content)
                    for ticker in set(found):
                        if (ticker not in self.BLACKLIST and 
                            ticker.isupper() and 
                            len(ticker) > 1 and
                            not any(c.isdigit() for c in ticker)):
                            trending[ticker] = trending.get(ticker, 0) + 2
        except Exception as e:
            logger.debug(f"Social: Google News fallback failed: {e}")

        # --- SOURCE 3: Silent Reddit Scraping (Stealth) ---
        for sub in self.SUBREDDITS:
            try:
                # Use a session to maintain cookies/state
                with c_requests.Session(impersonate="chrome110") as s:
                    url = f"https://www.reddit.com/r/{sub}/hot.json?limit=15"
                    headers = {
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Cache-Control": "max-age=0",
                        "Upgrade-Insecure-Requests": "1"
                    }
                    resp = s.get(url, timeout=10, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        for post in data.get("data", {}).get("children", []):
                            p_data = post.get("data", {})
                            content = f"{p_data.get('title', '')} {p_data.get('selftext', '')}"
                            found = self.TICKER_PATTERN.findall(content)
                            for ticker in set(found):
                                if (ticker not in self.BLACKLIST and 
                                    ticker.isupper() and 
                                    len(ticker) > 1 and 
                                    not any(c.isdigit() for c in ticker)):
                                    trending[ticker] = trending.get(ticker, 0) + 1
                    else:
                        # Silent failure for 403 (expected in Actions)
                        logger.debug(f"Social: Reddit r/{sub} blocked ({resp.status_code})")
            except:
                pass # Fail silently
                
        # Sort and return top 20
        sorted_trending = dict(sorted(trending.items(), key=lambda x: x[1], reverse=True)[:20])
        logger.info(f"Consolidated Social Tickers: {sorted_trending}")
        return sorted_trending

    def get_social_context(self, ticker: str) -> str:
        """Returns a string representation of the social hype for a ticker with velocity."""
        trending = self.get_reddit_trending()
        count = trending.get(ticker.upper(), 0)
        
        # Calculate Velocity
        velocity_info = self.detect_velocity(ticker.upper(), count)
        velocity_label = f" (Velocity: {velocity_info['status']})" if velocity_info else ""
        
        if count > 8: # Higher threshold now that sources are merged
            sentiment = "HIGH HYPE"
        elif count > 3:
            sentiment = "MODERATE INTEREST"
        elif count > 0:
            sentiment = "MENTIONED"
        else:
            sentiment = "QUIET"
            
        return f"[SOCIAL ORACLE: {ticker} -> {sentiment} ({count} influence score){velocity_label}]"

    def detect_velocity(self, ticker: str, current_count: int) -> Optional[Dict]:
        """Detect if mentions are surging compared to historical average."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Save current for future
            db.log_social_mentions(ticker, current_count)
            
            # Fetch last 12 hours
            history = db.get_social_history(ticker, hours=12)
            if not history or len(history) < 2:
                return {"status": "STABLE", "growth": 0}
            
            # Simple growth calculation: (current / average_of_last_3)
            # Take up to last 3 entries excluding the one we just added (history[0] is the newest)
            past_counts = [h['mentions'] for h in history[1:4]]
            if not past_counts:
                return {"status": "STABLE", "growth": 0}
                
            avg_past = sum(past_counts) / len(past_counts)
            
            if avg_past == 0:
                growth = current_count * 100 if current_count > 0 else 0
            else:
                growth = ((current_count - avg_past) / avg_past) * 100
            
            if growth > 150:
                status = "🚀 SURGING"
            elif growth > 50:
                status = "📈 GROWING"
            elif growth < -50:
                status = "📉 COOLING"
            else:
                status = "STABLE"
                
            return {"status": status, "growth": round(growth, 1)}
            
        except Exception as e:
            logger.warning(f"Error detecting velocity for {ticker}: {e}")
            return None

    def get_hype_score(self, ticker: str) -> float:
        """
        [Phase 2] Get a numeric hype score for ML feature extraction.
        
        Returns:
            float: 0-10 score where:
                0-2: Low/quiet (minimal mentions)
                3-5: Moderate interest
                6-8: High hype
                9-10: Viral/surging
        """
        try:
            trending = self.get_reddit_trending()
            count = trending.get(ticker.upper(), 0)
            
            # Get velocity info
            velocity_info = self.detect_velocity(ticker.upper(), count)
            velocity_multiplier = 1.0
            if velocity_info:
                status = velocity_info.get("status", "STABLE")
                if "SURGING" in status:
                    velocity_multiplier = 1.5
                elif "GROWING" in status:
                    velocity_multiplier = 1.2
                elif "COOLING" in status:
                    velocity_multiplier = 0.7
            
            # Base score from raw count (0-8 scale)
            if count >= 10:
                base_score = 8.0
            elif count >= 5:
                base_score = 6.0
            elif count >= 3:
                base_score = 4.0
            elif count >= 1:
                base_score = 2.0
            else:
                base_score = 0.0
            
            # Apply velocity modifier
            final_score = min(10.0, base_score * velocity_multiplier)
            
            logger.debug(f"Hype score for {ticker}: {final_score:.1f} (count={count}, velocity={velocity_multiplier}x)")
            return round(final_score, 1)
            
        except Exception as e:
            logger.warning(f"Error getting hype score for {ticker}: {e}")
            return 0.0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = SocialScraper()
    print(scraper.get_reddit_trending())
