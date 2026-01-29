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
        Scrapes Reddit subreddits via public .json endpoints with advanced stealth.
        Falls back directly to RSS if blocked, as old.reddit is often also blocked in GH Actions.
        """
        from curl_cffi import requests as c_requests
        import time
        import random
        
        trending = {}
        for sub in self.SUBREDDITS:
            logger.info(f"🔍 Scraping r/{sub}...")
            content = ""
            
            # Randomized jitter to avoid robotic patterns
            time.sleep(random.uniform(0.5, 1.5))
            
            try:
                # Use a session to maintain cookies/state
                with c_requests.Session(impersonate="chrome110") as s:
                    # 1. Try JSON endpoint
                    url = f"https://www.reddit.com/r/{sub}/hot.json?limit=30" # Reduced limit for speed and stealth
                    
                    headers = {
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Cache-Control": "max-age=0",
                        "Sec-Ch-Ua": '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
                        "Sec-Ch-Ua-Mobile": "?0",
                        "Sec-Ch-Ua-Platform": '"Windows"',
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1",
                        "Upgrade-Insecure-Requests": "1"
                    }
                    
                    resp = s.get(url, timeout=10, headers=headers)
                    
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            for post in data.get("data", {}).get("children", []):
                                p_data = post.get("data", {})
                                content += f" {p_data.get('title', '')} {p_data.get('selftext', '')}"
                        except:
                            # Might be blocked by a login-wall or redirect
                            resp.status_code = 403
                    
                    # 2. Optimized Fallback: RSS (Most resilient from Server/Actions IPs)
                    if resp.status_code in [403, 429]:
                        logger.warning(f"⚠️ Reddit blocked ({resp.status_code}) for r/{sub}, bypassing to RSS fallback...")
                        rss_url = f"https://www.reddit.com/r/{sub}/.rss"
                        # RSS usually needs a generic User-Agent, not fully impersonated Chrome
                        rss_resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"})
                        if rss_resp.status_code == 200:
                            content = rss_resp.text
                        else:
                            logger.error(f"❌ RSS fallback also failed for r/{sub}: {rss_resp.status_code}")
                    elif resp.status_code != 200:
                        logger.error(f"❌ Failed to scrape r/{sub}: {resp.status_code}")
                
                if content:
                    found_tickers = self.TICKER_PATTERN.findall(content)
                    for ticker in set(found_tickers):
                        # Filter criteria: length > 1, not in blacklist, all alpha
                        if (len(ticker) > 1 and 
                            ticker not in self.BLACKLIST and 
                            ticker.isupper() and 
                            not any(c.isdigit() for c in ticker)):
                            trending[ticker] = trending.get(ticker, 0) + 1
                            
            except Exception as e:
                logger.error(f"❌ Error scraping r/{sub}: {e}")
                
        # Sort and return top 20
        sorted_trending = dict(sorted(trending.items(), key=lambda x: x[1], reverse=True)[:20])
        logger.info(f"🔥 Top Reddit Tickers: {sorted_trending}")
        return sorted_trending

    def get_social_context(self, ticker: str) -> str:
        """Returns a string representation of the social hype for a ticker with velocity."""
        trending = self.get_reddit_trending()
        count = trending.get(ticker.upper(), 0)
        
        # Calculate Velocity
        velocity_info = self.detect_velocity(ticker.upper(), count)
        velocity_label = f" (Velocity: {velocity_info['status']})" if velocity_info else ""
        
        if count > 5:
            sentiment = "🔥 HIGH HYPE"
        elif count > 2:
            sentiment = "👀 MODERATE INTEREST"
        elif count > 0:
            sentiment = "🔹 MENTIONED"
        else:
            sentiment = "🌑 QUIET"
            
        return f"[SOCIAL ORACLE: {ticker} -> {sentiment} ({count} mentions){velocity_label}]"

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = SocialScraper()
    print(scraper.get_reddit_trending())
