import requests
import re
import logging
from datetime import datetime
from typing import Dict, List, Set

logger = logging.getLogger("SocialScraper")

class SocialScraper:
    """
    The Oracle: Social Intelligence Module.
    Scrapes public social data to detect early ticker hype.
    """
    
    SUBREDDITS = ["wallstreetbets", "cryptocurrency", "stocks", "pennystocks"]
    TICKER_PATTERN = re.compile(r'\b[A-Z]{2,5}\b')
    # Filter out common words that look like tickers
    BLACKLIST = {"THE", "AND", "ARE", "FOR", "NOT", "BUT", "HAS", "ANY", "ALL", "NEW", "NOW", "ONE"}

    def __init__(self):
        # We'll use curl_cffi's requests directly in the method for better impersonation
        pass

    def get_reddit_trending(self) -> Dict[str, int]:
        """
        Scrapes Reddit subreddits via public .json endpoints or .rss fallback.
        Returns a map of ticker -> mention_count.
        """
        from curl_cffi import requests as c_requests
        import time
        import random
        
        trending = {}
        for sub in self.SUBREDDITS:
            logger.info(f"🔍 Scraping r/{sub}...")
            content = ""
            
            # Simple jitter to avoid robotic pattern
            time.sleep(random.uniform(1.0, 3.0))
            
            try:
                # 1. Try JSON endpoint (Preferred)
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit=50"
                # Use impersonate to mimic a real browser TLS fingerprint
                resp = c_requests.get(
                    url, 
                    timeout=15, 
                    impersonate="chrome",
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-User": "?1"
                    }
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    for post in data.get("data", {}).get("children", []):
                        p_data = post.get("data", {})
                        content += f" {p_data.get('title', '')} {p_data.get('selftext', '')}"
                
                # 2. Try RSS Fallback if JSON blocked or empty
                elif resp.status_code == 403 or resp.status_code == 429:
                    logger.warning(f"⚠️ JSON blocked (403/429) for r/{sub}, trying RSS fallback...")
                    rss_url = f"https://www.reddit.com/r/{sub}/.rss"
                    rss_resp = c_requests.get(rss_url, timeout=10, impersonate="chrome")
                    if rss_resp.status_code == 200:
                        content = rss_resp.text
                    else:
                        logger.error(f"❌ RSS also failed for r/{sub}: {rss_resp.status_code}")
                else:
                    logger.error(f"❌ Failed to scrape r/{sub}: {resp.status_code}")
                
                if content:
                    found_tickers = self.TICKER_PATTERN.findall(content)
                    for ticker in set(found_tickers):
                        if ticker not in self.BLACKLIST:
                            trending[ticker] = trending.get(ticker, 0) + 1
                            
            except Exception as e:
                logger.error(f"❌ Error scraping r/{sub}: {e}")
                
        # Sort and return top 20
        sorted_trending = dict(sorted(trending.items(), key=lambda x: x[1], reverse=True)[:20])
        logger.info(f"🔥 Top Reddit Tickers: {sorted_trending}")
        return sorted_trending

    def get_social_context(self, ticker: str) -> str:
        """Returns a string representation of the social hype for a ticker."""
        # This will eventually combine Reddit + X + others
        trending = self.get_reddit_trending()
        count = trending.get(ticker.upper(), 0)
        
        if count > 5:
            sentiment = "🔥 HIGH HYPE"
        elif count > 2:
            sentiment = "👀 MODERATE INTEREST"
        elif count > 0:
            sentiment = "🔹 MENTIONED"
        else:
            sentiment = "🌑 QUIET"
            
        return f"[SOCIAL ORACLE: {ticker} -> {sentiment} ({count} mentions)]"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = SocialScraper()
    print(scraper.get_reddit_trending())
