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
        Scrapes Reddit subreddits via public .json endpoints with advanced stealth.
        Falls back to RSS or old.reddit.com if blocked.
        """
        from curl_cffi import requests as c_requests
        import time
        import random
        
        trending = {}
        for sub in self.SUBREDDITS:
            logger.info(f"🔍 Scraping r/{sub}...")
            content = ""
            
            # Randomized jitter to avoid robotic patterns
            time.sleep(random.uniform(1.5, 3.5))
            
            try:
                # Use a session to maintain cookies/state
                with c_requests.Session(impersonate="chrome110") as s:
                    # 1. Try JSON endpoint (Preferred)
                    url = f"https://www.reddit.com/r/{sub}/hot.json?limit=50"
                    
                    headers = {
                        "Accept": "application/json, text/plain, */*",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/",
                        "Sec-Fetch-Dest": "empty",
                        "Sec-Fetch-Mode": "cors",
                        "Sec-Fetch-Site": "same-origin",
                        "DNT": "1"
                    }
                    
                    resp = s.get(url, timeout=15, headers=headers)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        for post in data.get("data", {}).get("children", []):
                            p_data = post.get("data", {})
                            content += f" {p_data.get('title', '')} {p_data.get('selftext', '')}"
                    
                    # 2. Try Fallback if JSON blocked
                    elif resp.status_code in [403, 429]:
                        logger.warning(f"⚠️ JSON blocked ({resp.status_code}) for r/{sub}, trying old.reddit fallback...")
                        
                        # Fallback to old.reddit search or listing (often less protected)
                        alt_url = f"https://old.reddit.com/r/{sub}/"
                        alt_resp = s.get(alt_url, timeout=15, headers={"Referer": "https://www.google.com/"})
                        
                        if alt_resp.status_code == 200:
                            content = alt_resp.text
                        else:
                            # Final fallback: RSS
                            logger.warning(f"⚠️ old.reddit failed ({alt_resp.status_code}), trying RSS...")
                            rss_url = f"https://www.reddit.com/r/{sub}/.rss"
                            rss_resp = s.get(rss_url, timeout=15)
                            if rss_resp.status_code == 200:
                                content = rss_resp.text
                            else:
                                logger.error(f"❌ All Reddit fallbacks failed for r/{sub}: {rss_resp.status_code}")
                    else:
                        logger.error(f"❌ Failed to scrape r/{sub}: {resp.status_code}")
                
                if content:
                    found_tickers = self.TICKER_PATTERN.findall(content)
                    for ticker in set(found_tickers):
                        if ticker not in self.BLACKLIST and ticker.isupper():
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
