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
        self.session = requests.Session()
        # Enhanced headers to avoid 403 Forbidden
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.reddit.com/",
            "Origin": "https://www.reddit.com",
            "DNT": "1"
        })

    def get_reddit_trending(self) -> Dict[str, int]:
        """
        Scrapes Reddit subreddits via public .json endpoints.
        Returns a map of ticker -> mention_count.
        """
        trending = {}
        for sub in self.SUBREDDITS:
            logger.info(f"🔍 Scraping r/{sub}...")
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json?limit=50"
                resp = self.session.get(url, timeout=10)
                if resp.status_code != 200:
                    # [PHASE C.4] Elevating to ERROR as requested - this is a system failure
                    logger.error(f"❌ Failed to scrape r/{sub}: {resp.status_code} Forbidden (Reddit blocked us)")
                    continue
                
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    title = post_data.get("title", "")
                    content = post_data.get("selftext", "")
                    
                    combined = f"{title} {content}"
                    found_tickers = self.TICKER_PATTERN.findall(combined)
                    
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
