import feedparser
import logging
import ssl

# Configure logging
logger = logging.getLogger(__name__)

# Fix SSL certificate errors for some feeds
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

import trafilatura # Added for Full-Text Scraping
from curl_cffi import requests # 🚀 UPGRADE: TLS Fingerprint Spoofing

# ... imports ...

class NewsHunter:
    def __init__(self):
        self.rss_feeds = [
            # --- GENERAL FINANCE ---
            "https://finance.yahoo.com/news/rssindex",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", # CNBC Finance
            "https://feeds.content.dowjones.io/public/rss/mw_topstories", # MarketWatch
            "https://www.investing.com/rss/news.rss", # Investing.com General
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", # WSJ Markets
            
            # --- TECH & AI ---
            "https://techcrunch.com/feed/",
            "https://venturebeat.com/category/ai/feed/", # VentureBeat AI
            "https://www.artificialintelligence-news.com/feed/", # AI News
            "http://news.mit.edu/rss/topic/artificial-intelligence2", # MIT AI Research
            
            # --- CRYPTO ---
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            
            # --- GREEN ENERGY ---
            "https://www.renewableenergyworld.com/feed/",
            "https://cleantechnica.com/feed/",
        ]

    def _fetch_url_impersonate(self, url, browser_type="chrome120"):
        """Helper to fetch URL with specific browser impersonation."""
        return requests.get(
            url, 
            impersonate=browser_type, 
            timeout=10,
            headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Referer': 'https://www.google.com/',
                'Upgrade-Insecure-Requests': '1'
            }
        )

    def _fetch_full_text(self, url):
        """
        Attempt to scrape full text using multiple strategies:
        1. Direct Chrome Impersonation
        2. Direct Safari Impersonation (Fallback)
        3. Google Cache (Last Resort)
        """
        try:
            # STRATEGY 1: Direct Chrome
            response = self._fetch_url_impersonate(url, "chrome120")
            
            if response.status_code == 200:
                return trafilatura.extract(response.text)
            
            elif response.status_code == 401:
                # 401 is usually a Content Paywall (WSJ, etc). No point retrying.
                logger.info(f"Paywall detected (401) for {url}. Using Summary.")
                return None

            elif response.status_code == 403:
                # STRATEGY 2: Safari Impersonation (Sometimes bypasses Cloudflare better)
                logger.info(f"Chrome blocked (403), retrying as Safari for {url}...")
                response = self._fetch_url_impersonate(url, "safari15_5")
                if response.status_code == 200:
                   logger.info(f"Safari bypass successful for {url}!")
                   return trafilatura.extract(response.text)
            
            # STRATEGY 3: Google Cache Fallback
            if response.status_code in [403, 503]:
                logger.info(f"Direct access blocked. Trying Google Cache for {url}...")
                cache_url = f"http://webcache.googleusercontent.com/search?q=cache:{url}"
                # Cache often needs a clean simple UA, or sometimes the same impersonation
                response = self._fetch_url_impersonate(cache_url, "chrome110")
                if response.status_code == 200:
                     logger.info(f"Google Cache hit for {url}!")
                     # Trafilatura is good at extracting the article from the messy Cache wrapper
                     return trafilatura.extract(response.text)
            
            logger.warning(f"All scrape strategies failed ({response.status_code}) for {url}")
            return None

        except Exception as e:
            logger.warning(f"Scraping error for {url}: {e}")
            return None

    def fetch_news(self):
        """Fetch and parse news from RSS feeds."""
        all_news = []
        for url in self.rss_feeds:
            try:
                logger.info(f"Fetching news from: {url}")
                feed = feedparser.parse(url)
                
                if feed.bozo:
                    logger.warning(f"Feed malformed or error for {url}: {feed.bozo_exception}")
                    continue

                # Limit to top 3 per feed to manage scraping time/limits
                for entry in feed.entries[:3]: 
                    link = entry.get("link", "#")
                    
                    # 🚀 INTELLIGENCE UPGRADE: Fetch Full Body
                    full_text = None
                    if link and link != "#":
                         # Minimal delay to be polite? No, we need speed for serverless.
                         # But we catch exceptions.
                         full_text = self._fetch_full_text(link)
                    
                    summary = entry.get("summary", "") or entry.get("description", "")
                    
                    # Fallback: If scraping fails, use summary.
                    # If scraping works, use scraped text (truncated to 2000 chars for context)
                    final_content = summary
                    if full_text:
                        final_content = f"[FULL TEXT EXTRACTED]\n{full_text[:2500]}..." # 2500 char limit
                        logger.info(f"Successfully scraped content for: {entry.get('title')}")

                    news_item = {
                        "title": entry.get("title", "No Title"),
                        "summary": final_content, # Now contains full text if available
                        "link": link,
                        "published": entry.get("published", "Unknown Date"),
                        "source": feed.feed.get("title", "Unknown Source")
                    }
                    all_news.append(news_item)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")

        logger.info(f"Fetched {len(all_news)} news items.")
        return all_news

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hunter = NewsHunter()
    news = hunter.fetch_news()
    for n in news[:3]:
        print(n)
