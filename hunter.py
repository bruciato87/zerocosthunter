import feedparser
import logging
import ssl

# Configure logging
logger = logging.getLogger(__name__)

# Fix SSL certificate errors for some feeds
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

import trafilatura # Added for Full-Text Scraping

# ... imports ...

class NewsHunter:
    def __init__(self):
        # ... rss_feeds list ...
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

    def _fetch_full_text(self, url):
        """
        Download and strict the main text from a news URL.
        Returns a simplified string or None.
        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                return text
            return None
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
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
