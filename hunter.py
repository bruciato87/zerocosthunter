import feedparser
import logging
import ssl

# Configure logging
logger = logging.getLogger(__name__)

# Fix SSL certificate errors for some feeds
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

class NewsHunter:
    def __init__(self):
        self.rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", # CNBC Finance
            "https://finance.yahoo.com/news/rssindex",
            "https://feeds.content.dowjones.io/public/rss/mw_topstories", # MarketWatch
        ]

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

                for entry in feed.entries[:5]: # Top 5 from each feed
                    news_item = {
                        "title": entry.get("title", "No Title"),
                        "summary": entry.get("summary", "") or entry.get("description", ""),
                        "link": entry.get("link", "#"),
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
