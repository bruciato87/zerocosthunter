import feedparser
import logging
import ssl
from typing import Dict

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
            
            # --- CRYPTO (Extended for Altcoins) ---
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://www.theblock.co/rss.xml",  # The Block - Deep crypto coverage
            "https://cryptoslate.com/feed/",    # CryptoSlate - Altcoin coverage
            "https://cryptopotato.com/feed/",   # CryptoPotato - Altcoin news
            "https://u.today/rss",              # U.Today - Altcoin focused
            "https://bitcoinist.com/feed/",    # Bitcoinist - Crypto news
            "https://beincrypto.com/feed/",    # BeInCrypto - DeFi/Altcoins
            "https://www.newsbtc.com/feed/",   # NewsBTC - Altcoin analysis
            "https://dailycoin.com/feed/",     # DailyCoin - Crypto news
            "https://www.fxstreet.com/cryptocurrencies/feed",  # FXStreet Crypto - Price forecasts
            "https://ambcrypto.com/feed/",     # AMBCrypto - Altcoin analysis
            "https://cryptonews.com/news/feed/", # CryptoNews - General crypto
            
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
        0. Check cache first (2h TTL)
        1. Direct Chrome Impersonation
        2. Direct Safari Impersonation (Fallback)
        3. Google Cache (Last Resort)
        """
        try:
            # STRATEGY 0: Check cache first
            from db_handler import DBHandler
            db = DBHandler()
            cached_content, is_cached = db.get_cached_news(url, ttl_hours=2)
            if is_cached and cached_content:
                return cached_content
            
            # STRATEGY 1: Direct Chrome
            response = self._fetch_url_impersonate(url, "chrome120")
            
            if response.status_code == 200:
                content = trafilatura.extract(response.text)
                if content:
                    db.save_news_cache(url, content)  # Save to cache
                return content
            
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
                   content = trafilatura.extract(response.text)
                   if content:
                       db.save_news_cache(url, content)  # Save to cache
                   return content
            
            # STRATEGY 3: Google Cache Fallback
            if response.status_code in [403, 503]:
                logger.info(f"Direct access blocked. Trying Google Cache for {url}...")
                cache_url = f"http://webcache.googleusercontent.com/search?q=cache:{url}"
                # Cache often needs a clean simple UA, or sometimes the same impersonation
                response = self._fetch_url_impersonate(cache_url, "chrome110")
                if response.status_code == 200:
                     logger.info(f"Google Cache hit for {url}!")
                     content = trafilatura.extract(response.text)
                     if content:
                         db.save_news_cache(url, content)  # Save to cache
                     return content
            
            logger.warning(f"All scrape strategies failed ({response.status_code}) for {url}")
            return None

        except Exception as e:
            logger.warning(f"Scraping error for {url}: {e}")
            return None

    def fetch_news(self):
        """
        Fetch and parse news from RSS feeds.
        IMPROVEMENT 1: Parallel Fetching (Speedup)
        IMPROVEMENT 2: Caching (Avoid redundant fetches)
        """
        all_news = []
        
        # Check cache (Simple in-memory cache for concurrent runs, or DB for long-term if needed)
        # For serverless/CLI, minimal impact, but parallel is key.
        
        # Helper for processing a single feed URL
        def process_feed(url):
            feed_items = []
            try:
                # logger.info(f"Fetching news from: {url}") # Too spammy for parallel
                feed = feedparser.parse(url)
                
                if feed.bozo:
                    if not feed.entries:
                        logger.warning(f"Feed malformed or error for {url}: {feed.bozo_exception}")
                        return []
                    else:
                        logger.debug(f"Feed had minor issues (parsed {len(feed.entries)} items): {url} - {feed.bozo_exception}")

                # Limit to top 3 per feed
                for entry in feed.entries[:3]: 
                    link = entry.get("link", "#")
                    
                    # 🚀 INTELLIGENCE UPGRADE: Fetch Full Body
                    full_text = None
                    if link and link != "#":
                         full_text = self._fetch_full_text(link)
                    
                    summary = entry.get("summary", "") or entry.get("description", "")
                    
                    final_content = summary
                    if full_text:
                        final_content = f"[FULL TEXT EXTRACTED]\n{full_text[:2500]}..." 
                        # logger.info(f"Successfully scraped: {entry.get('title')}")

                    # IMPROVEMENT 5: Breaking News Check (Simple timestamp check)
                    published_str = entry.get("published", "")
                    # (Parsing date is complex without dateutil, skip complex logic for now, raw string ok)

                    news_item = {
                        "title": entry.get("title", "No Title"),
                        "summary": final_content,
                        "link": link,
                        "published": published_str,
                        "source": feed.feed.get("title", "Unknown Source")
                    }
                    feed_items.append(news_item)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
            return feed_items

        # Execute in Parallel (Reduced workers to prevent curl_cffi segfault on GitHub Actions)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(process_feed, url): url for url in self.rss_feeds}
            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    data = future.result()
                    all_news.extend(data)
                except Exception as exc:
                    logger.error(f"Feed generated an exception: {exc}")

        logger.info(f"Fetched {len(all_news)} news items (Parallel Mode).")
        return all_news

    def fetch_breaking_news(self, lookback_minutes=60, limit=5):
        """
        FAST LANE: Fetches only from high-frequency sources and filters for very recent items.
        Used for real-time monitoring loops.
        """
        fast_feeds = [
            "https://feeds.content.dowjones.io/public/rss/mw_topstories", # MarketWatch
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", # CNBC
            "https://www.coindesk.com/arc/outboundfeeds/rss/", # CoinDesk
            "https://www.theblock.co/rss.xml", # The Block
            "https://cointelegraph.com/rss" # CoinTelegraph
        ]
        
        breaking_news = []
        import time
        from datetime import datetime, timedelta
        
        # Calculate cutoff time (UTC)
        cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        
        def process_fast_feed(url):
            items = []
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]: # Top 5 only
                    # Parse time
                    published_time = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_time = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    
                    # If time parsing worked and it's fresh enough
                    if published_time:
                        if published_time >= cutoff_time:
                            items.append({
                                "title": entry.get("title", ""),
                                "link": entry.get("link", ""),
                                "summary": entry.get("summary", "")[:500],
                                "source": feed.feed.get("title", "Unknown"),
                                "published": entry.get("published", ""),
                                "is_breaking": True,
                                "age_minutes": int((datetime.utcnow() - published_time).total_seconds() / 60)
                            })
                    else:
                        # If no time, assume it's fresh if it's the very first item? No, unsafe.
                        # Some feeds don't parse well. Skip or include with warning?
                        pass 
            except Exception as e:
                logger.warning(f"Fast feed error {url}: {e}")
            return items

        # Parallel Fetch
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(process_fast_feed, url): url for url in fast_feeds}
            for future in concurrent.futures.as_completed(future_to_url):
                breaking_news.extend(future.result())
        
        # Sort by age (newest first)
        breaking_news.sort(key=lambda x: x.get('age_minutes', 999))
        
        logger.info(f"Breaking News: Found {len(breaking_news)} items < {lookback_minutes} mins old.")
        return breaking_news[:limit]

    def fetch_ticker_news(self, ticker: str, limit: int = 3):
        """
        Fetch specific news for a Single Ticker (for /analyze command).
        Uses yfinance news feed + Full Text Scraping.
        """
        import yfinance as yf
        logger.info(f"Fetching deep-dive news for {ticker}...")
        
        try:
            # 1. AUTO-FIX: Known Cryptos and Aliases
            known_crypto = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "LINK", "AVAX"]
            
            # Ticker aliases for news search (different names in news vs ticker)
            # YF uses different tickers than CoinGecko/Trade Republic
            NEWS_ALIASES = {
                "RENDER": ["RNDR", "RENDER-USD", "RNDR-USD"],
                "RENDER-USD": ["RNDR", "RENDER", "RNDR-USD"],
                "RNDR": ["RENDER", "RENDER-USD", "RNDR-USD"],
                # Add more common aliases
                "SOL": ["SOL-USD", "SOLANA"],
                "ETH": ["ETH-USD", "ETHEREUM"],
                "BTC": ["BTC-USD", "BITCOIN"],
                "XRP": ["XRP-USD", "RIPPLE"],
            }
            
            original_ticker = ticker
            
            # Use ticker_resolver for initial resolution
            from ticker_resolver import resolve_ticker
            ticker = resolve_ticker(ticker)
            
            t = yf.Ticker(ticker)
            yf_news = t.news
            
            # 2. FALLBACK: Try aliases if no news found
            if not yf_news and original_ticker.upper() in NEWS_ALIASES:
                for alt_ticker in NEWS_ALIASES[original_ticker.upper()]:
                    logger.info(f"No news for {original_ticker}, trying {alt_ticker}...")
                    t_retry = yf.Ticker(alt_ticker)
                    yf_news = t_retry.news
                    if yf_news:
                        break
            
            # 3. FALLBACK: If still no news and no suffix, try adding -USD
            if not yf_news and '-' not in original_ticker:
                 logger.info(f"No news for {original_ticker}, trying {original_ticker}-USD...")
                 t_retry = yf.Ticker(f"{original_ticker}-USD")
                 yf_news = t_retry.news
            
            processed_news = []
            
            # yfinance news format is list of dicts:
            # {'uuid': '...', 'title': '...', 'publisher': '...', 'link': '...', 'providerPublishTime': ...}
            
            for item in yf_news[:limit]:
                link = item.get('link')
                title = item.get('title')
                
                # Scrape Full Text
                full_text = None
                if link:
                    full_text = self._fetch_full_text(link)
                
                final_content = full_text if full_text else title # Fallback
                
                if full_text:
                     final_content = f"[FULL TEXT EXTRACTED]\n{full_text[:3000]}..."

                processed_news.append({
                    "source": item.get('publisher', 'Yahoo Finance'),
                    "title": title,
                    "summary": final_content,
                    "link": link
                })
            
            # 4. FALLBACK: If still no news, search RSS feeds (same as /hunt)
            if not processed_news:
                logger.info(f"YF has no news for {original_ticker}, searching RSS feeds...")
                search_terms = [original_ticker.upper()]
                # Add common name variants
                if original_ticker.upper() in ["RENDER", "RENDER-USD", "RNDR"]:
                    search_terms.extend(["RENDER", "RNDR", "RENDER NETWORK"])
                
                import feedparser
                for feed_url in self.rss_feeds[:14]:  # Check all 14 crypto feeds
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries[:5]:
                            title = entry.get('title', '')
                            summary = entry.get('summary', entry.get('description', ''))
                            combined = f"{title} {summary}".upper()
                            
                            # Check if any search term is mentioned
                            if any(term in combined for term in search_terms):
                                processed_news.append({
                                    "source": feed.feed.get('title', feed_url),
                                    "title": title,
                                    "summary": summary[:1000],
                                    "link": entry.get('link', '')
                                })
                                if len(processed_news) >= limit:
                                    break
                    except Exception as e:
                        logger.warning(f"RSS feed error for {feed_url}: {e}")
                    
                    if len(processed_news) >= limit:
                        break
                
                if processed_news:
                    logger.info(f"Found {len(processed_news)} news from RSS feeds for {original_ticker}")
            
            # 5. FINAL FALLBACK: Google News RSS (aggregates all sources)
            if not processed_news:
                logger.info(f"RSS feeds have no news, trying Google News for {original_ticker}...")
                import feedparser
                import urllib.parse
                
                # Build search query based on asset type
                if original_ticker.upper() in ["RENDER", "RENDER-USD", "RNDR"]:
                    query = "RNDR OR Render Network crypto"
                elif "-USD" in original_ticker.upper():
                    base = original_ticker.replace("-USD", "")
                    query = f"{base} cryptocurrency"
                else:
                    query = f"{original_ticker} stock OR crypto"
                
                encoded_query = urllib.parse.quote(query)
                google_news_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                
                try:
                    feed = feedparser.parse(google_news_url)
                    for entry in feed.entries[:limit]:
                        title = entry.get('title', '')
                        # Google News summary often contains source info, extract it
                        source = entry.get('source', {}).get('title', 'Google News')
                        link = entry.get('link', '')
                        
                        processed_news.append({
                            "source": source,
                            "title": title,
                            "summary": title,  # Google News doesn't provide full summary in RSS
                            "link": link
                        })
                    
                    if processed_news:
                        logger.info(f"Found {len(processed_news)} news from Google News for {original_ticker}")
                except Exception as e:
                    logger.warning(f"Google News error for {original_ticker}: {e}")
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching ticker news for {ticker}: {e}")
            return []

    def check_owned_asset_news(self, ticker: str) -> Dict:
        """
        Check for negative news on owned assets.
        Returns a dict with sentiment details.
        """
        logger.info(f"Watchdog: Checking news sentiment for {ticker}...")
        news = self.fetch_ticker_news(ticker, limit=5)
        
        if not news:
            return {"sentiment": 0, "is_negative": False, "summary": "No recent news found."}
            
        # Basic keyword-based negative sentiment detection
        negative_keywords = [
            "CRASH", "DROP", "FALL", "DOWN", "REJECTED", "LAWSUIT", "SEC", 
            "REGULATION", "HACK", "EXPLOIT", "SCAM", "BEARISH", "DUMP", 
            "BANKRUPTCY", "LIQUIDATION", "INVESTIGATION", "FRAUD", "LEAK"
        ]
        
        neg_count = 0
        news_titles = []
        for item in news:
            title = (item.get('title') or '').upper()
            news_titles.append(item.get('title') or '')
            if any(word in title for word in negative_keywords):
                neg_count += 1
                
        sentiment = - (neg_count / len(news)) if news else 0
        
        return {
            "sentiment": sentiment,
            "is_negative": sentiment < -0.3,
            "summary": " | ".join(news_titles[:2]),
            "count": len(news)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hunter = NewsHunter()
    news = hunter.fetch_news()
    for n in news[:3]:
        print(n)
