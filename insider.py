
import requests
import logging

logger = logging.getLogger("Insider")

class Insider:
    def __init__(self):
        self.crypto_api_url = "https://api.alternative.me/fng/"

    def get_crypto_fear_greed(self):
        """
        Fetches the Crypto Fear & Greed Index from alternative.me.
        Returns: { "value": 50, "classification": "Neutral" }
        """
        try:
            response = requests.get(self.crypto_api_url, timeout=10)
            data = response.json()
            if data and 'data' in data:
                item = data['data'][0]
                return {
                    "value": int(item['value']),
                    "classification": item['value_classification']
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching Crypto F&G: {e}")
            return None

    def get_stock_fear_greed(self):
        """
        Fetches the CNN Fear & Greed Index (Stock Market).
        Uses the specialized data visualization endpoint.
        """
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        try:
            # User-Agent is critical here
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            r = requests.get(url, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            
            # The API returns 'fear_and_greed_historical'
            # We want the latest data point
            
            if 'fear_and_greed_historical' in data:
                latest = data['fear_and_greed_historical']['data'][-1]
                score = round(latest['y'])
                rating = latest['rating'].upper() # e.g. "EXTREME FEAR"
                
                logger.info(f"Insider: Stock Fear & Greed is {rating} ({score})")
                return {
                    "value": score,
                    "classification": rating
                }
                
        except Exception as e:
            logger.warning(f"Failed to fetch Stock Fear & Greed: {e}")
        
        return None

    def get_market_mood(self):
        """
        Aggregates market sentiment from multiple sources (Crypto F&G, Stock F&G).
        """
        crypto_fg = self.get_crypto_fear_greed()
        stock_fg = self.get_stock_fear_greed()
        
        mood = {
            "crypto": crypto_fg,
            "stock": stock_fg,
            "overall": "NEUTRAL"
        }
        
        # Determine overall mood priority: Extreme > Normal
        # If Crypto is Extreme Fear, that dominates.
        if crypto_fg:
            mood['overall'] = crypto_fg.get('classification', 'NEUTRAL')
            
        if stock_fg:
            # If stock is more extreme, let it influence (simple logic for now)
            pass
            
        return mood

    def get_social_sentiment(self):
        """
        Fetches trending topics from Reddit (r/stocks, r/bitcoin, r/investing).
        Returns a list of top 5 combined hot headlines.
        """
        import feedparser
        import random
        import requests
        
        feeds = [
            "https://www.reddit.com/r/stocks/.rss",
            "https://www.reddit.com/r/bitcoin/.rss",
            "https://www.reddit.com/r/investing/.rss"
        ]
        
        headlines = []
        # Reddit requires unique UA
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        
        for url in feeds:
            try:
                # 1. Fetch raw XML with requests (better UA handling)
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code != 200:
                    logger.warning(f"Reddit RSS blocked: {resp.status_code}")
                    continue

                # 2. Parse string
                f = feedparser.parse(resp.content)
                
                if f.entries:
                    # Take top 3 from each
                    for e in f.entries[:3]:
                        # Clean title
                        title = e.title
                        headlines.append(f"Reddit ({f.feed.title}): {title}")
            except Exception as e:
                logger.warning(f"Failed to fetch RSS {url}: {e}")
                
        # Shuffle and return top 7 to avoid clutter
        random.shuffle(headlines)
        return headlines[:7]

if __name__ == "__main__":
    insider = Insider()
    print("Market Mood:", insider.get_market_mood())
    print("Social Sentiment:", insider.get_social_sentiment())
