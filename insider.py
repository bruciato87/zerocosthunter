
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
        Aggregates social sentiment using the advanced SocialScraper.
        Identifies trending tickers with their Social Velocity.
        """
        try:
            from social_scraper import SocialScraper
            scraper = SocialScraper()
            
            # 1. Get current trending tickers
            trending_dict = scraper.get_reddit_trending()
            
            headlines = []
            for ticker, mentions in trending_dict.items():
                # 2. Add Velocity context for the Hype Oracle
                velocity_info = scraper.detect_velocity(ticker, mentions)
                status = velocity_info['status'] if velocity_info else "STABLE"
                
                # Format as a high-signal headline
                headlines.append(f"🔥 SOCIAL SURGE: {ticker} mentionato {mentions} volte (Velocity: {status})")
                
            if not headlines:
                # Minimal fallback if Reddit is totally blocked
                headlines.append("Social sentiment: Quiet (No significant surges detected)")
                
            return headlines[:10]
            
        except Exception as e:
            logger.warning(f"Social Scraper integration failed: {e}")
            return ["Social sentiment survey failed."]

if __name__ == "__main__":
    insider = Insider()
    print("Market Mood:", insider.get_market_mood())
    print("Social Sentiment:", insider.get_social_sentiment())
