
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
        [PLACEHOLDER] Fetches Stock Fear & Greed Index.
        Currently returns a static value until a reliable scraping method is implemented.
        """
        # CNN Fear & Greed is hard to scrape without Selenium/Browser.
        # For V1, we will skip or mock. Let's return None to indicate no data.
        return None

    def get_market_mood(self):
        """
        Returns a simplified mood string: "EXTREME FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME GREED".
        Prioritizes Crypto for now.
        """
        crypto = self.get_crypto_fear_greed()
        if crypto:
            return {
                "crypto": crypto,
                "stock": None,
                "overall": crypto['classification'].upper()
            }
        return None

if __name__ == "__main__":
    insider = Insider()
    print(insider.get_market_mood())
