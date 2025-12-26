from google import genai
from google.genai import types
import os
import logging
import json
import time

# Configure logging
logger = logging.getLogger(__name__)

class Brain:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set.")
        self.client = genai.Client(api_key=self.api_key)

    def analyze_news_batch(self, news_list):
        """
        Analyze a batch of news items to find high-quality trading opportunities.
        Filters for Trade Republic friendly assets (High Cap/Liquidity).
        """
        if not news_list:
            logger.info("No news to analyze.")
            return []

        # Prepare the prompt
        news_text = "\n\n".join([f"Source: {item['source']}\nTitle: {item['title']}\nSummary: {item['summary']}" for item in news_list])
        
        prompt = f"""
        **SYSTEM ROLE:**
        You are a Senior Investment Analyst & Technical Trader.
        Your goal is to validate market news with Technical Data (RSI, Trend) AND Portfolio Context to issue high-probability signals.

        **CRITICAL FILTERS:**
        1.  **Trade Republic Friendly Only:** Focus ONLY on major High-Cap Stocks (S&P 500, Nasdaq 100, DAX 40) and Major Cryptocurrencies (BTC, ETH, SOL).
        2.  **Ignore:** Penny stocks, low volume altcoins, obscure companies, and general economic noise with no clear actionable ticker.
        3.  **Technical Validation:**
            -   **GOOD NEWS + OVERBOUGHT (RSI > 75):**  Signal "HOLD" or "SELL" (taking profits). Do NOT buy the top.
            -   **BAD NEWS + OVERSOLD (RSI < 30):** Signal "ACCUMULATE" (contrarian play) if the asset is solid.
            -   **GOOD NEWS + UPTREND (Price > SMA200):** Signal "BUY" (Trend Following).
        4.  **Portfolio Context:**
            -   If valid portfolio data is provided (e.g., [Portfolio: Own 10 @ $150]):
            -   **Price < Avg Price:** Suggest "ACCUMULATE/AVERAGE DOWN".
            -   **Price >> Avg Price:** Suggest "HOLD/SELL (PROTECT PROFITS)".
            -   **Heavy Exposure:** If user owns a lot, be more conservative.

        **OUTPUT FORMAT:** JSON list.

        **NEWS DATA:**
        {news_text}

        **INSTRUCTIONS:**
        For each news item that contains a SIGNIFICANT, actionable signal:
        - Extract the **Ticker Symbol** (e.g., AAPL, TSLA, BTC-USD).
        - Assign **Sentiment**: "BUY", "SELL", "ACCUMULATE", "PANIC SELL", "HOLD".
        - Provide a 1-sentence **Prediction/Reasoning** that references News, Technicals AND Portfolio (if applicable).
        - Assign a **Confidence Score** (0.0 to 1.0).

        Return strictly a JSON list of objects. If no valid signals are found, return an empty list [].
        Example JSON Structure:
        [
            {{
                "ticker": "AAPL",
                "sentiment": "BUY",
                "reasoning": "Strong earnings combined with valid technical entry (RSI 45, Above SMA200).",
                "confidence": 0.85,
                "source": "CNBC"
            }}
        ]
        """

        try:
            logger.info("Sending news batch to Gemini...")
            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            # Parse JSON
            try:
                # With the new SDK and response_mime_type, the text is the JSON string
                analysis_results = json.loads(response.text)
                logger.info(f"Gemini returned {len(analysis_results)} potential signals.")
                return analysis_results
            except json.JSONDecodeError:
                logger.error("Failed to parse Gemini response as JSON.")
                logger.debug(f"Raw response: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            return []

    def evaluate_portfolio(self, portfolio, market_news):
        """
        Cross-reference news with existing holdings to suggest actions.
        (Simplified version: This could be expanded to specialized prompts).
        """
        # For now, we rely on the main batch analysis to catch news relevant to holdings 
        # as long as the news source covers them.
        pass

if __name__ == "__main__":
    # Test stub
    b = Brain()
    fake_news = [{"source": "Test", "title": "Apple releases new iPhone", "summary": "Revolutionary AI features included."}]
    print(b.analyze_news_batch(fake_news))
