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

    def parse_portfolio_from_image(self, image_path):
        """
        Uses Gemini Vision to extract structured portfolio data from a screenshot.
        """
        logger.info(f"Parsing portfolio image: {image_path}...")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Determine mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """
        **SYSTEM ROLE:**
        You are a Financial Data Extraction Assistant.
        Your goal is to extract financial holdings from the provided SCREENSHOT of a portfolio app (e.g., Trade Republic).

        **INSTRUCTIONS:**
        1. Look at the image and identify each stock/crypto position.
        2. Extract the **Ticker Symbol** (e.g., "Nvidia" -> NVDA, "Render" -> RNDR-USD, "Solana" -> SOL-USD).
           - BE SMART about Crypto names.
        3. **CRITICAL:** The user might be showing a "List View" which only shows **Current Value** and **PnL** (Profit/Loss), but NOT Quantity.
           - If "Quantity" is visible, extract it.
           - If "Quantity" is MISSING, extract **Current Value** (e.g., "584,63 €" -> 584.63) and **PnL** (e.g., "-237,52 €" -> -237.52).
        4. Extract **Sector** (Tech, Crypto, Auto, etc.).

        **OUTPUT FORMAT:**
        Return strictly a JSON list of objects:
        [
            {
                "ticker": "NVDA",
                "quantity": 10.5,       // null if not found
                "avg_price": 80.0,      // null if not found
                "current_value": 840.0, // Extract this if quantity is missing
                "pnl": 40.0,            // Extract this (can be negative)
                "sector": "Tech"
            }
        ]
        If no data found, return [].
        """

        try:
            # Using gemini-1.5-pro as requested (Most powerful model before v3)
            response = self.client.models.generate_content(
                model='gemini-1.5-pro', 
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw_data = json.loads(response.text)
            
            # Post-Processing: Fill missing links using Market Data
            final_data = []
            market = MarketData()
            
            for item in raw_data:
                ticker = item.get('ticker')
                qty = item.get('quantity')
                avg = item.get('avg_price')
                val = item.get('current_value')
                pnl = item.get('pnl')
                
                # Case 1: We have Quantity and Avg Price (Perfect)
                if qty is not None and avg is not None:
                    final_data.append(item)
                    continue
                    
                # Case 2: We have Value and PnL (Back-calculate)
                if val is not None and pnl is not None:
                    try:
                        # 1. Get Live Price
                        # Since we don't have an easy sync method in MarketData yet that returns float, 
                        # we might need to instantiate or check if we can get a quick price.
                        # For now, let's assume we can fetch technical summary or just use a helper.
                        # Actually, MarketData relies on yfinance, which is synchronous.
                        # We will make a quick helper in MarketData or use a direct call here?
                        # Better to keep it clean. Let's assume we proceed with an approximation or 0 if fail.
                        
                        # Note: MarketData.get_technical_summary does a lot. We just need price.
                        # Let's import yfinance directly here for speed/simplicity or trust MarketData?
                        # Accessing a private method or adding one to MarketData is cleaner.
                        # We'll use a hack using yfinance directly here to avoid editing MarketData again unnecessarily,
                        # OR we assume the prompt gave us enough. 
                        # Wait, we really need the price.
                        import yfinance as yf
                        t = yf.Ticker(ticker)
                        current_price = t.history(period="1d")['Close'].iloc[-1]
                        
                        # 2. Derive Quantity
                        # Value = Qty * CurrentPrice => Qty = Value / CurrentPrice
                        calculated_qty = val / current_price
                        
                        # 3. Derive Cost Basis and Avg Price
                        # Value = Cost + PnL => Cost = Value - PnL
                        cost_basis = val - pnl
                        calculated_avg = cost_basis / calculated_qty
                        
                        item['quantity'] = round(calculated_qty, 4)
                        item['avg_price'] = round(calculated_avg, 2)
                        
                        logger.info(f"Back-calculated {ticker}: Qty={calculated_qty}, Avg={calculated_avg}")
                        final_data.append(item)
                    except Exception as ex:
                        logger.warning(f"Could not back-calculate for {ticker}: {ex}")
                        # Append anyway, maybe user can fix manually? Or skip?
                        # If we append with None, DB might fail if columns are not nullable. 
                        # DB Handler expects float. Let's put 0.0 placeholders if fail.
                        item['quantity'] = item.get('quantity') or 0.0
                        item['avg_price'] = item.get('avg_price') or 0.0
                        final_data.append(item)
                else:
                    # Partial data unusable
                    continue

            return final_data

        except Exception as e:
            logger.error(f"Error parsing portfolio image: {e}")
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
