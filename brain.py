from google import genai
from google.genai import types
import os
import logging
import json
from market_data import MarketData
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
        1. Identify the **Currency** of the portfolio (e.g., EUR, USD). Look for symbols € or $.
        2. Identify each stock/crypto position.
        3. Extract the **Ticker Symbol** efficiently.
           - For Crypto, default to USD pair if unsure (e.g. "RNDR-USD"), but we will adjust for currency later.
        4. **CRITICAL:** Handle "List View" data:
           - If "Quantity" is MISSING, extract **Current Value** and **PnL**.
           - **IMPORTANT:** Convert all numbers to standard **FLOAT format with DOTS** (e.g., "1.250,50" -> 1250.50). **REMOVE THOUSAND SEPARATORS. REPLACE DECIMAL COMMAS WITH DOTS.**

        **OUTPUT FORMAT:**
        Return strictly a JSON list of objects (plus a 'meta' field for currency if possible, but let's keep it simple list).
        ACTUALLY, return a JSON object:
        {
            "currency": "EUR",
            "holdings": [
                {
                    "ticker": "RNDR-USD",
                    "quantity": null,
                    "avg_price": null,
                    "current_value": 564.63,
                    "pnl": -237.52,
                    "sector": "Crypto"
                }
            ]
        }
        """

        try:
            # Using gemini-2.5-flash (Newest Stable, High Free Quota)
            response = self.client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            raw_response = json.loads(response.text)
            
            # Handle both list (legacy) and new dict format
            if isinstance(raw_response, list):
                raw_data = raw_response
                currency = "USD" # Default
            else:
                raw_data = raw_response.get("holdings", [])
                currency = raw_response.get("currency", "USD")

            final_data = []
            market = MarketData()
            
            import yfinance as yf

            for item in raw_data:
                ticker = item.get('ticker')
                # Clean numbers just in case
                def clean_float(val):
                    if isinstance(val, str):
                        val = val.replace('.', '').replace(',', '.') # Assume EU format input if string remnants exist
                    return float(val) if val is not None else None

                qty = clean_float(item.get('quantity'))
                avg = clean_float(item.get('avg_price'))
                val = clean_float(item.get('current_value'))
                pnl = clean_float(item.get('pnl'))
                
                # Update item with cleaned floats
                item['quantity'] = qty
                item['avg_price'] = avg
                
                # Logic to fix Ticker Currency if needed
                # If portfolio is EUR and Ticker is USD pair, we should try to fetch EUR price for math
                fetch_ticker = ticker
                if currency == "EUR" and "USD" in ticker:
                    # Try switching -USD to -EUR
                    fetch_ticker = ticker.replace("USD", "EUR")
                
                # Case 1: We have Quantity and Avg Price (Perfect)
                if qty is not None and avg is not None:
                    final_data.append(item)
                    continue
                    
                # Case 2: We have Value and PnL (Back-calculate)
                if val is not None and pnl is not None:
                    try:
                        # 1. Get Live Price
                        # Use yfinance correctly
                        t = yf.Ticker(fetch_ticker)
                        hist = t.history(period="1d")
                        
                        if hist.empty:
                            # Fallback: maybe the EUR ticker doesn't exist? Try original USD ticker and convert?
                            # For simplicity, if RNDR-EUR fails, warn.
                            logger.warning(f"Could not fetch data for {fetch_ticker}")
                            item['ticker'] = ticker # Keep original
                            item['quantity'] = 0.0
                            item['avg_price'] = 0.0
                            final_data.append(item)
                            continue
                            
                        current_price = hist['Close'].iloc[-1]
                        
                        # 2. Derive Quantity
                        # Value (EUR) / Price (EUR) = Qty
                        calculated_qty = val / current_price
                        
                        # 3. Derive Cost Basis and Avg Price
                        # Value = Cost + PnL => Cost = Value - PnL
                        cost_basis = val - pnl
                        calculated_avg = cost_basis / calculated_qty
                        
                        item['ticker'] = ticker # Store the canonical ticker (e.g. RNDR-USD) for global monitoring? 
                        # Actually if user tracks in EUR, we might want to store EUR ticker?
                        # But system is USD based. 
                        # Let's keep the Ticker as the 'System' ticker (USD) but calc qty correctly.
                        
                        # Wait, if we calculated Qty using EUR price, Qty is correct.
                        # Avg Price (in EUR) is correct.
                        # But if we store RNDR-USD, subsequent lookups will be USD.
                        # Mixed currency portfolios are tricky. 
                        # For now, let's just make the Numbers Correct (Quantity is absolute). 
                        # We will store the Avg Price in the CURRENCY DETECTED?
                        # The DB schema has no currency column. 
                        # Let's assume we store everything as is, and the user understands the price is in their currency.
                        
                        item['quantity'] = round(calculated_qty, 4)
                        item['avg_price'] = round(calculated_avg, 2)
                        
                        logger.info(f"Back-calculated {ticker} ({currency}): Qty={calculated_qty}, Avg={calculated_avg}")
                        final_data.append(item)
                    except Exception as ex:
                        logger.warning(f"Could not back-calculate for {ticker}: {ex}")
                        item['quantity'] = item.get('quantity') or 0.0
                        item['avg_price'] = item.get('avg_price') or 0.0
                        final_data.append(item)
                else:
                    final_data.append(item)

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
