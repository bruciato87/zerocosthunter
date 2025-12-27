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
        1. **Identify the View Type:**
           - **List View:** Multiple assets shown.
           - **Detail View:** A single asset is shown. **CRITICAL:** The Asset Name is usually at the very top (e.g. "Bitcoin", "NVIDIA", "S&P 500").
        2. **Identify the Currency:** (e.g., EUR, USD). Look for symbols € or $.
        3. **Extract Ticker Symbol:**
           - **Map Names to Tickers:**
             - "Bitcoin" -> "BTC-USD"
             - "Ethereum" -> "ETH-USD"
             - "NVIDIA" -> "NVDA"
             - "Apple" -> "AAPL"
             - **ETFs (CRITICAL for Trade Republic):**
               - "Core MSCI World" / "iShares Core MSCI World" -> "EUNL.DE"
               - "S&P 500" / "iShares Core S&P 500" -> "SXR8.DE"
               - "Nasdaq 100" -> "SXRV.DE"
               - "Global Clean Energy" -> "INRG.DE"
               - "Core DAX" -> "DAXEX.DE"
           - **Look Everywhere:** If the ticker isn't explicit, infer it from the Asset Name.
           - **CRITICAL:** DO NOT RETURN NULL. If you absolutely cannot match it, return "UNKNOWN".
        4. **Data Extraction:**
           - If "Quantity" is MISSING, extract **Current Value** and **PnL**.
           - **Detail View Specific:**
             - "Totale" or big number at top = **Current Value**.
             - "Guadagno" or "+/-" number = **PnL**.
             - "Prezzo d'acq" = **Avg Price** (use this if possible!).
             - "Azioni" or "Shares" or "Quote" = **Quantity**. **EXTRACT THIS EXACTLY.**
             - "La tua posizione" block contains the data.
           - **IMPORTANT:** Convert all numbers to standard **FLOAT format with DOTS** (e.g., "1.250,50" -> 1250.50).

        **OUTPUT FORMAT:**
        Return strictly a JSON object:
        {
            "currency": "EUR",
            "holdings": [
                {
                    "ticker": "EUNL.DE",
                    "quantity": null,
                    "avg_price": null,
                    "current_value": 3384.23,
                    "pnl": 111.34,
                    "sector": "ETF"
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

            # Post-Processing: Fill missing links using Market Data
            final_data = []
            market = MarketData()
            
            import yfinance as yf

            # 1. Get FX Rate (EUR -> USD)
            # We need to know how much 1 EUR is in USD to convert the User's Value(EUR) to Value(USD)
            # Ticker: EURUSD=X (Returns USD per 1 EUR)
            eur_usd_rate = 1.1 # Default fallback
            try:
                fx = yf.Ticker("EURUSD=X")
                hist_fx = fx.history(period="1d")
                if not hist_fx.empty:
                    eur_usd_rate = hist_fx['Close'].iloc[-1]
            except:
                logger.warning("Could not fetch EURUSD rate, using 1.1 fallback")

            for item in raw_data:
                ticker = item.get('ticker')
                
                # SAFETY CHECK: If Ticker is None or UNKNOWN, skip or warn
                if not ticker or ticker == "UNKNOWN":
                    logger.warning(f"Skipping item with invalid ticker: {item}")
                    continue

                # Helper to clean "1.000,00" to float
                def clean_float(val):
                    if isinstance(val, str):
                        # Remove dots (thousands), replace comma with dot
                        if ',' in val and '.' in val:
                             # mixed usage
                             val = val.replace('.', '').replace(',', '.')
                        elif ',' in val:
                             # simple comma decimal
                             val = val.replace(',', '.')
                    
                    try:
                        return float(val) if val is not None else None
                    except:
                        return None

                qty = clean_float(item.get('quantity'))
                avg = clean_float(item.get('avg_price'))
                val = clean_float(item.get('current_value'))
                pnl = clean_float(item.get('pnl'))
                
                # Update item with cleaned floats
                item['quantity'] = qty
                item['avg_price'] = avg
                
                # Case 1: We have Quantity and Avg Price (Perfect)
                if qty is not None and avg is not None:
                    final_data.append(item)
                    continue
                    
                # Case 2: We have Value, PnL AND (Crucially) Avg Price from OCR
                # This is the most precise way to get Quantity without external API
                if val is not None and pnl is not None and avg is not None and avg > 0:
                    try:
                         # Logic:
                         # Cost Basis = Value - PnL
                         # Quantity = Cost Basis / Avg Price
                         cost_basis = val - pnl
                         calculated_qty = cost_basis / avg
                         
                         item['quantity'] = round(calculated_qty, 4)
                         logger.info(f"Deriving Qty from OCR AvgPrice for {ticker}: Cost={cost_basis}, Avg={avg} -> Qty={calculated_qty}")
                         final_data.append(item)
                         continue
                    except Exception as e:
                         logger.warning(f"Math error using OCR AvgPrice: {e}")
                
                # Case 3: We have Value and PnL but NO Avg Price (Back-calculate using Live Price)
                if val is not None and pnl is not None:
                    try:
                        # Strategy:
                        # 1. We assume Ticker is USD based (e.g. RNDR-USD, NVDA).
                        # 2. We have Value in EUR.
                        # 3. Increase Robustness: specific fix for Render
                        fetch_ticker = ticker
                        if "RNDR" in ticker:
                             fetch_ticker = "RENDER-USD"
                        elif "MSCIWORLD" in ticker or "CORE MSCI" in ticker.upper():
                             # Fallback if Vision returned generic name
                             fetch_ticker = "EUNL.DE" 
                             ticker = "EUNL.DE" # Update saved ticker too
                        
                        t = yf.Ticker(fetch_ticker)
                        hist = t.history(period="1d")
                        
                        if hist.empty:
                             # Try fallback
                             t = yf.Ticker(ticker)
                             hist = t.history(period="1d")
                             
                        if hist.empty:
                            logger.warning(f"Could not fetch data for {ticker}")
                            item['quantity'] = 0.0
                            item['avg_price'] = 0.0
                            final_data.append(item)
                            continue
                            
                        current_price_usd = hist['Close'].iloc[-1]
                        
                        # Calculate Value in USD
                        # If currency is EUR: Value_USD = Value(EUR) * Rate(USD/EUR)
                        value_usd = val
                        if currency == "EUR":
                            value_usd = val * eur_usd_rate
                            
                        # Derive Quantity = Value(USD) / Price(USD)
                        calculated_qty = value_usd / current_price_usd
                        
                        # Derive Cost Basis and Avg Price (keep in EUR)
                        cost_basis_eur = val - pnl
                        calculated_avg_eur = cost_basis_eur / calculated_qty
                        
                        item['quantity'] = round(calculated_qty, 4)
                        item['avg_price'] = round(calculated_avg_eur, 2)
                        
                        logger.info(f"Back-calculated {ticker}: Qty={calculated_qty}, Avg={calculated_avg_eur}€ (Live Price Used)")
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
