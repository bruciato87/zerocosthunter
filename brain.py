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

    def analyze_news_batch(self, news_list, performance_context=None, insider_context=None):
        """
        [2024 UPDATE] V3.0 Hybrid Brain with Memory & Insider Info.
        Analyze a batch of news items to find high-quality trading opportunities.
        - performance_context: Dict of {"ticker": {stats}} representing past accuracy.
        - insider_context: Dict of { "overall": "EXTREME FEAR", ... }
        """
        if not news_list:
            logger.info("No news to analyze.")
            return []

        # [FEEDBACK LOOP INJECTION]
        memories = ""
        if performance_context:
            memories = "\n[PAST PERFORMANCE MEMORY]\n"
            for ticker, stats in performance_context.items():
                if stats['status'] == 'NEGATIVE':
                    memories += f"WARNING: {ticker} -> High Failure Rate ({stats['win_rate']}% wins). Be extremely skeptical. Require overwhelming evidence to BUY.\n"
                elif stats['status'] == 'POSITIVE':
                    memories += f"NOTE: {ticker} -> High Success Rate ({stats['win_rate']}% wins). Your past logic works well here.\n"
                else:
                    memories += f"INFO: {ticker} -> Neutral/Insufficient history.\n"

        # [INSIDER SENTIMENT INJECTION]
        sentiment_bg = ""
        if insider_context:
            mood = insider_context.get('overall', 'NEUTRAL')
            fg_val = insider_context.get('crypto', {}).get('value', 50)
            
            # Formatting social headlines
            social_lines = ""
            headlines = insider_context.get('social', [])
            if headlines:
                social_lines = "\n\n[SOCIAL HYPE - REDDIT TRENDING]\n" + "\n".join([f"- {h}" for h in headlines])

            sentiment_bg = f"""
            [MARKET SENTIMENT CONTEXT]
            Market Mood: {mood} (Index: {fg_val}/100).
            STRATEGY: "Be Greedy when others are Fearful".
            - If "EXTREME FEAR" (<20): Look for quality assets at a discount ("Buy the Dip").
            - If "EXTREME GREED" (>80): Be cautious of tops ("Take Profit" or "Wait").
            {social_lines}
            """

        # Prepare the prompt
        news_text = "\n\n".join([f"Source: {item['source']}\nTitle: {item['title']}\nSummary: {item['summary']}" for item in news_list])
        
        prompt = f"""
        ROLES: [Financial Analyst, Hedge Fund Manager, Quantitative Trader]
        {memories}
        {sentiment_bg}
        
        **SYSTEM ROLE:**
        You are a Senior Investment Analyst & Quantitative Trader.
        Your goal is to validate market news with Technical Data AND produce a concrete Quantitative Prediction.

        **USER CONTEXT:**
        - The user is a **European Investor** (Currency: **EUR**).
        - If news mentions USD prices (e.g. "Apple to $200"), KEEP the target in USD ($) but ensure your reasoning considers the asset quality.
        - **Upside %** is universal, so focus heavily on that.

        **CRITICAL INSTRUCTION: TECHNICAL CHECKS (PATH C)**
        - The input news will contain technical tags (e.g. `[Technical: Price: $100, RSI: 80, Trend: BULLISH]`).
        - **YOU MUST INCORPORATE THIS DATA.**
        - **RSI RULE:** If RSI > 75 (Overbought), avoid "BUY" unless news is fundamental-shifting (e.g. Buyout). Prefer "HOLD".
        - **TREND RULE:** If Trend is "BEARISH", be cautious. "Cheaper" is often a trap.
        - **ATH RULE:** If "Diff from 6m High" is < -20%, this is a "Discount". Good for "Buy the Dip".

        **LANGUAGE:**
        - **Reasoning**: MUST be in **ITALIAN**.
        - **Sentiment**: MUST be ONE of: ["BUY", "SELL", "ACCUMULATE", "PANIC SELL", "HOLD"].
        
        **CRITICAL FILTERS:**
        1.  **Trade Republic Friendly Only:** Focus ONLY on major High-Cap Stocks and Major Cryptocurrencies.
        2.  **Ignore:** Penny stocks, low volume altcoins, obscure companies.
        3.  **OWNERSHIP RULE (CRITICAL):**
            -   **IF [Portfolio] tag is present:** You may issue ANY signal.
            -   **IF [Portfolio] tag is MISSING:**
                -   **MUST** use "**BUY**" if the opportunity is good.
                -   **MUST NOT** use "ACCUMULATE", "HOLD", "SELL".
                -   If neutral/negative, **SKIP IT**.
        
        **QUANTITATIVE ANALYSIS (NEW):**
        For every signal, you MUST estimate:
        1.  **Risk Score (1-10):**
            - 1-3: Low Risk (Trend Bullish, RSI Neutral, Blue Chip)
            - 4-7: Medium Risk
            - 8-10: High Risk (Crypto, RSI > 80, Speculative)
        2.  **Target Price (Short Term):**
            - Extract from the news (e.g., "Analyst sets $150 target").
            - If no analyst target, estimate a logical resistance.
            - **Format:** String with currency (e.g. "$150" or "€140").
        3.  **Upside Percentage:**
            - Numeric value of potential gain (e.g. 15.5 for +15.5%).
            - Return 0.0 if unknown/negative.

        **OUTPUT FORMAT:** JSON list.

        **NEWS DATA:**
        {news_text}

        **INSTRUCTIONS:**
        For each news item that contains a SIGNIFICANT signal:
        - Extract **Ticker**, **Type**, **Sentiment**.
        - Write **Reasoning** in **ITALIAN**, concise but insightful.
        - **Risk Score** (1-10), **Target Price**, **Upside %**.
        - **Confidence Score** (0.0 to 1.0).
        
        **JSON FIELDS:** 
        ticker, asset_type, sentiment, reasoning, confidence, risk_score (int), target_price (str), upside_percentage (float)

        Return strictly a JSON list.
        Example:
        [
            {{
                "ticker": "AAPL",
                "sentiment": "BUY",
                "reasoning": "Strong earnings. Analyst upgraded target to $200. Upside is clear.",
                "confidence": 0.85,
                "risk_score": 4,
                "target_price": "$200",
                "upside_percentage": 12.5
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
        2. **Identify the Currency:**
           - **RULE #1:** Look at the symbol next to the **Value** or **Price** (e.g. `100 €` vs `$100`).
           - **RULE #2:** IGNORE text in the Asset Name. "Core MSCI World USD" is just a name; if the price is `100 €`, the currency is **EUR**.
           - **RULE #3:** Look for decimal format: `1.000,00` -> **EUR**. `1,000.00` -> **USD**.
           - **Default to EUR** if you see `€` anywhere. Only return "USD" if you see `$`.
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
           - **Extract NAME:** The full name of the asset (e.g. "NVIDIA Corp").
           - **Extract TYPE:** "Crypto", "Stock", "ETF", or "Unknown".
         4. **Data Extraction:**
           - If "Quantity" is MISSING, extract **Current Value** and **PnL**.
            - **Detail View Specific:**
              - "Totale" or big number at top = **Current Value**.
              - "Guadagno" or "+/-" number = **PnL**.
              - **Avg Price**: Look for **"Prezzo d'acquisto >"** or "Prezzo d'acq". The number is usually **BELOW** or to the right.
              - **Quantity**: Look for **"Azioni"** or "Shares" or "Quote". The number is usually **BELOW** or to the right.
              - "La tua posizione" block contains the data.
              - **EXAMPLE (Layout with newlines):**
                 "Azioni
                  30,395484"      -> Quantity = 30.395484
                 "Prezzo d'acquisto >
                  107,68 €"       -> Avg Price = 107.68
            - **IMPORTANT:** Convert all numbers to standard **FLOAT format with DOTS** (e.g., "1.250,50" -> 1250.50).

        **OUTPUT FORMAT:**
        Return strictly a JSON object:
        {
            "currency": "EUR",
            "holdings": [
                {
                    "ticker": "EUNL.DE",
                    "name": "Core MSCI World",
                    "asset_type": "ETF",
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
            # Using gemini-flash-latest (Generic Alias, often most reliable/generous)
            response = self.client.models.generate_content(
                model='gemini-flash-latest',
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
            # Handle both list (legacy) and new dict format
            if isinstance(raw_response, list):
                raw_data = raw_response
                currency = "EUR" # Default changed to EUR (User Preference)
            else:
                raw_data = raw_response.get("holdings", [])
                currency = raw_response.get("currency", "EUR")

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
            
            # Rate for USD to EUR conversion (Inverse)
            usd_eur_rate = 1 / eur_usd_rate if eur_usd_rate else 0.91

            for item in raw_data:
                ticker = item.get('ticker')
                
                # SAFETY FORCE: If ticker is from a European exchange (.DE, .MI), force currency to EUR
                # This prevents "MSCI World USD" name from tricking the model into thinking it's USD
                if ticker and (ticker.endswith('.DE') or ticker.endswith('.MI')):
                    currency = "EUR"

                
                # SAFETY CHECK: If Ticker is None or UNKNOWN
                # Update: Allow UNKNOWN if we have a valid Quantity (Split View Scenario)
                if (not ticker or ticker == "UNKNOWN") and (not item.get('quantity') and not item.get('current_value')):
                    logger.warning(f"Skipping item with invalid ticker and no value: {item}")
                    continue
                
                if not ticker:
                    ticker = "UNKNOWN"

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

                # Case 2.5 (New): We have Quantity, Value, PnL but MISSING Avg Price
                # We can calculate Avg Price directly: (Value - PnL) / Quantity
                if qty is not None and qty > 0 and val is not None and pnl is not None and avg is None:
                    try:
                        cost_basis = val - pnl
                        calculated_avg = cost_basis / qty
                        item['avg_price'] = round(calculated_avg, 2)
                        logger.info(f"Deriving AvgPrice from Qty/Val/PnL for {ticker}: {calculated_avg}")
                        final_data.append(item)
                        continue
                    except Exception as e:
                         logger.warning(f"Math error deriving AvgPrice: {e}")
                
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
                            
                        current_price = hist['Close'].iloc[-1]
                        
                        # Determine if we need to convert the fetched price from USD to EUR logic
                        # If ticker is EUNL.DE, the price is in EUR.
                        price_is_eur = ticker.endswith('.DE') or ticker.endswith('.MI') or ticker.endswith('.PA')
                        
                        calculated_qty = 0.0
                        
                        if price_is_eur:
                             # Direct calculation: Value (EUR) / Price (EUR)
                             calculated_qty = val / current_price
                             # Cost Basis (EUR)
                             cost_basis_eur = val - pnl
                             calculated_avg_eur = cost_basis_eur / calculated_qty
                        else:
                             # USD Logic (Original)
                             # Calculate Value in USD first
                             value_usd = val
                             if currency == "EUR":
                                value_usd = val * eur_usd_rate
                                
                             calculated_qty = value_usd / current_price
                             
                             # Derive Cost Basis and Avg Price (keep in EUR)
                             cost_basis_eur = val - pnl
                             calculated_avg_eur = cost_basis_eur / calculated_qty
                        
                        item['quantity'] = round(calculated_qty, 4)
                        item['avg_price'] = round(calculated_avg_eur, 2)
                        
                        logger.info(f"Back-calculated {ticker}: Qty={calculated_qty}, Avg={calculated_avg_eur}€ (Live Price Used, Is_Eur={price_is_eur})")
                        final_data.append(item)
                        continue # Skip to next item to avoid appending twice
                    except Exception as ex:
                        logger.warning(f"Could not back-calculate for {ticker}: {ex}")
                        item['quantity'] = item.get('quantity') or 0.0
                        item['avg_price'] = item.get('avg_price') or 0.0
                        final_data.append(item)
                else:
                    final_data.append(item)

            # FINAL STEP: Normalize to EUR if source was USD
            # This ensures DB always stores EUR
            if currency == "USD":
                logger.info("Converting USD portfolio data to EUR...")
                for item in final_data:
                    # Quantity does not change
                    # Avg Price, Current Value, PnL MUST convert
                    if item.get('avg_price'):
                        item['avg_price'] = round(item['avg_price'] * usd_eur_rate, 2)
                    if item.get('current_value'):
                        item['current_value'] = round(item['current_value'] * usd_eur_rate, 2)
                    if item.get('pnl'):
                        item['pnl'] = round(item['pnl'] * usd_eur_rate, 2)
            
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

    def generate_deep_dive(self, ticker: str, news_list: list, technical_data: str, portfolio_context: str = None):
        """
        Generates a Strategic Analysis Report (Path B).
        """
        if not news_list:
            return f"❌ Non ho trovato notizie recenti rilevanti per **{ticker}**."

        news_text = "\n\n".join([f"Source: {item['source']}\nTitle: {item['title']}\nContent: {item['summary']}" for item in news_list])
        
        prompt = f"""
        **SYSTEM ROLE:**
        You are a Senior Hedge Fund Strategist. The user has requested a DEEP DIVE analysis on **{ticker}**.
        
        **INPUT DATA:**
        - **Technical Context:** {technical_data}
        - **Portfolio Context:** {portfolio_context or "Not owned"}
        - **Recent News (Full Text):**
        {news_text}

        **TASK:**
        Write a concise but professional "Battle Report" in **ITALIAN**.
        
        **FORMAT:**
        
        # 🛡️ Analisi Strategica: {ticker}
        
        ## 🐂 Bull Case (Perché potrebbe salire)
        - Bullet point 1 (cite specific earnings/news)
        - Bullet point 2
        
        ## 🐻 Bear Case (Rischi Principali)
        - Bullet point 1 (Risk factors)
        - Bullet point 2
        
        ## 🔮 The Verdict
        - **Decisione:** [BUY / HOLD / SELL / WAIT]
        - **Target Price (Est):** [Price or "N/A"]
        - **Risk Score:** [1-10]
        - **Catalyst:** Cosa stiamo aspettando? (e.g. Earnings date, FDA approval)

        **TONE:** Professional, Direct, No Fluff. Use Markdown.
        """

        try:
            logger.info(f"Generating Deep Dive for {ticker}...")
            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Deep Dive failed: {e}")
            return "⚠️ Errore durante l'analisi approfondita."

if __name__ == "__main__":
    b = Brain()
    # Test stub
