import os
import logging
import sys
from dotenv import load_dotenv

# Load env vars if running locally
load_dotenv()

from db_handler import DBHandler
from hunter import NewsHunter
from market_data import MarketData
from brain import Brain
from telegram_bot import TelegramNotifier
import re
import asyncio

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainController")

def run_pipeline():
    asyncio.run(run_async_pipeline())

async def run_async_pipeline():
    logger.info("Starting Zero-Cost Investment Hunter Pipeline...")

    # 1. Initialize Modules
    try:
        db = DBHandler()
        hunter = NewsHunter()
        brain = Brain()
        notifier = TelegramNotifier()
        market = MarketData()
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 2. Fetch News
    news_items = hunter.fetch_news()
    if not news_items:
        logger.info("No news found. Exiting.")
        return

    # 2.2 Load Portfolio
    logger.info("Loading Portfolio...")
    portfolio_map = db.get_portfolio_map()
    if portfolio_map:
        logger.info(f"Loaded {len(portfolio_map)} holdings.")

    # 2.5 Enrich News with Technical Data & Portfolio Context
    logger.info("Enriching news with Technical Data & Portfolio Context...")
    
    # Simple whitelist for Zero-Cost extraction (expandable)
    MONITORED_TICKERS = {
        "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
        "SOL": "SOL-USD", "SOLANA": "SOL-USD",
        "NVDA": "NVDA", "NVIDIA": "NVDA",
        "TSLA": "TSLA", "TESLA": "TSLA",
        "AAPL": "AAPL", "APPLE": "AAPL",
        "MSFT": "MSFT", "MICROSOFT": "MSFT",
        "AMZN": "AMZN", "AMAZON": "AMZN",
        "GOOG": "GOOGL", "GOOGLE": "GOOGL",
        "META": "META",
        "AMD": "AMD",
        "SPY": "SPY", "S&P 500": "SPY"
    }

    # Add Portfolio Tickers to Monitored list dynamically
    for p_ticker in portfolio_map.keys():
        MONITORED_TICKERS[p_ticker] = p_ticker

    for item in news_items:
        text_content = (item.get('title', '') + " " + item.get('summary', '')).upper()
        
        # Find first matching ticker
        detected_ticker = None
        for key, symbol in MONITORED_TICKERS.items():
            # word boundary check to avoid matching "BETTER" as "ETH"
            if re.search(r'\b' + re.escape(key) + r'\b', text_content):
                detected_ticker = symbol
                break
        
        if detected_ticker:
            extras = []
            
            # 1. Technicals
            tech_summary = market.get_technical_summary(detected_ticker)
            extras.append(f"Technical: {tech_summary}")
            
            # 2. Portfolio
            if detected_ticker in portfolio_map:
                holding = portfolio_map[detected_ticker]
                p_summary = f"OWNED {holding['quantity']} @ ${holding['avg_price']}"
                extras.append(f"Portfolio: {p_summary}")
                logger.info(f"Enriched {detected_ticker} with Portfolio data: {p_summary}")

            item['summary'] += "\n\n[" + " | ".join(extras) + "]"
            logger.info(f"Enriched {detected_ticker} news.")

    # 3. Analyze with AI
    logger.info("Analyzing news with Gemini...")
    predictions = brain.analyze_news_batch(news_items)

    processed_count = 0
    
    # 4. Process Predictions
    for pred in predictions:
        ticker = pred.get("ticker")
        sentiment = pred.get("sentiment")
        reasoning = pred.get("reasoning")
        confidence = pred.get("confidence", 0.0)
        source = pred.get("source", "Unknown")

        if not ticker or confidence < 0.7:  # Filter low confidence
            continue

        # Check if recently analyzed (Same Ticker + Same Sentiment = SPAM)
        if db.check_if_analyzed_recently(ticker, sentiment):
            continue

        # 5. Log to DB and Notify
        db.log_prediction(ticker, sentiment, reasoning, reasoning, confidence, source)
        
        # Format Alert
        icon = "🟢" if sentiment in ["BUY", "ACCUMULATE"] else "🔴" if sentiment in ["SELL", "PANIC SELL"] else "⚪"
        alert_msg = (
            f"{icon} **Signal Detected: {ticker}**\n"
            f"**Action:** {sentiment}\n"
            f"**Confidence:** {int(confidence * 100)}%\n"
            f"**Reasoning:** {reasoning}\n"
            f"**Source:** {source}"
        )
        
        await notifier.send_alert(alert_msg)
        processed_count += 1

    db.log_system_event("INFO", "Hunter", "Pipeline Finished")
    logger.info(f"Pipeline finished. Processed {processed_count} actionable signals.")

if __name__ == "__main__":
    run_pipeline()
