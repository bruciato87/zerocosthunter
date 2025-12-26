import os
import logging
import sys
from dotenv import load_dotenv

# Load env vars if running locally
load_dotenv()

from db_handler import DBHandler
from hunter import NewsHunter
from brain import Brain
from telegram_bot import TelegramNotifier

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainController")

def run_pipeline():
    logger.info("Starting Zero-Cost Investment Hunter Pipeline...")

    # 1. Initialize Modules
    try:
        db = DBHandler()
        hunter = NewsHunter()
        brain = Brain()
        notifier = TelegramNotifier()
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 2. Fetch News
    news_items = hunter.fetch_news()
    if not news_items:
        logger.info("No news found. Exiting.")
        return

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

        # Check if recently analyzed to avoid spam
        if db.check_if_analyzed_recently(ticker):
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
        
        notifier.send_sync(alert_msg)
        processed_count += 1

    logger.info(f"Pipeline finished. Processed {processed_count} actionable signals.")

if __name__ == "__main__":
    run_pipeline()
