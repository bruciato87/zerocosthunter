import os
import logging
from datetime import datetime, timedelta
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DBHandler:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")
        self.supabase: Client = create_client(url, key)

    def get_portfolio(self):
        """Fetch current holdings from the portfolio table."""
        try:
            response = self.supabase.table("portfolio").select("*").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return []

    def log_prediction(self, ticker: str, sentiment: str, reasoning: str, prediction_sentence: str, confidence_score: float, source_url: str):
        """Save AI analysis to predictions table."""
        try:
            data = {
                "ticker": ticker,
                "sentiment": sentiment,
                "reasoning": reasoning,
                "prediction_sentence": prediction_sentence,
                "confidence_score": confidence_score,
                "source_news_url": source_url,
                "created_at": datetime.utcnow().isoformat()
            }
            self.supabase.table("predictions").insert(data).execute()
            logger.info(f"Logged prediction for {ticker}: {sentiment}")
        except Exception as e:
            logger.error(f"Error logging prediction for {ticker}: {e}")

    def check_if_analyzed_recently(self, ticker: str, hours: int = 24) -> bool:
        """Check if a ticker has been analyzed in the last N hours to avoid spam."""
        try:
            time_threshold = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            response = self.supabase.table("predictions") \
                .select("id") \
                .eq("ticker", ticker) \
                .gte("created_at", time_threshold) \
                .execute()
            
            analyzed = len(response.data) > 0
            if analyzed:
                logger.info(f"Ticker {ticker} analyzed recently. Skipping.")
            return analyzed
        except Exception as e:
            logger.error(f"Error checking recent analysis for {ticker}: {e}")
            return False

    def log_system_event(self, level: str, module: str, message: str):
        """Log system events to the logs table."""
        try:
            data = {
                "level": level,
                "module": module,
                "message": message,
                "created_at": datetime.utcnow().isoformat()
            }
            self.supabase.table("logs").insert(data).execute()
        except Exception as e:
            print(f"Failed to log system event to DB: {e}") # Fallback to print

if __name__ == "__main__":
    # Test connection
    try:
        db = DBHandler()
        print("Supabase connection successful.")
    except Exception as e:
        print(f"Supabase connection failed: {e}")
