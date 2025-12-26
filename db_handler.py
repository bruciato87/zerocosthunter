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

    def get_portfolio_map(self):
        """Returns portfolio as a dictionary {ticker: {qty, avg_price, sector}} for fast lookup."""
        data = self.get_portfolio()
        if not data:
            return {}
        return {item['ticker']: item for item in data}

    def add_to_portfolio(self, ticker: str, amount: float, price: float, sector: str = "Unknown", is_confirmed: bool = True, chat_id: int = None):
        """Add or update a holding. Supports 'Draft' mode via is_confirmed=False."""
        try:
            data = {
                "ticker": ticker,
                "quantity": amount,
                "avg_price": price,
                "sector": sector,
                "is_confirmed": is_confirmed
            }
            if chat_id:
                data["chat_id"] = chat_id

            # upsert=True is default behavior if ID matches, but for ticker we rely on unique constraint
            # Note: unique constraint on 'ticker' might be an issue if multiple users draft the same ticker.
            # For a single-user bot, it's fine. For multi-user, unique should be (ticker, chat_id).
            # We'll assume single user for now or that 'ticker' is unique globally in this table.
            self.supabase.table("portfolio").upsert(data, on_conflict="ticker").execute()
            logger.info(f"Updated portfolio: {ticker} (Confirmed: {is_confirmed})")
        except Exception as e:
            logger.error(f"Error updating portfolio for {ticker}: {e}")

    def confirm_portfolio(self, chat_id: int):
        """Mark all drafts for a user as confirmed."""
        try:
            self.supabase.table("portfolio") \
                .update({"is_confirmed": True}) \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", False) \
                .execute()
            logger.info(f"Confirmed portfolio for chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Error confirming portfolio: {e}")
            raise e

    def delete_drafts(self, chat_id: int):
        """Delete unconfirmed drafts."""
        try:
            self.supabase.table("portfolio") \
                .delete() \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", False) \
                .execute()
            logger.info(f"Deleted drafts for chat_id {chat_id}")
        except Exception as e:
            logger.error(f"Error deleting drafts: {e}")
            raise e

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
