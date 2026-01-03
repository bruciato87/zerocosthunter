import os
import logging
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

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

    # --- USER MANAGEMENT (V8) ---
    # PAUSED

    # --- PORTFOLIO MANAGEMENT ---

    def get_portfolio(self, chat_id: int = None):
        """Fetch current holdings from the portfolio table. Optionally filter by chat_id."""
        try:
            query = self.supabase.table("portfolio").select("*")
            if chat_id:
                # Also filter for confirmed only if we are showing the portfolio view?
                # Usually "Show Portfolio" implies confirmed items. 
                # Let's filter for is_confirmed=True if chat_id is provided, 
                # assuming the user wants to see their active portfolio.
                query = query.eq("chat_id", chat_id).eq("is_confirmed", True)
            
            response = query.execute()
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

    def add_to_portfolio(self, ticker: str, amount: float, price: float, sector: str = "Unknown", asset_name: str = None, asset_type: str = "Unknown", is_confirmed: bool = True, chat_id: int = None):
        """Add or update a holding. Supports 'Draft' mode via is_confirmed=False."""
        try:
            data = {
                "ticker": ticker,
                "quantity": amount,
                "avg_price": price,
                "sector": sector,
                "asset_name": asset_name,
                "asset_type": asset_type,
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
        """Delete unconfirmed drafts. Keep chat_id based as drafts are ephemeral per chat."""
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
    def get_drafts(self, chat_id: int):
        """Fetch unconfirmed drafts for a user."""
        try:
            response = self.supabase.table("portfolio") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", False) \
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching drafts: {e}")
            return []

    def delete_asset(self, chat_id: int, ticker: str):
        """Delete a specific asset from the portfolio."""
        try:
            self.supabase.table("portfolio") \
                .delete() \
                .eq("chat_id", chat_id) \
                .eq("ticker", ticker) \
                .execute()
            logger.info(f"Deleted asset {ticker} for chat_id {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting asset {ticker}: {e}")
            return False

    def delete_portfolio(self, chat_id: int):
        """Delete ALL assets for a user (Reset)."""
        try:
            self.supabase.table("portfolio") \
                .delete() \
                .eq("chat_id", chat_id) \
                .execute()
            logger.info(f"Deleted entire portfolio for chat_id {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting portfolio for {chat_id}: {e}")
            return False

    def update_draft_quantity(self, record_id: str, quantity: float, avg_price: float = None):
        """Update quantity (and optionally avg_price) of a specific draft record."""
        try:
            data = {"quantity": quantity}
            if avg_price is not None and avg_price > 0:
                data["avg_price"] = avg_price
                
            self.supabase.table("portfolio") \
                .update(data) \
                .eq("id", record_id) \
                .execute()
            logger.info(f"Updated quantity={quantity}, avg_price={avg_price} for draft {record_id}")
        except Exception as e:
            logger.error(f"Error updating draft quantity: {e}")

    def update_draft_ticker(self, record_id: str, ticker: str, price: float):
        """Update ticker and price of a specific draft record."""
        try:
            data = {"ticker": ticker}
            if price:
                data["avg_price"] = price
            
            self.supabase.table("portfolio") \
                .update(data) \
                .eq("id", record_id) \
                .execute()
            logger.info(f"Updated ticker for draft {record_id} to {ticker}")
        except Exception as e:
            logger.error(f"Error updating draft ticker: {e}")

    def update_asset_price(self, chat_id: int, ticker: str, new_price: float):
        """Manually update the average buy price of a confirmed asset."""
        try:
            self.supabase.table("portfolio") \
                .update({"avg_price": new_price}) \
                .eq("chat_id", chat_id) \
                .eq("ticker", ticker) \
                .execute()
            logger.info(f"Manual price update: {ticker} -> {new_price} for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating asset price: {e}")
            return False

    def update_asset_ticker(self, chat_id: int, old_ticker: str, new_ticker: str):
        """Manually update the ticker of a confirmed asset."""
        try:
            self.supabase.table("portfolio") \
                .update({"ticker": new_ticker}) \
                .eq("chat_id", chat_id) \
                .eq("ticker", old_ticker) \
                .execute()
            logger.info(f"Manual ticker update: {old_ticker} -> {new_ticker} for {chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating asset ticker: {e}")
            return False

    def get_recent_confirmed_portfolio(self, chat_id: int, minutes: int = 5):
        """Fetch portfolio items confirmed in the last N minutes."""
        try:
            # Note: 'updated_at' is used as proxy for 'recently added' since created_at is missing
            # Using updated_at is a good proxy for 'recently added'.
            time_threshold = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
            response = self.supabase.table("portfolio") \
                .select("*") \
                .eq("chat_id", chat_id) \
                .eq("is_confirmed", True) \
                .gte("updated_at", time_threshold) \
                .execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching recent confirmed items: {e}")
            return []

    def get_audit_stats(self):
        """
        Calculates performance metrics from signal_tracking table.
        Returns: { 'win_rate': 0.0, 'total_trades': 0, 'wins': 0, 'losses': 0, 'open': 0 }
        """
        try:
            response = self.supabase.table("signal_tracking").select("status").execute()
            signals = response.data
            
            total = len(signals)
            wins = sum(1 for s in signals if s['status'] == 'WIN')
            losses = sum(1 for s in signals if s['status'] == 'LOSS')
            open_sigs = sum(1 for s in signals if s['status'] == 'OPEN')
            
            closed_trades = wins + losses
            win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0
            
            return {
                "win_rate": round(win_rate, 1),
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "open": open_sigs,
                "closed": closed_trades
            }
        except Exception as e:
            logger.error(f"Audit Stats Error: {e}")
            return {"win_rate": 0, "total_trades": 0, "wins": 0, "losses": 0, "open": 0, "closed": 0}

    def log_prediction(self, ticker: str, sentiment: str, reasoning: str, prediction_sentence: str, confidence_score: float, source_url: str, risk_score: int = 5, target_price: str = None, upside_percentage: float = 0.0):
        """Save AI analysis to predictions table."""
        try:
            data = {
                "ticker": ticker,
                "sentiment": sentiment,
                "reasoning": reasoning,
                "prediction_sentence": prediction_sentence,
                "confidence_score": confidence_score,
                "source_news_url": source_url,
                "risk_score": risk_score,
                "target_price": target_price,
                "upside_percentage": upside_percentage,
                "created_at": datetime.utcnow().isoformat()
            }
            response = self.supabase.table("predictions").insert(data).execute()
            logger.info(f"Logged prediction for {ticker}: {sentiment}")
            return response.data[0]['id'] if response.data else None
        except Exception as e:
            logger.error(f"Error logging prediction for {ticker}: {e}")

    def get_settings(self):
        """Fetch user settings (single row). Returns dict with defaults if empty."""
        try:
            response = self.supabase.table("user_settings").select("*").limit(1).execute()
            if response.data:
                return response.data[0]
            return {"min_confidence": 0.70, "only_portfolio": False} # Defaults
        except Exception as e:
            logger.error(f"Error fetching settings: {e}")
            return {"min_confidence": 0.70, "only_portfolio": False}

    def update_settings(self, min_confidence=None, only_portfolio=None):
        """Update the single settings row."""
        try:
            # First get the ID
            settings = self.get_settings()
            updates = {}
            if min_confidence is not None:
                updates["min_confidence"] = min_confidence
            if only_portfolio is not None:
                updates["only_portfolio"] = only_portfolio
            
            if updates and "id" in settings:
                self.supabase.table("user_settings").update(updates).eq("id", settings["id"]).execute()
                logger.info(f"Settings updated: {updates}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False

    def check_if_analyzed_recently(self, ticker: str, new_sentiment: str, hours: int = 24) -> bool:
        """
        Check if we should SKIP this alert.
        Returns TRUE (skip) if:
        - Ticker analyzed in last N hours AND Sentiment is SAME.
        Returns FALSE (allow) if:
        - Ticker not analyzed recently.
        - OR Sentiment has CHANGED (e.g. was HOLD, now BUY).
        """
        try:
            time_threshold = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            
            # Fetch most recent prediction for this ticker
            response = self.supabase.table("predictions") \
                .select("sentiment") \
                .eq("ticker", ticker) \
                .gte("created_at", time_threshold) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if not response.data:
                return False # No recent analysis, allow it.

            last_sentiment = response.data[0]['sentiment']
            
            if last_sentiment == new_sentiment:
                logger.info(f"Duplicate Signal: {ticker} was already {last_sentiment} recently. Skipping.")
                return True # SKIP (Duplicate)
            else:
                logger.info(f"Sentiment Shift: {ticker} changed from {last_sentiment} to {new_sentiment}. Allowing.")
                return False # ALLOW (Change)

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

    # --- DISTRIBUTED LOCK (Ad-Hoc via Logs) ---
    def acquire_hunt_lock(self, expiry_minutes: int = 2) -> bool:
        """
        Attempts to acquire a lock for the 'hunt' process.
        Returns True if acquired, False if already locked by another instance.
        Uses the 'logs' table with module='HUNTER_LOCK' as a semaphore.
        """
        try:
            # 1. Check state
            # Fetch most recent lock event
            response = self.supabase.table("logs") \
                .select("*") \
                .eq("module", "HUNTER_LOCK") \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            if response.data:
                last_event = response.data[0]
                status = last_event.get("message")
                created_at = datetime.fromisoformat(last_event.get("created_at").replace('Z', '+00:00')).replace(tzinfo=None)
                
                # Check if currently locked or recently finished (Debounce)
                if status == "LOCKED":
                    # Check expiry
                    now = datetime.utcnow()
                    if (now - created_at).total_seconds() < (expiry_minutes * 60):
                        logger.warning(f"Hunt Locked. Active since {created_at} (Expires in {expiry_minutes}m)")
                        return False # BUSY
                
                elif status == "RELEASED":
                    # Debounce Vercel Retries: If finished < 5 mins ago, block.
                    now = datetime.utcnow()
                    if (now - created_at).total_seconds() < 300: # 5 Minutes Debounce
                        logger.warning(f"Hunt Debounced. Recently finished at {created_at}. Ignoring retry.")
                        return False # DEBOUNCED

            # 2. Acquire
            self.log_system_event("INFO", "HUNTER_LOCK", "LOCKED")
            return True
        except Exception as e:
            logger.error(f"Lock Error: {e}")
            return True # Fail-open to avoid permanent blockage, or False? True is risky but prevents deadlock. 

    def release_hunt_lock(self):
        """Releases the hunt lock."""
        try:
             self.log_system_event("INFO", "HUNTER_LOCK", "RELEASED")
        except: pass

if __name__ == "__main__":
    # Test connection
    try:
        db = DBHandler()
        print("Supabase connection successful.")
    except Exception as e:
        print(f"Supabase connection failed: {e}")
