import logging
import datetime
import uuid
from db_handler import DBHandler

# Configure logging
logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self, db_handler=None):
        self.db = db_handler if db_handler else DBHandler()

    def get_portfolio(self, chat_id: int = None):
        """Fetches paper portfolio. If chat_id is None, fetches all (Admin view)."""
        try:
            query = self.db.supabase.table("paper_portfolio").select("*")
            if chat_id:
                query = query.eq("chat_id", chat_id)
            
            res = query.execute()
            return res.data
        except Exception as e:
            logger.error(f"PaperTrader: Error fetching portfolio: {e}")
            return []

    def execute_trade(self, chat_id: int, ticker: str, action: str, quantity: float, price_eur: float, reason: str = ""):
        """
        Executes a simulated trade.
        action: 'BUY' or 'SELL'
        """
        try:
            ticker = ticker.upper()
            action = action.upper()
            total_val = quantity * price_eur

            # 1. Log the Trade
            trade_data = {
                "chat_id": chat_id,
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "price": price_eur,
                "total_value": total_val,
                "trade_reason": reason,
                "simulated_at": datetime.datetime.utcnow().isoformat()
            }
            self.db.supabase.table("paper_trades").insert(trade_data).execute()

            # 2. Update Portfolio
            current_holdings = self.get_portfolio(chat_id)
            holding = next((h for h in current_holdings if h['ticker'] == ticker), None)

            if action == 'BUY':
                if holding:
                    # Average Up/Down
                    old_qty = holding['quantity']
                    old_avg = holding['avg_price']
                    new_qty = old_qty + quantity
                    new_avg = ((old_qty * old_avg) + (quantity * price_eur)) / new_qty
                    
                    self.db.supabase.table("paper_portfolio").update({
                        "quantity": new_qty,
                        "avg_price": new_avg
                    }).eq("id", holding['id']).execute()
                else:
                    # New Position
                    self.db.supabase.table("paper_portfolio").insert({
                        "chat_id": chat_id,
                        "ticker": ticker,
                        "quantity": quantity,
                        "avg_price": price_eur
                    }).execute()

            elif action == 'SELL':
                if not holding:
                    logger.warning(f"PaperTrader: Cannot SELL {ticker}, not owned.")
                    return False
                
                # Update realized PnL in trade log? (Skipped for V1 simplicity)
                
                new_qty = holding['quantity'] - quantity
                if new_qty <= 0:
                    # Full Close
                    self.db.supabase.table("paper_portfolio").delete().eq("id", holding['id']).execute()
                else:
                    # Partial Close (Avg Price stays same)
                    self.db.supabase.table("paper_portfolio").update({
                        "quantity": new_qty
                    }).eq("id", holding['id']).execute()

            logger.info(f"PaperTrader: {action} {quantity} {ticker} @ €{price_eur} executed.")
            return True

        except Exception as e:
            logger.error(f"PaperTrader: Trade Execution Failed: {e}")
            return False

    def get_valuation(self, chat_id: int, market_data):
        """Calculates total value of paper portfolio."""
        portfolio = self.get_portfolio(chat_id) # Handles None internally now
        total_val = 0.0
        
        for p in portfolio:
            price, _ = market_data.get_smart_price_eur(p['ticker'])
            if price > 0:
                total_val += p['quantity'] * price
            else:
                total_val += p['quantity'] * p['avg_price'] # Fallback
                
        return total_val
