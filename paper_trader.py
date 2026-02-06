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

    
    def _get_or_create_account(self, chat_id: int):
        """Get account cash balance or create if not exists (Default €10k)."""
        try:
            res = self.db.supabase.table("paper_accounts").select("*").eq("chat_id", chat_id).execute()
            if res.data:
                return res.data[0]
            else:
                # Create new account
                new_acc = {"chat_id": chat_id, "cash_balance": 10000.00}
                self.db.supabase.table("paper_accounts").insert(new_acc).execute()
                return new_acc
        except Exception as e:
            logger.error(f"PaperTrader: Failed to get account for {chat_id}: {e}")
            return None

    def execute_trade(self, chat_id: int, ticker: str, action: str, quantity: float, price_eur: float, reason: str = "", sl: float = None, tp: float = None):
        """
        Executes a simulated trade with Cash Check and Risk Management.
        action: 'BUY' or 'SELL'
        sl: Stop Loss Price (Optional)
        tp: Take Profit Price (Optional)
        """
        try:
            ticker = ticker.upper()
            action = action.upper()
            total_val = quantity * price_eur
            
            # --- CASH MANAGEMENT ---
            account = self._get_or_create_account(chat_id)
            if not account:
                logger.warning(f"PaperTrader: No account found/created for {chat_id}. Aborting trade.")
                return False
                
            cash = float(account.get('cash_balance', 0))
            
            if action == 'BUY':
                if cash < total_val:
                    logger.warning(f"PaperTrader: Insufficient funds for {ticker}. Cash: €{cash:.2f}, Required: €{total_val:.2f}")
                    return False
                # Deduct Cash
                new_cash = cash - total_val
                
            elif action == 'SELL':
                # Add Cash
                new_cash = cash + total_val

            # Update Cash Balance
            self.db.supabase.table("paper_accounts").update({"cash_balance": new_cash}).eq("chat_id", chat_id).execute()
            # -----------------------

            # 1. Log the Trade
            trade_data = {
                "chat_id": chat_id,
                "ticker": ticker,
                "action": action,
                "quantity": quantity,
                "price": price_eur,
                "total_value": total_val,
                "trade_reason": reason,
                "simulated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
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
                    
                    update_data = {
                        "quantity": new_qty,
                        "avg_price": new_avg
                    }
                    # Update SL/TP only if provided (or strict override logic?)
                    # For now, if provided, valid. If not, keep old or None?
                    # Let's overwrite if provided.
                    if sl: update_data["stop_loss"] = sl
                    if tp: update_data["take_profit"] = tp
                    
                    self.db.supabase.table("paper_portfolio").update(update_data).eq("id", holding['id']).execute()
                else:
                    # New Position
                    new_pos = {
                        "chat_id": chat_id,
                        "ticker": ticker,
                        "quantity": quantity,
                        "avg_price": price_eur,
                        "stop_loss": sl,
                        "take_profit": tp
                    }
                    self.db.supabase.table("paper_portfolio").insert(new_pos).execute()

            elif action == 'SELL':
                if not holding:
                    logger.warning(f"PaperTrader: Cannot SELL {ticker}, not owned.")
                    # Revert cash add? Yes.
                    self.db.supabase.table("paper_accounts").update({"cash_balance": cash}).eq("chat_id", chat_id).execute()
                    return False
                
                new_qty = holding['quantity'] - quantity
                if new_qty <= 0:
                    # Full Close
                    self.db.supabase.table("paper_portfolio").delete().eq("id", holding['id']).execute()
                else:
                    # Partial Close (Avg Price stays same)
                    self.db.supabase.table("paper_portfolio").update({
                        "quantity": new_qty
                    }).eq("id", holding['id']).execute()

            logger.info(f"PAPER: RECORDED {action} of {quantity} {ticker} @ {price_eur:.4f} EUR executed. Cash remaining: €{new_cash:.2f}")
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
