import logging
from db_handler import DBHandler
from market_data import MarketData
import time
from datetime import datetime, timezone

logger = logging.getLogger("Auditor")

class Auditor:
    def __init__(self):
        self.db = DBHandler()
        self.market = MarketData()
        # Configuration
        self.TAKE_PROFIT_PCT = 15.0  # +15%
        self.STOP_LOSS_PCT = -10.0   # -10%
        self.MAX_AGE_DAYS = 30       # Expire after 30 days if stagnating
        self.MIN_AGE_HOURS = 1       # Don't close signals younger than 1h (Prevent instant win/loss)
        
        # Heuristic: Known Cryptos vs Stocks
        self.KNOWN_CRYPTO = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "LINK", "LTC", "BCH", "UNI", "MATIC"}

    # ... record_signal omitted (unchanged) ...

    def audit_open_signals(self):
        """
        Updates PnL and Status for all OPEN signals.
        """
        try:
            # 1. Fetch OPEN signals
            response = self.db.supabase.table("signal_tracking").select("*").eq("status", "OPEN").execute()
            open_signals = response.data
            
            if not open_signals:
                logger.info("Auditor: No open signals to audit.")
                return []

            updates = []
            
            for sig in open_signals:
                ticker = sig['ticker']
                entry_price = float(sig['entry_price'])
                req_target_price = sig.get('target_price')
                target_price = float(req_target_price) if req_target_price else None
                created_at_iso = sig['created_at']
                
                # Check Signal Age
                try:
                    # ISO Format from Supabase: "2024-01-01T10:00:00+00:00" or "2024-01-01T10:00:00.123456+00:00"
                    # Replace 'Z' just in case, handle potential microseconds automatically via fromisoformat in modern python
                    # Note: Python 3.11+ handles fromisoformat strictly.
                    # Fallback to simple string split if needed.
                    c_ts = created_at_iso.replace('Z', '+00:00')
                    created_dt = datetime.fromisoformat(c_ts)
                    now_dt = datetime.now(timezone.utc)
                    age_hours = (now_dt - created_dt).total_seconds() / 3600
                except Exception as e:
                    logger.warning(f"Auditor: Timestamp parse failed for {ticker}: {e}. Treating as NEW.")
                    age_hours = 0 # Default SAFE (treat as new, don't close)
                
                # 2. Get Live Price
                # Uses MarketData helper which handles Crypto/Stocks
                # Using get_technical_summary logic simplified
                live_price, _ = self.market.get_crypto_data_coingecko(ticker)
                
                if not live_price:
                    # Fallback to Yahoo
                    try:
                        import yfinance as yf
                        
                        attempts = []
                        if ticker.upper() in self.KNOWN_CRYPTO:
                            attempts = [f"{ticker}-USD", ticker]
                        else:
                            attempts = [ticker, f"{ticker}-USD"]

                        for sym in attempts:
                            try:
                                t = yf.Ticker(sym)
                                hist = t.history(period="1d")
                                if not hist.empty:
                                    live_price = hist['Close'].iloc[-1]
                                    break
                            except: continue
                            
                    except: pass
                
                if not live_price:
                    logger.warning(f"Auditor: Could not fetch price for {ticker}. Skipping.")
                    continue

                # 3. Calculate PnL
                if entry_price > 0:
                    pnl_pct = ((live_price - entry_price) / entry_price) * 100
                else: 
                    pnl_pct = 0.0
                    
                new_status = "OPEN"
                
                # 4. Check Exit Conditions (Only if older than MIN_AGE_HOURS)
                if age_hours >= self.MIN_AGE_HOURS:
                    if pnl_pct >= self.TAKE_PROFIT_PCT:
                        new_status = "WIN"
                    # Require positive PnL to confirm Target Hit (Avoids 0% wins if target <= entry)
                    elif target_price and live_price >= target_price and pnl_pct > 0.5:
                        new_status = "WIN"
                    elif pnl_pct <= self.STOP_LOSS_PCT:
                        new_status = "LOSS"
                    
                    # Check Expiration (Age)
                    elif age_hours > (self.MAX_AGE_DAYS * 24):
                        new_status = "EXPIRED"

                # 5. Update DB
                update_data = {
                    "current_price": live_price,
                    "pnl_percent": round(pnl_pct, 2),
                    "status": new_status,
                    "updated_at": "now()"
                }
                
                self.db.supabase.table("signal_tracking").update(update_data).eq("id", sig['id']).execute()
                
                if new_status != "OPEN":
                    updates.append(f"{ticker}: {new_status} ({pnl_pct:+.2f}%)")
                    logger.info(f"Auditor: Signal Closed - {ticker} is {new_status}")

            return updates

        except Exception as e:
            logger.error(f"Auditor Audit Failed: {e}")
            return []

    def record_signal(self, ticker: str, entry_price: float = None, signal_id: str = None, target_price: float = None):
        """
        Snapshots a new signal into the tracking table.
        """
        try:
            if entry_price is None:
                # User did not provide price, fetch it now
                price, _ = self.market.get_crypto_data_coingecko(ticker)
                
                if not price:
                    # Fallback Yahoo
                    import yfinance as yf
                    
                    # Determine attempt order based on asset type heuristic
                    attempts = []
                    if ticker.upper() in self.KNOWN_CRYPTO:
                        attempts = [f"{ticker}-USD", ticker]
                    else:
                        # For unknown assets, assume Stock first (TICKER), then Crypto (TICKER-USD)
                        attempts = [ticker, f"{ticker}-USD"]
                    
                    for sym in attempts:
                        try:
                            t = yf.Ticker(sym)
                            hist = t.history(period="1d")
                            if not hist.empty:
                                price = hist['Close'].iloc[-1]
                                break
                        except: pass
                
                entry_price = price
            
            if not entry_price:
                logger.error(f"Auditor: Could not determine entry price for {ticker}. Aborting track.")
                return False

            data = {
                "ticker": ticker,
                "entry_price": entry_price,
                "current_price": entry_price,
                "target_price": target_price,
                "status": "OPEN",
                "pnl_percent": 0.0,
                "signal_id": signal_id
            }
            res = self.db.supabase.table("signal_tracking").insert(data).execute()
            logger.info(f"Auditor: Tracking started for {ticker} at ${entry_price:.2f}")
            return True
        except Exception as e:
            logger.error(f"Auditor Error recording {ticker}: {e}")
            return False

    # Removed duplicate audit_open_signals method.
    # The valid one is defined above (lines 24-128).

    def get_ticker_stats(self, ticker: str):
        """
        Returns performance stats for a specific ticker to inject into AI prompt.
        """
        try:
            # Fetch all closed signals for this ticker
            response = self.db.supabase.table("signal_tracking") \
                .select("status") \
                .eq("ticker", ticker) \
                .in_("status", ["WIN", "LOSS", "EXPIRED"]) \
                .execute()
            
            signals = response.data
            total = len(signals)
            
            if total < 3:
                return None # Not enough history to judge

            wins = sum(1 for s in signals if s['status'] == 'WIN')
            losses = sum(1 for s in signals if s['status'] == 'LOSS')
            
            win_rate = (wins / total * 100) if total > 0 else 0
            
            status = "NEUTRAL"
            if win_rate >= 60:
                status = "POSITIVE"
            elif win_rate <= 40:
                status = "NEGATIVE"
                
            return {
                "total": total,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 1),
                "status": status
            }
        except Exception as e:
            logger.error(f"Error fetching stats for {ticker}: {e}")
            return None

if __name__ == "__main__":
    # Test stub
    auditor = Auditor()
    print("Auditor initialized.")
