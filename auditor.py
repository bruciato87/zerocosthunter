import logging
from db_handler import DBHandler
from market_data import MarketData
import time
from datetime import datetime, timezone

logger = logging.getLogger("Auditor")

class Auditor:
    def __init__(self, market_instance=None):
        self.db = DBHandler()
        self.market = market_instance if market_instance else MarketData()
        # Configuration
        self.TAKE_PROFIT_PCT = 15.0  # +15%
        self.STOP_LOSS_PCT = -10.0   # -10%
        self.MAX_AGE_DAYS = 30       # Expire after 30 days if stagnating
        self.MIN_AGE_HOURS = 1       # Don't close signals younger than 1h (Prevent instant win/loss)
        
        # Heuristic: Known Cryptos vs Stocks
        self.KNOWN_CRYPTO = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "LINK", "LTC", "BCH", "UNI", "MATIC"}

    # ... record_signal omitted (unchanged) ...

    async def audit_open_signals(self):
        """
        Updates PnL and Status for all OPEN signals.
        [OPTIMIZED] Uses parallel price checks.
        """
        try:
            import asyncio
            # 1. Fetch OPEN signals
            response = self.db.supabase.table("signal_tracking").select("*").eq("status", "OPEN").execute()
            open_signals = response.data
            
            if not open_signals:
                logger.info("Auditor: No open signals to audit.")
                return []

            logger.info(f"Auditor: Auditing {len(open_signals)} open signals in parallel...")

            async def audit_single_signal(sig):
                ticker = sig['ticker']
                entry_price = float(sig['entry_price'])
                req_target_price = sig.get('target_price')
                target_price = float(req_target_price) if req_target_price else None
                created_at_iso = sig['created_at']
                
                try:
                    c_ts = created_at_iso.replace('Z', '+00:00')
                    created_dt = datetime.fromisoformat(c_ts)
                    now_dt = datetime.now(timezone.utc)
                    age_hours = (now_dt - created_dt).total_seconds() / 3600
                except Exception as e:
                    logger.warning(f"Auditor: Timestamp parse failed for {ticker}: {e}. Treating as NEW.")
                    age_hours = 0
                
                # 2. Get Live Price (Async)
                live_price, _ = await self.market.get_smart_price_eur_async(ticker)
                
                if not live_price:
                    return None

                # 3. Calculate PnL
                pnl_pct = ((live_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                new_status = "OPEN"
                
                # 4. Check Exit Conditions
                if age_hours >= self.MIN_AGE_HOURS:
                    if pnl_pct >= self.TAKE_PROFIT_PCT:
                        new_status = "WIN"
                    elif target_price and live_price >= target_price and pnl_pct > 0.5:
                        new_status = "WIN"
                    elif pnl_pct <= self.STOP_LOSS_PCT:
                        new_status = "LOSS"
                    elif age_hours > (self.MAX_AGE_DAYS * 24):
                        new_status = "EXPIRED"

                return {
                    "id": sig["id"],
                    "ticker": ticker,
                    "current_price": live_price,
                    "pnl_percent": round(pnl_pct, 2),
                    "status": new_status,
                    "updated_at": "now()"
                }

            # Run parallel checks
            results = await asyncio.gather(*(audit_single_signal(s) for s in open_signals))
            
            # Filter None and terminal signals for batch update
            pending_updates = [r for r in results if r is not None]
            
            # Batch update (in simple sequential loop for now as DBHandler usually handles items)
            # but ideally we collect them and do ONE DB call.
            # Auditor currently iterates anyway in step 5? No, I'll optimize it here.
            
            # Auditor typically returns terminal updates for logging
            terminal_updates = []
            for up in pending_updates:
                self.db.supabase.table("signal_tracking").update({
                    "current_price": up["current_price"],
                    "pnl_percent": up["pnl_percent"],
                    "status": up["status"],
                    "updated_at": up["updated_at"]
                }).eq("id", up["id"]).execute()
                
                if up["status"] != "OPEN":
                    # Fetch ticker for logging if needed, or pass it through
                    terminal_updates.append(up)
            
            return terminal_updates
            
            # 6. [PERFORMANCE] Batch all updates at once
            if pending_updates:
                logger.info(f"Auditor: Batching {len(pending_updates)} updates...")
                for pu in pending_updates:
                    self.db.supabase.table("signal_tracking").update(pu["data"]).eq("id", pu["id"]).execute()
                    
                    # AUTO-LEARN: Generate lesson for LOSS signals (still sequential for now)
                    if pu["new_status"] == "LOSS":
                        try:
                            from memory import Memory
                            mem = Memory()
                            
                            # Get original reasoning from memory (if it exists)
                            original_reasoning = "Segnale BUY originale - nessun dettaglio disponibile"
                            memories = mem.recall_memory(pu["ticker"], limit=1)
                            if memories and memories[0].get('reasoning'):
                                original_reasoning = memories[0]['reasoning']
                            
                            # Generate AI lesson
                            lesson = mem.generate_lesson(pu["ticker"], pu["pnl_pct"], original_reasoning)
                            
                            if lesson:
                                # Save lesson to memory table
                                mem.save_memory(
                                    ticker=pu["ticker"],
                                    event_type="lesson",
                                    reasoning=f"Trade chiuso in perdita: {pu['pnl_pct']:+.1f}%",
                                    signal_id=pu["sig"].get('signal_id'),
                                    source="auto_auditor"
                                )
                                # Update with lesson
                                mem.update_outcome(pu["sig"].get('signal_id'), pu["pnl_pct"], lesson)
                                logger.info(f"Lesson generated for {pu['ticker']}: {lesson[:50]}...")
                        except Exception as lesson_err:
                            logger.warning(f"Lesson generation failed for {pu['ticker']}: {lesson_err}")

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
                    # Fallback Yahoo with ticker_resolver
                    import yfinance as yf
                    from ticker_resolver import resolve_ticker
                    
                    try:
                        sym = resolve_ticker(ticker)
                        t = yf.Ticker(sym)
                        hist = t.history(period="1d")
                        if not hist.empty:
                            price = hist['Close'].iloc[-1]
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
