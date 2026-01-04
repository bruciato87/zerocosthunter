import logging
import datetime
from db_handler import DBHandler

# Configure logging
logger = logging.getLogger(__name__)

class Sentinel:
    def __init__(self, db_handler=None):
        self.db = db_handler if db_handler else DBHandler()

    def check_alerts(self, market_data):
        """
        Returns a list of notification dictionaries.
        """
        notifications = []
        
        # 1. Price Alerts
        notifications.extend(self._check_price_alerts(market_data))
        
        # 2. Volatility Breaker & Trailing Stop (Portfolio Scan)
        # We need portfolio data. 
        # Sentinel checks this independently of "alerts" table, 
        # but needs access to portfolio table.
        try:
            portfolio = self.db.get_portfolio() # Fetch all holdings
            if portfolio:
                notifications.extend(self._check_portfolio_risks(market_data, portfolio))
        except Exception as e:
            logger.error(f"Sentinel: Portfolio check failed: {e}")
            
        return notifications

    def _check_price_alerts(self, market_data):
        notifications = []
        try:
            active_alerts = self.db.get_active_alerts()
            if not active_alerts:
                return []


            logger.info(f"Sentinel: Checking {len(active_alerts)} active alerts...")

            # Optimization: Group by ticker to minimize API calls
            ticker_groups = {}
            for alert in active_alerts:
                base_ticker = alert['ticker']
                if base_ticker not in ticker_groups:
                    ticker_groups[base_ticker] = []
                ticker_groups[base_ticker].append(alert)

            for ticker, alerts in ticker_groups.items():
                try:
                    # Use Centralized Market Data
                    price, used_ticker = market_data.get_smart_price_eur(ticker)
                    
                    if price <= 0:
                        logger.warning(f"Sentinel: Could not fetch price for {ticker}")
                        continue

                    for alert in alerts:
                        condition = alert['condition'] # ABOVE (>), BELOW (<)
                        threshold = alert['price_threshold']
                        alert_id = alert['id']
                        chat_id = alert['chat_id']

                        triggered = False
                        msg = ""

                        if condition == "ABOVE" and price > threshold:
                            triggered = True
                            msg = f"🚨 **ALERT:** {ticker} è sopra €{threshold}!\n💰 Prezzo Attuale: €{price:.2f}"
                        elif condition == "BELOW" and price < threshold:
                            triggered = True
                            msg = f"🚨 **ALERT:** {ticker} è sotto €{threshold}!\n💰 Prezzo Attuale: €{price:.2f}"

                        if triggered:
                            logger.info(f"Sentinel: Alert Triggered for {ticker} ({condition} {threshold})")
                            # Mark as inactive in DB
                            self.db.deactivate_alert(alert_id, trigger_msg=f"Triggered at €{price}")
                            
                            notifications.append({
                                "chat_id": chat_id,
                                "text": msg
                            })

                except Exception as e:
                    logger.error(f"Sentinel: Error checking ticker {ticker}: {e}")

        except Exception as e:
            logger.error(f"Sentinel: Error in _check_price_alerts: {e}")
        
        return notifications

    def _check_portfolio_risks(self, market_data, portfolio):
        """
        Scans portfolio for:
        1. Volatility Breaker: Daily drop > 5%
        2. Trailing Stop: If Gain > 20%, suggest Stop at Gain 15%
        """
        notifications = []
        # optimization: get unique tickers
        unique_tickers = list(set([p['ticker'] for p in portfolio]))
        
        # We need to notify the 'owner'. Since we are single-user focused for now,
        # we can iterate per ticker but we need to map back to chat_id. 
        # For simplicity in this loop, let's group by chat_id/user_id first?
        # A simpler approach: Iterate portfolio items.
        
        for p in portfolio:
            ticker = p['ticker']
            chat_id = p.get('chat_id') 
            # If DBHandler.get_portfolio uses user_id, ensure we have a way to message.
            # Assuming 'chat_id' is enriched or we use the user's main ID. 
            # If not present, we skip (or log).
            if not chat_id: continue

            try:
                # 1. Fetch Price & Change (Strict EUR)
                price, _, change_pct = market_data.get_smart_price_eur(ticker, include_change=True)
                
                if price <= 0: continue

                avg_price = p['avg_price']
                
                # --- A. VOLATILITY BREAKER ---
                # Trigger if Daily Drop > 5%
                if change_pct < -5.0:
                     msg = f"📉 **Volatility Breaker**: {ticker} è crollato del {change_pct:.1f}% oggi!\nValuta se vendere o accumulare."
                     # Deduplication needed (don't spam every hour).
                     # Simple dedup: Only trigger if we haven't logged "VOL_BREAKER" today?
                     # For V1, we accept repetition or user ignores.
                     # Better: Check DB logs? Too expensive.
                     # Let's send it. Telegram bot is noisy by design for Sentinel.
                     notifications.append({"chat_id": chat_id, "text": msg})

                # --- B. TRAILING STOP ASSISTANT ---
                # Trigger if Total Gain > 20%
                if avg_price > 0:
                    gain_pct = ((price - avg_price) / avg_price) * 100
                    if gain_pct > 20.0:
                        stop_price = avg_price * 1.15 # Lock in 15% gain
                        msg = f"💰 **Profit Protection**: {ticker} sta guadagnando il +{gain_pct:.1f}%!\nConsiglio: Imposta uno Stop Loss a €{stop_price:.2f} (+15%) per proteggere i profitti."
                        
                        # Dedup: Only notify if we hit new milestones? 
                        # For V1 safety, notify. User can manage.
                        notifications.append({"chat_id": chat_id, "text": msg})

            except Exception as e:
                logger.error(f"Sentinel: Risk check error for {ticker}: {e}")
        
        return notifications

