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
        Checks all active alerts against current market prices.
        Returns a list of notification dictionaries.
        """
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
            logger.error(f"Sentinel: Error in check_alerts: {e}")
        
        return notifications
