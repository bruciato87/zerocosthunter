import logging
import datetime
from typing import Dict
from db_handler import DBHandler
from paper_trader import PaperTrader


# Configure logging
logger = logging.getLogger(__name__)

class Sentinel:
    def __init__(self, db_handler=None):
        self.db = db_handler if db_handler else DBHandler()
        self.paper_trader = PaperTrader(self.db)


    async def check_alerts(self, market_data):
        """
        Returns a list of notification dictionaries.
        """
        notifications = []
        
        # 1. Price Alerts
        p_alerts = await self._check_price_alerts(market_data)
        notifications.extend(p_alerts)
        
        # 2. Paper Portfolio Protection (SL/TP Auto-Execute)
        await self._check_paper_protection(market_data)

        # 3. Portfolio Risks (Volatility, SL/TP Alerts)
        try:
            portfolio = self.db.get_portfolio()
            if portfolio:
                p_risks = await self._check_portfolio_risks(market_data, portfolio)
                notifications.extend(p_risks)
        except Exception as e:
            logger.error(f"Sentinel: Risk check failed: {e}")
            
        return notifications

    def get_strategic_forecast(self, market_data) -> Dict:
        """
        Level 9 Predictive: Analyze portfolio against regime and identify GAPs.
        Returns a strategic forecast report.
        """
        try:
            from strategy_manager import StrategyManager
            from advisor import Advisor
            sm = StrategyManager()
            adv = Advisor(market_instance=market_data)
            
            # 1. Determine Market Regime
            regime_data = sm.get_market_regime()
            regime = regime_data.get('regime', 'SIDEWAYS')
            
            # 2. Portfolio Gap Analysis
            portfolio = self.db.get_portfolio()
            if not portfolio:
                return {"error": "Portfolio vuoto. Nessuna analisi predittiva possibile."}
            
            analysis = adv.analyze_portfolio(portfolio)
            current_allocations = analysis.get('sector_percent', {})
            target_allocations = regime_data.get('targets', {})
            
            gaps = []
            for sector, target in target_allocations.items():
                current = current_allocations.get(sector, 0.0)
                diff = target - current
                if diff > 5.0: # Significant Underweight
                    gaps.append({
                        "sector": sector,
                        "type": "UNDERWEIGHT",
                        "gap_pct": round(diff, 1),
                        "recommendation": f"Espandi {sector} (+{diff:.1f}%) per allinearti al regime {regime}."
                    })
                elif diff < -5.0: # Significant Overweight
                    gaps.append({
                        "sector": sector,
                        "type": "OVERWEIGHT",
                        "gap_pct": round(abs(diff), 1),
                        "recommendation": f"Riduci {sector} (-{abs(diff):.1f}%) - Eccessiva esposizione per il regime {regime}."
                    })

            # 3. Correlation Risk Prediction
            tickers = [p['ticker'] for p in portfolio][:10]
            correlation_warnings = []
            if len(tickers) >= 2:
                corr_matrix = market_data.calculate_correlation_matrix(tickers)
                high_corr = corr_matrix.get('high_correlation_pairs', [])
                for t1, t2, corr in high_corr[:2]:
                    correlation_warnings.append(f"⚠️ {t1} e {t2} sono correlati al {corr:.0%}. Un ribasso colpirà entrambi contemporaneamente.")

            return {
                "regime": regime_data.get('description'),
                "risk_level": regime_data.get('risk_level'),
                "gaps": gaps,
                "correlation_warnings": correlation_warnings,
                "strategy_summary": regime_data.get('recommendation')
            }

        except Exception as e:
            logger.error(f"Sentinel: Strategic forecast failed: {e}")
            return {"error": str(e)}

    async def _check_price_alerts(self, market_data):
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
                    # Use Async Centralized Market Data
                    price, used_ticker = await market_data.get_smart_price_eur_async(ticker)
                    
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

    async def _check_portfolio_risks(self, market_data, portfolio):
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
                price, _, change_pct = await market_data.get_smart_price_eur_async(ticker, include_change=True)
                
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

                # --- B. DIGITAL TWIN PROTECTION (SL/TP Monitoring) ---
                # Check if price hits user-defined SL or TP
                stop_loss = p.get('stop_loss')
                take_profit = p.get('take_profit')
                
                if stop_loss and price <= stop_loss:
                     msg = f"🛑 **STOP LOSS HIT**: {ticker} ha toccato €{price:.2f} (SL: €{stop_loss:.2f})\n📉 Consigliata chiusura immediata."
                     notifications.append({"chat_id": chat_id, "text": msg})

                if take_profit and price >= take_profit:
                     msg = f"💰 **TAKE PROFIT HIT**: {ticker} ha raggiunto €{price:.2f} (TP: €{take_profit:.2f})\n🥂 Consigliata presa di profitto."
                     notifications.append({"chat_id": chat_id, "text": msg})

                # --- C. TRAILING STOP ASSISTANT ---
                # Trigger if Total Gain > 20%
                if avg_price > 0:
                    gain_pct = ((price - avg_price) / avg_price) * 100
                    if gain_pct > 20.0:
                        stop_price = avg_price * 1.15 # Lock in 15% gain
                        msg = f"💰 **Profit Protection**: {ticker} sta guadagnando il +{gain_pct:.1f}%!\nConsiglio: Imposta uno Stop Loss a €{stop_price:.2f} (+15%) per proteggere i profitti."
                        
                        # Only notify if no TP is set (priority to explicit TP)
                        if not take_profit:
                            # notifications.append({"chat_id": chat_id, "text": msg}) # Disabled to reduce noise if SL/TP exists
                            pass
                        notifications.append({"chat_id": chat_id, "text": msg})

            except Exception as e:
                logger.error(f"Sentinel: Risk check error for {ticker}: {e}")
        
        return notifications

    async def _check_paper_protection(self, market_data):
        """
        Checks Paper Portfolio for Stop Loss and Take Profit hits.
        Executes AUTO-SELL if triggered.
        [OPTIMIZED] Uses parallel price checks.
        """
        try:
            import asyncio
            # Get admin view of all paper positions
            positions = self.paper_trader.get_portfolio(chat_id=None)
            if not positions: return

            logger.info(f"Sentinel: Monitoring {len(positions)} paper positions for SL/TP in parallel...")

            async def check_single_position(pos):
                ticker = pos['ticker']
                sl = float(pos.get('stop_loss') or 0)
                tp = float(pos.get('take_profit') or 0)
                if sl == 0 and tp == 0: return

                try:
                    price, _ = await market_data.get_smart_price_eur_async(ticker)
                    if price <= 0: return

                    trigger = None
                    reason = ""
                    if sl > 0 and price <= sl:
                        trigger = "STOP_LOSS"
                        reason = f"Sentinel: STOP LOSS triggered at €{price:.2f} (SL: €{sl:.2f})"
                    elif tp > 0 and price >= tp:
                        trigger = "TAKE_PROFIT"
                        reason = f"Sentinel: TAKE PROFIT triggered at €{price:.2f} (TP: €{tp:.2f})"
                    
                    if trigger:
                        chat_id = pos['chat_id']
                        qty = float(pos['quantity'])
                        logger.info(f"Sentinel: Executing {trigger} for {ticker}...")
                        self.paper_trader.execute_trade(chat_id, ticker, "SELL", qty, price, reason)
                except Exception as e:
                    logger.error(f"Sentinel: Error checking paper pos {ticker}: {e}")

            # Run all checks in parallel
            await asyncio.gather(*(check_single_position(pos) for pos in positions))

        except Exception as e:
            logger.error(f"Sentinel: Paper protection scan failed: {e}")


