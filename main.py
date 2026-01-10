import os
import logging
import sys
from dotenv import load_dotenv

# Load env vars if running locally
load_dotenv()

from db_handler import DBHandler
from hunter import NewsHunter
from market_data import MarketData
from brain import Brain
from telegram_bot import TelegramNotifier
from auditor import Auditor
from auditor import Auditor
from economist import Economist
from sentinel import Sentinel
from paper_trader import PaperTrader
import re
import asyncio

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainController")

def run_pipeline():
    asyncio.run(run_async_pipeline())

async def run_async_pipeline():
    logger.info("Starting Zero-Cost Investment Hunter Pipeline...")
    
    # 0. Log Start (for Dashboard visibility even if timeout occurs)
    try:
        tmp_db = DBHandler()
        tmp_db.log_system_event("INFO", "Hunter", "Pipeline Started")
    except:
        pass

    # 1. Initialize Modules
    try:
        db = DBHandler()
        hunter = NewsHunter()
        brain = Brain()
        notifier = TelegramNotifier()
        market = MarketData()
        auditor = Auditor()
        auditor = Auditor()
        economist = Economist()
        sentinel = Sentinel()
        paper_trader = PaperTrader()
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 1.5 Sentinel Checks (Price Alerts)
    logger.info("Running Sentinel Checks...")
    try:
        notifications = sentinel.check_alerts(market)
        for n in notifications:
            await notifier.send_message(n['chat_id'], n['text'])
            logger.info(f"Sentinel: Notification sent to {n['chat_id']}")
    except Exception as e:
        logger.error(f"Sentinel Failed: {e}")

    # 2. Fetch News
    news_items = hunter.fetch_news()
    if not news_items:
        print("Hunter: No news found. Exiting.", flush=True)
        return

    # 2.2 Load Portfolio
    # 2.2 Load Portfolio
    logger.info("Loading Portfolio...")
    portfolio_map = db.get_portfolio_map()
    if portfolio_map:
        logger.info(f"Loaded {len(portfolio_map)} holdings.")

    # 2.5 Enrich News with Technical Data & Portfolio Context
    logger.info("Enriching news with Technical Data & Portfolio Context...")
    
    # ... (Monitored Tickers Setup - Unchanged) ...
    # Simple whitelist for Zero-Cost extraction (expandable)
    MONITORED_TICKERS = {
        "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
        "SOL": "SOL-USD", "SOLANA": "SOL-USD",
        "NVDA": "NVDA", "NVIDIA": "NVDA",
        "TSLA": "TSLA", "TESLA": "TSLA",
        "AAPL": "AAPL", "APPLE": "AAPL",
        "MSFT": "MSFT", "MICROSOFT": "MSFT",
        "AMZN": "AMZN", "AMAZON": "AMZN",
        "GOOG": "GOOGL", "GOOGLE": "GOOGL",
        "META": "META",
        "AMD": "AMD",
        "SPY": "SPY", "S&P 500": "SPY"
    }

    # Canonical Map for De-duplication
    CANONICAL_MAP = {
        "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
        "SOL": "SOL-USD", "SOLANA": "SOL-USD",
        "RNDR": "RENDER-USD", "RENDER": "RENDER-USD",
        "RNDR-USD": "RENDER-USD",
        "BYD": "BYDDF", # Map BYD to USD OTC (prevents Boyd Gaming mixup)
        "BYD COMPANY": "BYDDF"
    }

    # Add Portfolio Tickers to Monitored list dynamically
    for p_ticker in portfolio_map.keys():
        MONITORED_TICKERS[p_ticker] = p_ticker

    for item in news_items:
        text_content = (item.get('title', '') + " " + item.get('summary', '')).upper()
        
        # Find first matching ticker
        detected_ticker = None
        for key, symbol in MONITORED_TICKERS.items():
            # word boundary check
            if re.search(r'\b' + re.escape(key) + r'\b', text_content):
                detected_ticker = symbol
                break
        
        if detected_ticker:
            # 1. Normalize Ticker
            # A) Hardcoded Canonical Map (BTC -> BTC-USD)
            if detected_ticker in CANONICAL_MAP:
                detected_ticker = CANONICAL_MAP[detected_ticker]
            
            # B) Dynamic Portfolio Match (Catch-all for future assets)
            # If "ADA" is found but I own "ADA-USD", normalize to "ADA-USD"
            elif f"{detected_ticker}-USD" in portfolio_map:
                detected_ticker = f"{detected_ticker}-USD"
            elif f"{detected_ticker}USD" in portfolio_map:
                 detected_ticker = f"{detected_ticker}USD"
            
            # Persist normalized ticker
            item['ticker'] = detected_ticker

            extras = []
            
            # 2. Technicals
            tech_summary = market.get_technical_summary(detected_ticker)
            extras.append(f"Technical: {tech_summary}")
            
            # 2.5 Signal Intelligence Context (NEW)
            try:
                from signal_intelligence import SignalIntelligence
                si = SignalIntelligence()
                si_context = si.generate_context_for_ai(detected_ticker)
                extras.append(si_context)
            except Exception as e:
                logger.warning(f"Signal Intelligence failed for {detected_ticker}: {e}")
            
            # 3. Portfolio - with flexible matching
            # Helper: find portfolio entry with ticker variants
            def find_portfolio_entry(ticker, portfolio):
                """Match ticker to portfolio considering -USD variants."""
                if ticker in portfolio:
                    return ticker, portfolio[ticker]
                # Try without -USD suffix
                base = ticker.replace('-USD', '').replace('-EUR', '')
                if base in portfolio:
                    return base, portfolio[base]
                # Try with -USD suffix
                if f"{ticker}-USD" in portfolio:
                    return f"{ticker}-USD", portfolio[f"{ticker}-USD"]
                return None, None
            
            portfolio_ticker, holding = find_portfolio_entry(detected_ticker, portfolio_map)
            if holding:
                # Use MarketData for consistent EUR pricing (avg_price is stored in EUR)
                current_price_eur, _ = market.get_smart_price_eur(detected_ticker)
                
                pnl_str = ""
                if current_price_eur > 0 and holding['avg_price'] > 0:
                    pnl_pct = ((current_price_eur - holding['avg_price']) / holding['avg_price']) * 100
                    sign = "+" if pnl_pct >= 0 else ""
                    pnl_str = f" | PnL: {sign}{pnl_pct:.2f}%"
                    logger.info(f"PnL for {detected_ticker}: €{current_price_eur:.4f} vs avg €{holding['avg_price']:.4f} = {pnl_pct:.2f}%")

                p_summary = f"OWNED {holding['quantity']} @ €{holding['avg_price']:.2f}{pnl_str}"
                extras.append(f"Portfolio: {p_summary}")
                logger.info(f"Enriched {detected_ticker} with Portfolio data: {p_summary}")
            else:
                # CRITICAL: Explicitly tell AI that asset is NOT owned to prevent hallucinations
                extras.append("Portfolio: NOT OWNED (New Entry)")
                logger.info(f"Marked {detected_ticker} as NOT OWNED")

            item['summary'] += "\n\n[" + " | ".join(extras) + "]"
            logger.info(f"Enriched {detected_ticker} news.")

    # --- SYNTHETIC PORTFOLIO INJECTION ---
    # Ensure ALL portfolio assets are analyzed. Use CANONICAL tickers.
    found_tickers = set()
    for i in news_items:
        if i.get('ticker'):
            t = i['ticker']
            if t in CANONICAL_MAP: t = CANONICAL_MAP[t]
            found_tickers.add(t)

    for p_ticker, holding in portfolio_map.items():
        # Normalize portfolio ticker for check
        norm_p_ticker = CANONICAL_MAP.get(p_ticker, p_ticker)
        
        if norm_p_ticker not in found_tickers:
            print(f"Hunter: Portfolio Asset {norm_p_ticker} not in news. Generating Synthetic Check...", flush=True)
            try:
                # 1. Fetch Technicals using ORIGINAL ticker (market data handles aliases) or Normalized?
                # Best to use normalized if it's a standard YF ticker like BTC-USD
                fetch_ticker = norm_p_ticker
                
                tech_summary = market.get_technical_summary(fetch_ticker)
                
                # PnL Calculation for Synthetic - USE EUR PRICING (consistent with real news enrichment)
                pnl_str = ""
                current_price_eur, _ = market.get_smart_price_eur(fetch_ticker)
                
                if current_price_eur > 0 and holding['avg_price'] > 0:
                    pnl_pct = ((current_price_eur - holding['avg_price']) / holding['avg_price']) * 100
                    sign = "+" if pnl_pct >= 0 else ""
                    pnl_str = f" | PnL: {sign}{pnl_pct:.2f}%"
                    logger.info(f"Synthetic PnL for {fetch_ticker}: €{current_price_eur:.4f} vs avg €{holding['avg_price']:.4f} = {pnl_pct:.2f}%")

                # 2. Create Synthetic Item
                synthetic_item = {
                    "title": f"PORTFOLIO CHECK: {fetch_ticker}",
                    "link": f"https://finance.yahoo.com/quote/{fetch_ticker}",
                    "summary": f"Routine technical check for owned asset. {tech_summary}. [Portfolio: OWNED {holding['quantity']} @ €{holding['avg_price']:.2f}{pnl_str}]",
                    "published": "Just Now",
                    "ticker": fetch_ticker, # Use Normalized
                    "synthetic": True,
                    "source": "Portfolio Technicals"
                }
                news_items.append(synthetic_item)
                logger.info(f"Injected Synthetic Item for {fetch_ticker}")
            except Exception as e:
                logger.error(f"Failed to generate synthetic item for {norm_p_ticker}: {e}")
    # -------------------------------------

    # 3. Analyze with AI
    # [FEEDBACK LOOP] Fetch past performance for detected tickers
    performance_context = {}
    detected_tickers = set(i.get('ticker') for i in news_items if i.get('ticker'))
    for t in detected_tickers:
        stats = auditor.get_ticker_stats(t)
        if stats:
            performance_context[t] = stats
            logger.info(f"Feedback Loop: Found history for {t} -> {stats['status']} ({stats['win_rate']}%)")

    # [INSIDER] Fetch Market Mood & Social Sentiment
    from insider import Insider
    ins = Insider()
    market_mood = ins.get_market_mood()
    social_sentiment = ins.get_social_sentiment()
    
    insider_context = None
    if market_mood or social_sentiment:
        insider_context = market_mood if market_mood else {}
        if social_sentiment:
            insider_context['social'] = social_sentiment
            logger.info(f"Insider: Found {len(social_sentiment)} trending social headlines.")
        
        if market_mood:
            logger.info(f"Insider: Market Mood is {market_mood.get('overall')} ({market_mood.get('crypto',{}).get('value')})")

    # [ADVISOR] Portfolio Health Analysis
    from advisor import Advisor
    adv = Advisor()
    # Fetch current portfolio from DB for analysis
    # We use portfolio_map values (loaded earlier)
    portfolio_list = list(portfolio_map.values()) if portfolio_map else []
    advisor_analysis = adv.analyze_portfolio(portfolio_list)
    if advisor_analysis:
        logger.info(f"Advisor: Portfolio Value ${advisor_analysis['total_value']:.2f}. Tips: {len(advisor_analysis.get('tips', []))}")

    # [ECONOMIST] Macro Context (V4.0)
    macro_context = economist.get_macro_summary()
    logger.info(f"Economist: Macro Context Generated. ({len(macro_context)} chars)")

    # [WHALE WATCHER] On-Chain Context (V4.0 Phase 11)
    from whale_watcher import WhaleWatcher
    whale = WhaleWatcher()
    whale_context = whale.analyze_flow()
    
    # Safe logging
    w_lines = whale_context.splitlines()
    if len(w_lines) > 2:
        log_hint = w_lines[6].strip() if len(w_lines) > 6 and 'strategy_hint' in whale_context else ''
        logger.info(f"WhaleWatcher: {w_lines[2].strip()} | {log_hint}")
    else:
        logger.info(f"WhaleWatcher: {whale_context}")

    # --- PRE-BRAIN DEDUPLICATION & MERGING ---
    merged_map = {}
    for item in news_items:
        t = item.get('ticker')
        if not t: continue
        
        if t in merged_map:
            existing = merged_map[t]
            existing['summary'] += f"\n\n--- ADDITIONAL CONTEXT ---\n{item['title']}: {item['summary']}"
            if not item.get('synthetic', False):
                existing['synthetic'] = False
                existing['source'] = item.get('source', 'News') 
        else:
            merged_map[t] = item

    unique_news_items = list(merged_map.values())
    logger.info(f"Deduplicated News Items: {len(news_items)} -> {len(unique_news_items)}")
    # -----------------------------------------

    logger.info("Analyzing news with Gemini...")
    try:
        predictions = brain.analyze_news_batch(
            unique_news_items, 
            performance_context=performance_context, 
            insider_context=insider_context,
            portfolio_context=advisor_analysis,
            macro_context=macro_context,
            whale_context=whale_context
        )
        logger.info(f"Gemini analysis complete. Received {len(predictions)} predictions.")
    except Exception as e:
        logger.error(f"CRITICAL: Gemini analysis FAILED: {e}")
        predictions = []

    processed_count = 0
    
    # 3.5 Fetch User Settings for Filtering
    user_settings = db.get_settings()
    min_conf = float(user_settings.get("min_confidence", 0.70))
    only_portfolio = user_settings.get("only_portfolio", False)
    logger.info(f"Smart Filters Active: Min Confidence={min_conf}, Only Portfolio={only_portfolio}")

    # 4. Process Predictions
    for pred in predictions:
        ticker = pred.get("ticker")
        sentiment = pred.get("sentiment")
        reasoning = pred.get("reasoning")
        confidence = float(pred.get("confidence", 0.0))
        source = pred.get("source", "Unknown")

        if not ticker: 
            continue

        # FILTER 1: Confidence Score
        if confidence < min_conf:
            logger.info(f"Skipped {ticker}: Confidence {confidence:.2f} < Threshold {min_conf}")
            continue

        # FILTER 2: Portfolio Only Mode
        if only_portfolio and ticker not in portfolio_map:
            logger.info(f"Skipped {ticker}: Portfolio Mode ON and asset not owned.")
            continue

        # FILTER 3: Logic Check (Cannot SELL/HOLD/ACCUMULATE what you don't own)
        # Use flexible matching for -USD variants (RENDER-USD matches RENDER)
        def is_owned(t, pmap):
            if t in pmap:
                return True
            base = t.replace('-USD', '').replace('-EUR', '')
            if base in pmap:
                return True
            if f"{t}-USD" in pmap:
                return True
            return False
        
        if sentiment in ["SELL", "PANIC SELL", "HOLD", "ACCUMULATE"] and not is_owned(ticker, portfolio_map):
            logger.warning(f"Skipped {ticker}: Ignored {sentiment} signal for unowned asset.")
            continue

        # FILTER 4: Market Hours (Stock signals blocked when market closed)
        def is_crypto(t):
            """Check if ticker is a crypto asset (always tradeable 24/7)"""
            crypto_indicators = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE', 'RENDER', 'MATIC', 'DOT', 'AVAX', 'LINK']
            t_upper = t.upper()
            for c in crypto_indicators:
                if c in t_upper:
                    return True
            if '-USD' in t_upper or '-EUR' in t_upper:
                return True
            return False
        
        if not is_crypto(ticker):
            # It's a stock - check if market is open
            try:
                eco = Economist()
                market_status = eco.get_market_status()
                
                # Check if this is a US stock (most common) or EU stock
                is_eu_stock = ticker.endswith('.DE') or ticker.endswith('.MI') or ticker.endswith('.FRA')
                
                if is_eu_stock and '🔴' in market_status['eu_stocks']:
                    logger.info(f"Skipped {ticker}: EU market closed - no stock signals when closed")
                    continue
                elif not is_eu_stock and '🔴' in market_status['us_stocks']:
                    logger.info(f"Skipped {ticker}: US market closed - no stock signals when closed")
                    continue
            except Exception as e:
                logger.warning(f"Market hours check failed for {ticker}: {e}")

        # Check if recently analyzed (Same Ticker + Same Sentiment = SPAM)
        if db.check_if_analyzed_recently(ticker, sentiment):
            continue
        
        # --- SIGNAL INTELLIGENCE LAYER (NEW) ---
        # Apply advanced filtering and adjustments
        try:
            from signal_intelligence import SignalIntelligence
            si = SignalIntelligence()
            logger.info(f"Signal Intelligence: Analyzing {ticker} ({sentiment} @ {confidence:.2f})")
            si_analysis = si.analyze_signal(ticker, sentiment, confidence)
            
            # Apply adjustments
            original_sentiment = sentiment
            sentiment = si_analysis.get('adjusted_sentiment', sentiment)
            confidence = si_analysis.get('adjusted_confidence', confidence)
            
            # Log any actions taken
            for action in si_analysis.get('actions', []):
                logger.info(f"Signal Intelligence [{ticker}]: {action}")
                reasoning += f" [SI: {action}]"
            
            # Log warnings too
            for warning in si_analysis.get('warnings', []):
                logger.info(f"Signal Intelligence [{ticker}] WARNING: {warning}")
            
            # Re-check confidence after adjustment
            if confidence < min_conf:
                logger.info(f"Skipped {ticker}: SI adjusted confidence {confidence:.2f} < Threshold {min_conf}")
                continue
            
            # If sentiment changed to WAIT/HOLD, skip notification
            if sentiment in ['WAIT', 'HOLD'] and original_sentiment in ['BUY', 'ACCUMULATE']:
                logger.info(f"Downgraded {ticker}: {original_sentiment} -> {sentiment} (not notifying)")
                continue
            
            logger.info(f"Signal Intelligence [{ticker}]: PASSED - {sentiment} @ {confidence:.2f}")
                
        except Exception as e:
            logger.warning(f"Signal Intelligence failed for {ticker}: {e}")
        # -----------------------------------------

        risk_score = pred.get("risk_score", 5)
        target_price = pred.get("target_price")
        upside_percentage = pred.get("upside_percentage", 0.0)
        
        # --- TARGET PRICE VALIDATION ---
        # If AI returns absurd target (>100% above current price), recalculate from upside
        try:
            if target_price and upside_percentage > 0:
                clean_tp = re.sub(r'[^\d.]', '', str(target_price))
                tp_float = float(clean_tp) if clean_tp else 0
                
                # Get current price for validation
                current_price, _ = market.get_smart_price_eur(ticker)
                
                if current_price > 0 and tp_float > 0:
                    # Check if target is absurdly high (>100% gain)
                    implied_gain = (tp_float - current_price) / current_price * 100
                    if implied_gain > 100:  # More than 100% gain is suspicious
                        # Recalculate from upside percentage
                        corrected_tp = current_price * (1 + upside_percentage / 100)
                        target_price = f"€{corrected_tp:.2f}"
                        logger.warning(f"Target price corrected for {ticker}: {tp_float} -> {corrected_tp:.2f} (used upside {upside_percentage}%)")
        except Exception as e:
            logger.warning(f"Target price validation failed for {ticker}: {e}")

        # 5. Log to DB and Notify
        signal_id = db.log_prediction(ticker, sentiment, reasoning, reasoning, confidence, source, risk_score, target_price, upside_percentage)
        
        # --- AUDITOR INTEGRATION ---
        # If BUY/ACCUMULATE signal, start tracking performance
        if sentiment in ["BUY", "ACCUMULATE"] and signal_id:
            try:
                # Parse Target Price string to float (remove symbols)
                tp_float = None
                if target_price:
                    # re is imported globally
                    clean_tp = re.sub(r'[^\d.]', '', str(target_price))
                    if clean_tp:
                        tp_float = float(clean_tp)
                
                auditor.record_signal(ticker, signal_id=signal_id, target_price=tp_float)
            except Exception as e:
                logger.error(f"Failed to record signal for audit: {e}")
            except Exception as e:
                logger.error(f"Failed to record signal for audit: {e}")
        # ---------------------------

        # --- PHASE 14: PAPER TRADER EXECUTION ---
        # Execute in Lab if Confidence is High (Validation Mode)
        if hasattr(paper_trader, 'execute_trade'):
            try:
                # Decide trade size: Fixed €1000 for simulation consistency
                # Or dynamic based on "risk_score"
                trade_size_eur = 1000.0
                
                # Fetch fresh price for execution accuracy
                # (We have technical_summary but not clean float price here easily, assume market_data is fast)
                p_price, _ = market.get_smart_price_eur(ticker)
                
                if p_price > 0:
                    qty = trade_size_eur / p_price
                    
                    # Log buy or sell
                    if sentiment in ["BUY", "ACCUMULATE"]:
                         paper_trader.execute_trade(999999999, ticker, "BUY", qty, p_price, f"Auto-Signal: {confidence:.2f}")
                    elif sentiment in ["SELL"]:
                         # For sell, we need to know how much we have. PaperTrader handles logic?
                         # The execute_trade SELL logic handles partials.
                         # Let's try to sell 100% of holdings if Panic Sell.
                         paper_trader.execute_trade(999999999, ticker, "SELL", 0, p_price, "Auto-Signal Sell")
                         # Note: quantity=0 in SELL logic needs update? 
                         # Actually checking paper_trader logic: it expects quantity.
                         # Better: Let PaperTrader handle 'SELL ALL' if quantity is generic or explicit method.
                         # For V1: Simple Buy testing.
            except Exception as e:
                logger.error(f"Paper Trader Failed: {e}")
        # ----------------------------------------

        # Format Alert
        asset_type = pred.get("asset_type", "Asset")
        icon = "🟢" if sentiment in ["BUY", "ACCUMULATE"] else "🔴" if sentiment in ["SELL", "PANIC SELL"] else "⚪"
        
        # Build "Prophet" Badge
        prophet_badge = ""
        if target_price:
             prophet_badge = f"\n🎯 **Target:** {target_price}"
             if upside_percentage > 0:
                 prophet_badge += f" (Up +{upside_percentage}%)"
             prophet_badge += f"\n🎲 **Risk Score:** {risk_score}/10"

        alert_msg = (
            f"{icon} **Signal Detected: {ticker} ({asset_type})**\n"
            f"**Action:** {sentiment}\n"
            f"**Confidence:** {int(confidence * 100)}%{prophet_badge}\n\n"
            f"**Reasoning:** {reasoning}\n"
            f"**Source:** {source}"
        )
        
        await notifier.send_alert(alert_msg)
        
        # Save to Memory (Neuro-Link) for historical recall
        try:
            from memory import Memory
            mem = Memory()
            mem.save_memory(
                ticker=ticker,
                event_type="signal",
                reasoning=reasoning,
                sentiment=sentiment,
                confidence=confidence,
                target_price=float(target_price.replace('$', '').replace('€', '').replace(',', '')) if target_price else None,
                risk_score=risk_score,
                signal_id=signal_id,
                source=source
            )
        except Exception as e:
            logger.warning(f"Memory save failed for {ticker}: {e}")
        
        processed_count += 1

    # --- AUDIT PHASE ---
    logger.info("Running Auditor Checkup...")
    audit_results = auditor.audit_open_signals()
    if audit_results:
        summary_audit = "\n".join(audit_results)
        await notifier.send_alert(f"⚖️ **Auditor Monitoring Update:**\n{summary_audit}")

    # --- MAINTENANCE PHASE (Storage Monitoring) ---
    try:
        from db_maintenance import DBMaintenance
        maint = DBMaintenance()
        health = maint.check_storage_health()
        
        if health["status"] == "critical":
            # Auto-cleanup and notify
            deleted = maint.cleanup_old_records(force=True)
            total_deleted = sum(v for v in deleted.values() if v > 0)
            await notifier.send_alert(f"⚠️ **Storage Alert:**\n{health['message']}\n🧹 Auto-cleaned {total_deleted} old records.")
        elif health["status"] == "warning":
            await notifier.send_alert(f"⚡ **Storage Warning:**\n{health['message']}")
        
        logger.info(f"Maintenance: {health['message']}")
    except Exception as e:
        logger.warning(f"Maintenance check failed: {e}")

    db.log_system_event("INFO", "Hunter", "Pipeline Finished")
    logger.info(f"Pipeline finished. Processed {processed_count} actionable signals.")
    
    # Send completion notification to Telegram
    try:
        if processed_count > 0:
            await notifier.send_alert(f"✅ **Caccia Completata!**\n📊 Processati {processed_count} segnali validi.")
        else:
            await notifier.send_alert("✅ **Caccia Completata!**\n🔍 Nessun nuovo segnale significativo trovato.\n(I duplicati e quelli sotto la soglia sono stati filtrati)")
    except Exception as e:
        logger.warning(f"Failed to send completion notification: {e}")

if __name__ == "__main__":
    # Scheduled / Manual CLI Execution
    # MUST acquire lock to avoid overlap with Webhook hunts
    import time
    
    # Generate Unique ID for this run
    request_id = f"SCHEDULED_{int(time.time())}"
    
    try:
        db = DBHandler()
        # Try to acquire lock
        if db.acquire_hunt_lock(request_id=request_id, expiry_minutes=5):
            try:
                run_pipeline()
            finally:
                db.release_hunt_lock(request_id=request_id)
        else:
            logger.warning(f"Scheduled Hunt Skipped: System is BUSY (Locked by another process). Request ID: {request_id}")
    except Exception as e:
        logger.error(f"Scheduled Execution Failed to Initialize: {e}")
        # Fallback: Try to run anyway if DB failed? No, safer to fail.
