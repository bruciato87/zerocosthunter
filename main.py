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
from economist import Economist
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
    # Force re-configuration of logging to ensure visibility in Vercel/Threaded context
    # We use FORCE=True and remove existing handlers to be strictly sure
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    # Re-acquire logger after config
    logger = logging.getLogger("MainController")
    logger.setLevel(logging.INFO)
    
    # FORCE Sub-module Loggers to INFO/DEBUG
    for mod in ["hunter", "brain", "market_data", "advisor", "auditor", "economist", "whale_watcher"]:
        logging.getLogger(mod).setLevel(logging.INFO)

    # DEBUG: Use print to confirm thread stdout is alive
    print("DEBUG: Pipeline Thread Started - Stdout Check", flush=True)
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
        economist = Economist()
    # DEBUG: Use print to confirm thread stdout is alive
    print("DEBUG: Pipeline Thread Started - Stdout Check", flush=True)
    print("Hunter: Starting Zero-Cost Investment Hunter Pipeline...", flush=True)
    
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
        economist = Economist()
    except Exception as e:
        print(f"CRITICAL: Initialization failed: {e}", flush=True)
        return

    # 2. Fetch News
    news_items = hunter.fetch_news()
    if not news_items:
        print("Hunter: No news found. Exiting.", flush=True)
        return

    # 2.2 Load Portfolio
    print("Hunter: Loading Portfolio...", flush=True)
    portfolio_map = db.get_portfolio_map()
    if portfolio_map:
        print(f"Hunter: Loaded {len(portfolio_map)} holdings.", flush=True)

    # 2.5 Enrich News with Technical Data & Portfolio Context
    print("Hunter: Enriching news with Technical Data & Portfolio Context...", flush=True)
    
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
        "RNDR-USD": "RENDER-USD"
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
            
            # 3. Portfolio
            if detected_ticker in portfolio_map:
                holding = portfolio_map[detected_ticker]
                p_summary = f"OWNED {holding['quantity']} @ ${holding['avg_price']}"
                extras.append(f"Portfolio: {p_summary}")
                print(f"Hunter: Enriched {detected_ticker} with Portfolio data: {p_summary}", flush=True)

            item['summary'] += "\n\n[" + " | ".join(extras) + "]"
            print(f"Hunter: Enriched {detected_ticker} news.", flush=True)

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
                
                # 2. Create Synthetic Item
                synthetic_item = {
                    "title": f"PORTFOLIO CHECK: {fetch_ticker}",
                    "link": f"https://finance.yahoo.com/quote/{fetch_ticker}",
                    "summary": f"Routine technical check for owned asset. {tech_summary}. [Portfolio: OWNED {holding['quantity']} @ ${holding['avg_price']}]",
                    "published": "Just Now",
                    "ticker": fetch_ticker, # Use Normalized
                    "synthetic": True,
                    "source": "Portfolio Technicals"
                }
                news_items.append(synthetic_item)
                print(f"Hunter: Injected Synthetic Item for {fetch_ticker}", flush=True)
            except Exception as e:
                print(f"ERROR: Failed to generate synthetic item for {norm_p_ticker}: {e}", flush=True)
    # -------------------------------------

    # 3. Analyze with AI
    # [FEEDBACK LOOP] Fetch past performance for detected tickers
    performance_context = {}
    detected_tickers = set(i.get('ticker') for i in news_items if i.get('ticker'))
    for t in detected_tickers:
        stats = auditor.get_ticker_stats(t)
        if stats:
            performance_context[t] = stats
            print(f"Hunter: Feedback Loop: Found history for {t} -> {stats['status']} ({stats['win_rate']}%)")

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
            print(f"Insider: Found {len(social_sentiment)} trending social headlines.", flush=True)
        
        if market_mood:
            print(f"Insider: Market Mood is {market_mood.get('overall')} ({market_mood.get('crypto',{}).get('value')})", flush=True)

    # [ADVISOR] Portfolio Health Analysis
    from advisor import Advisor
    adv = Advisor()
    # Fetch current portfolio from DB for analysis
    # We use portfolio_map values (loaded earlier)
    portfolio_list = list(portfolio_map.values()) if portfolio_map else []
    advisor_analysis = adv.analyze_portfolio(portfolio_list)
    if advisor_analysis:
        print(f"Advisor: Portfolio Value ${advisor_analysis['total_value']:.2f}. Tips: {len(advisor_analysis.get('tips', []))}", flush=True)

    # [ECONOMIST] Macro Context (V4.0)
    macro_context = economist.get_macro_summary()
    print(f"Economist: Macro Context Generated. ({len(macro_context)} chars)", flush=True)

    # [WHALE WATCHER] On-Chain Context (V4.0 Phase 11)
    from whale_watcher import WhaleWatcher
    whale = WhaleWatcher()
    whale_context = whale.analyze_flow()
    
    # Safe logging
    w_lines = whale_context.splitlines()
    if len(w_lines) > 2:
        log_hint = w_lines[6].strip() if len(w_lines) > 6 and 'strategy_hint' in whale_context else ''
        print(f"WhaleWatcher: {w_lines[2].strip()} | {log_hint}", flush=True)
    else:
        print(f"WhaleWatcher: {whale_context}", flush=True)

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
    print(f"Hunter: Deduplicated News Items: {len(news_items)} -> {len(unique_news_items)}", flush=True)
    # -----------------------------------------

    print("Hunter: Analyzing news with Gemini...", flush=True)
    predictions = brain.analyze_news_batch(
        unique_news_items, 
        performance_context=performance_context, 
        insider_context=insider_context,
        portfolio_context=advisor_analysis,
        macro_context=macro_context,
        whale_context=whale_context
    )

    processed_count = 0
    
    # 3.5 Fetch User Settings for Filtering
    user_settings = db.get_settings()
    min_conf = float(user_settings.get("min_confidence", 0.70))
    only_portfolio = user_settings.get("only_portfolio", False)
    print(f"Hunter: Smart Filters Active: Min Confidence={min_conf}, Only Portfolio={only_portfolio}", flush=True)

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
            print(f"Skipped {ticker}: Confidence {confidence:.2f} < Threshold {min_conf}", flush=True)
            continue

        # FILTER 2: Portfolio Only Mode
        if only_portfolio and ticker not in portfolio_map:
            print(f"Skipped {ticker}: Portfolio Mode ON and asset not owned.", flush=True)
            continue

        # Check if recently analyzed (Same Ticker + Same Sentiment = SPAM)
        if db.check_if_analyzed_recently(ticker, sentiment):
            continue

        risk_score = pred.get("risk_score", 5)
        target_price = pred.get("target_price")
        upside_percentage = pred.get("upside_percentage", 0.0)

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
                print(f"ERROR: Failed to record signal for audit: {e}", flush=True)
        # ---------------------------

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
        processed_count += 1

    # --- AUDIT PHASE ---
    print("Hunter: Running Auditor Checkup...", flush=True)
    audit_results = auditor.audit_open_signals()
    if audit_results:
        summary_audit = "\n".join(audit_results)
        await notifier.send_alert(f"⚖️ **Auditor Monitoring Update:**\n{summary_audit}")

    db.log_system_event("INFO", "Hunter", "Pipeline Finished")
    print(f"Hunter: Pipeline finished. Processed {processed_count} actionable signals.", flush=True)

if __name__ == "__main__":
    run_pipeline()
