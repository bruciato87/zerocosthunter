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
from advisor import Advisor
from signal_intelligence import SignalIntelligence
from sentinel import Sentinel
from paper_trader import PaperTrader
from ml_predictor import MLPredictor
from social_scraper import SocialScraper
from onchain_watcher import OnChainWatcher
from insider import Insider
from whale_watcher import WhaleWatcher
from market_regime import MarketRegimeClassifier
from rebalancer import Rebalancer
from pulse_hunter import PulseHunter
from consensus_engine import ConsensusEngine
from run_observability import RunObservability
import re
import asyncio
from datetime import datetime, timezone

# [PHASE C.6] Global fix for nested asyncio loops (GH Actions)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MainController")


def _safe_parse_published_datetime(item):
    """Parse published datetime fields and always return a timezone-aware datetime or None."""
    if not isinstance(item, dict):
        return None

    pub_dt = item.get("published_datetime")
    if isinstance(pub_dt, datetime):
        return pub_dt if pub_dt.tzinfo else pub_dt.replace(tzinfo=timezone.utc)

    raw_published = item.get("published")
    if not raw_published:
        return None

    try:
        from dateutil import parser

        parsed = parser.parse(raw_published)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.debug(f"Freshness: invalid published date '{raw_published}': {e}")
        return None


def _safe_total_tokens_from_usage(usage):
    """Extract a friendly token counter string from usage payloads."""
    try:
        if isinstance(usage, dict):
            total_tok = usage.get("total_tokens", "?")
            if total_tok == "?":
                return "Direct"
            if "FAILED" in str(total_tok):
                return "Exhausted (429)"
            return str(total_tok)
        return str(usage)
    except Exception as e:
        logger.debug(f"Failed to parse usage tokens: {e}")
        return "?"

def _truncate_text(text, max_len=260):
    """Keep Telegram content compact and readable."""
    if text is None:
        return ""
    s = " ".join(str(text).strip().split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"

def _extract_dissent_snippet(full_debate: str) -> str:
    """Extract and shorten dissent note from council debate."""
    if not full_debate:
        return ""
    m = re.search(r"⚠️ \*\*Dissent[^\n]*\*\*:?\s*(.*)", full_debate, flags=re.IGNORECASE)
    if not m:
        return ""
    return _truncate_text(m.group(1), max_len=180)

def _localize_council_summary(summary: str) -> str:
    s = str(summary or "")
    return (
        s.replace("UNANIMOUS VERDICT", "VERDETTO UNANIME")
         .replace("MAJORITY VERDICT", "VERDETTO DI MAGGIORANZA")
         .replace("DISPUTED VERDICT", "VERDETTO CONTESO")
         .replace("OWNED_ASSET", "ASSET GIÀ IN PORTAFOGLIO")
    )

def _build_hunt_digest_message(
    *,
    analyzed_items: int,
    total_time: float,
    ai_footer: str,
    processed_count: int,
    signal_cards: list,
    propagated_cards: list,
    watchdog_report: str,
    audit_report: str,
    maintenance_report: str,
    flash_tip: str,
    sector_signals: list,
    sentinel_cards: list,
    ml_training_note: str,
):
    """Build a single Telegram message for the entire hunt run."""
    sections = []
    header = f"🏹 **Hunt completata**\n🔍 Analizzati {analyzed_items} item in {total_time:.1f}s."
    if ai_footer:
        header += f"\n{ai_footer}"
    header += f"\n🎯 Segnali validati: {processed_count}"
    sections.append(header)

    if signal_cards:
        body = "\n\n──────────\n\n".join(signal_cards[:8])
        if len(signal_cards) > 8:
            body += f"\n\n…altri {len(signal_cards) - 8} segnali non mostrati."
        sections.append(f"📌 **Segnali Hunt**\n{body}")
    else:
        sections.append("📌 **Segnali Hunt**\nNessun nuovo segnale significativo.")

    if propagated_cards:
        prop = "\n".join(f"• {c}" for c in propagated_cards[:10])
        if len(propagated_cards) > 10:
            prop += f"\n• …altri {len(propagated_cards) - 10} segnali correlati."
        sections.append(f"🔗 **Segnali Correlati**\n{prop}")

    if watchdog_report:
        sections.append(watchdog_report)
    if audit_report:
        sections.append(audit_report)
    if maintenance_report:
        sections.append(maintenance_report)
    if sentinel_cards:
        sent = "\n".join(f"• {_truncate_text(s, 180)}" for s in sentinel_cards[:8])
        if len(sentinel_cards) > 8:
            sent += f"\n• …altri {len(sentinel_cards) - 8} avvisi."
        sections.append(f"🚨 **Avvisi Sentinel**\n{sent}")
    if ml_training_note:
        sections.append(ml_training_note)
    if sector_signals:
        sections.append("📈 **Sector Rotation**\n" + "\n".join(sector_signals))
    if flash_tip:
        sections.append(f"💡 {flash_tip}")

    return "\n\n".join(s for s in sections if s)

def _split_telegram_message(text: str, max_len: int = 3800) -> list:
    """
    Split long telegram text without dropping content.
    Prefers section boundaries, then line boundaries, then hard slicing.
    """
    if not text:
        return [""]
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""

    def _flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    blocks = text.split("\n\n")
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_len:
            current = candidate
            continue

        _flush()
        if len(block) <= max_len:
            current = block
            continue

        # Oversized block: split by lines first.
        line_buf = ""
        for line in block.split("\n"):
            line_candidate = line if not line_buf else f"{line_buf}\n{line}"
            if len(line_candidate) <= max_len:
                line_buf = line_candidate
                continue
            if line_buf:
                chunks.append(line_buf)
                line_buf = ""
            if len(line) <= max_len:
                line_buf = line
                continue
            # Single oversize line: hard split.
            start = 0
            while start < len(line):
                chunks.append(line[start:start + max_len])
                start += max_len
        if line_buf:
            chunks.append(line_buf)

    _flush()
    return chunks

def format_alert_msg(ticker, sentiment, confidence, reasoning, source, pred, stop_loss, take_profit, critic_score, critic_reasoning, council_summary, consensus_data=None, ai_footer=None):
    """
    Build a compact Italian signal card for the consolidated hunt digest.
    """
    asset_type = pred.get("asset_type", "Asset")
    icon = "🟢" if sentiment in ["BUY", "ACCUMULATE"] else "🔴" if sentiment in ["SELL", "PANIC SELL"] else "⚪"
    target_price = pred.get("target_price")
    upside_percentage = pred.get("upside_percentage", 0)
    risk_score = pred.get("risk_score", 5)
    consensus_action = consensus_data.get("final_action", sentiment) if consensus_data else sentiment

    lines = [
        f"{icon} **{ticker}** ({asset_type})",
        f"⚖️ Azione consenso: **{consensus_action}**",
        f"🤖 Hunter: **{sentiment}** ({int(confidence * 100)}%)",
    ]
    if target_price:
        target_line = f"🎯 Target: {target_price}"
        if upside_percentage > 0:
            target_line += f" (+{upside_percentage}%)"
        lines.append(target_line)
    lines.append(f"🎲 Rischio: {risk_score}/10")
    if stop_loss:
        lines.append(f"🛑 Stop loss: €{stop_loss}")
    if take_profit:
        lines.append(f"💰 Take profit: €{take_profit}")

    # Keep full reasoning text as requested; do not truncate per-signal motivation.
    lines.append(f"💡 Motivo: {str(reasoning or '').strip()}")
    if critic_reasoning:
        lines.append(f"🛡️ Critic: {_truncate_text(critic_reasoning, 180)}")
    if council_summary:
        lines.append(f"🏛️ Council: {_localize_council_summary(council_summary)}")
        dissent = _extract_dissent_snippet(pred.get("council_full_debate", ""))
        if dissent:
            lines.append(f"⚠️ Dissent: {dissent}")
    else:
        lines.append("🏛️ Council: non convocato (segnale non idoneo al dibattito ad alta confidenza)")
    if source:
        lines.append(f"📰 Fonte: {_truncate_text(source, 90)}")

    return "\n".join(lines)

def run_pipeline():
    asyncio.run(run_async_pipeline())

async def run_async_pipeline():
    logger.info("Starting Zero-Cost Investment Hunter Pipeline...")
    
    # Generate unique run_id for tracking API calls this run
    import time
    run_id = f"RUN_{int(time.time())}"
    run_observer = RunObservability(
        "hunt",
        run_id=run_id,
        context={"entrypoint": "main.run_async_pipeline"},
    )
    openrouter_this_run = 0
    
    # 0. Log Start (for Dashboard visibility even if timeout occurs)
    try:
        tmp_db = DBHandler()
        tmp_db.log_system_event("INFO", "Hunter", "Pipeline Started")
        
        # --- API USAGE REPORT AT START ---
        api_usage = tmp_db.get_api_usage()
        last_model = api_usage.get('last_model', 'None (first run)')
        models_used = api_usage.get('models', {})
        
        logger.info(f"API Usage Report [START]:")
        logger.info(f"   Date: {api_usage.get('date')}")
        logger.info(f"   Last Model: {last_model}")
        logger.info(f"   OpenRouter calls today: {api_usage.get('openrouter', 0)}")
        if models_used:
            for model, count in list(models_used.items())[:3]:  # Show top 3 models
                model_short = model.split('/')[-1].replace(':free', '')
                logger.info(f"      └ {model_short}: {count} calls")
        logger.info(f"   Reset at: {api_usage.get('reset_at_local', 'N/A')} ({api_usage.get('hours_until_reset', 0):.1f}h from now)")
        logger.info(f"   Run ID: {run_id}")
    except Exception as e:
        logger.debug(f"Startup API usage report skipped: {e}")

    # 1. Initialize Modules
    try:
        db = DBHandler()
        hunter = NewsHunter()
        brain = Brain()
        # Store run_id in brain for later use
        brain.current_run_id = run_id
        notifier = TelegramNotifier()
        market = MarketData()
        auditor = Auditor(market_instance=market)
        adv = Advisor(market_instance=market)
        si = SignalIntelligence(market_instance=market, advisor_instance=adv)
        economist = Economist()
        sentinel = Sentinel()
        paper_trader = PaperTrader()
        ml_predictor = MLPredictor()
        consensus_engine = ConsensusEngine()
        
        # Oracle Modules (Phase B)
        social_scraper = SocialScraper()
        onchain_watcher = OnChainWatcher()
        pulse = PulseHunter(market_instance=market)
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        run_observer.add_error("initialization", e)
        run_observer.finalize(
            status="error",
            summary="Hunt initialization failed before pipeline execution.",
        )
        return

    # Consolidated telegram delivery (single message per hunt run)
    sentinel_cards = []
    signal_cards = []
    propagated_cards = []
    watchdog_report = ""
    audit_report = ""
    maintenance_report = ""
    ml_training_note = ""

    # 1.5 Sentinel Checks (Price Alerts)
    logger.info("Running Sentinel Checks...")
    try:
        notifications = await sentinel.check_alerts(market)
        for n in notifications:
            txt = n.get("text", "")
            if txt:
                sentinel_cards.append(txt)
            logger.info(f"Sentinel: queued notification for consolidated hunt digest")
    except Exception as e:
        logger.error(f"Sentinel Failed: {e}")

    # 1.6 Weekly ML Training (Sundays only, 18:00 UTC)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if now.weekday() == 6 and now.hour == 18:  # Sunday 18:00 UTC
        logger.info("Weekly ML Training Check...")
        try:
            ml_predictor = MLPredictor()
            if not ml_predictor.is_ml_ready or ml_predictor.get_training_data_count() > (ml_predictor.get_dashboard_stats().get('training_samples', 0) + 5):
                logger.info("ML: New data available, retraining model...")
                if ml_predictor.train():
                    logger.info("ML: Weekly training completed successfully!")
                    ml_training_note = "🤖 **Training ML completato**\n📊 Il modello è stato aggiornato con i nuovi dati."
        except Exception as e:
            logger.error(f"ML Weekly Training Failed: {e}")

    # 2. Fetch News
    import time as timing_module
    _run_start_time = timing_module.time()
    _news_fetch_start = timing_module.time()
    
    news_items = hunter.fetch_news()
    _news_fetch_time = timing_module.time() - _news_fetch_start
    
    if not news_items:
        logger.info("Hunter: No fresh RSS news found. Proceeding with Pulse/Synthetic checks.")
        news_items = []

    # 2.2 Load Portfolio
    # 2.2 Get Last Run Time for Freshness Filter
    last_run_time = None
    try:
        settings = db.get_settings()
        last_run_str = settings.get("last_successful_hunt_ts")
        if last_run_str:
            from datetime import datetime
            last_run_time = datetime.fromisoformat(last_run_str.replace('Z', '+00:00'))
            logger.info(f"FRESHNESS: Filtering news published before {last_run_time}")
    except Exception as e:
        logger.warning(f"Could not fetch last run time: {e}")

    # Freshness Filter Loop
    fresh_news = []
    skipped_count = 0
    
    if last_run_time:
        for item in news_items:
            # Helper to get datetime
            pub_dt = _safe_parse_published_datetime(item)
            
            if pub_dt:
                if pub_dt > last_run_time:
                    fresh_news.append(item)
                else:
                    skipped_count += 1
            else:
                # No date? Keep it to be safe or Log warning. 
                fresh_news.append(item)
        
        logger.info(f"Freshness Filter: Kept {len(fresh_news)} new items. Skipped {skipped_count} old items.")
        news_items = fresh_news
    else:
        logger.info("Freshness Filter: No last run time found. Processing ALL news.")

    logger.info("Loading Portfolio...")
    portfolio_map = db.get_portfolio_map()
    if portfolio_map:
        logger.info(f"Loaded {len(portfolio_map)} holdings.")

    # 2.5 Enrich News with Technical Data & Portfolio Context
    logger.info("Enriching news with Technical Data & Portfolio Context...")
    
    # ... (Monitored Tickers Setup - Unchanged) ...
    # Dynamic Ticker Detection Setup
    from ticker_resolver import resolve_ticker
    
    # Common words to ignore to reduce API/DB spam
    IGNORE_LIST = {
        'THE', 'AND', 'FOR', 'NEW', 'CEO', 'IPO', 'AI', 'US', 'EU', 'UK', 'HK',
        'API', 'APP', 'BIG', 'BUY', 'NOW', 'TOP', 'HOT', 'SOS', 'RUN', 'SET',
        'EFT', 'ETF', 'CRYPTO', 'BITCOIN', 'ETHEREUM', 'SOLANA', 'RIPPLE', 
        'CARDANO', 'DOGECOIN', 'POLKADOT', 'CHAINLINK', 'AVALANCHE', 'POLYGON',
        'THIS', 'THAT', 'WITH', 'FROM', 'INTO', 'OVER', 'MORE', 'LESS', 'BEST',
        'REAL', 'TIME', 'YEAR', 'WEEK', 'DAY', 'HOUR', 'LIFE', 'GOOD', 'BAD',
        'LOW', 'HIGH', 'MAX', 'MIN', 'ONE', 'TWO', 'SIX', 'TEN', 'ALL', 'ANY',
        'CAN', 'GET', 'HAS', 'HAD', 'NOT', 'BUT', 'WHY', 'HOW', 'WHO', 'ITS',
        'WAR', 'TAX', 'LAW', 'JOB', 'PAY', 'FEE', 'WIN', 'LOSE', 'NET', 'GRO',
        'GDP', 'CPI', 'PPI', 'FED', 'ECB', 'SEC', 'DOJ', 'FTX', 'SBF', 'KYC',
        'AML', 'NFT', 'DAO', 'DEX', 'CEX', 'PUMP', 'DUMP', 'FOMO', 'FUD', 'ATH',
        'ATL', 'ROI', 'APR', 'APY', 'TVL', 'MCAP', 'VOL', 'PNL', 'YTD', 'QTD',
        'LTD', 'INC', 'CORP', 'LLC', 'PLC', 'AG', 'GMBH', 'SA', 'SPA', 'NV', 'BV',
        'UP', 'DOWN', 'LEFT', 'RIGHT', 'NORTH', 'SOUTH', 'EAST', 'WEST',
        'HITS', 'MISS', 'BEAT', 'DROP', 'FALL', 'RISE', 'JUMP', 'DIVE', 'SOAR',
        'SURGE', 'TANK', 'CRASH', 'BOOM', 'BUST', 'HOLD', 'SELL', 'SWAP', 'LONG',
        'SHORT', 'CALL', 'PUT', 'ASK', 'BID', 'AVG', 'EST', 'EPS', 'REV', 
        'SAYS', 'SAID', 'WILL', 'WENT', 'GONE', 'SEEN', 'DONE', 'MADE', 'MAKE',
        'KEEP', 'HELD', 'SOLD', 'BOUGHT', 'PAID', 'OWED', 'LENT', 'SENT', 'TOOK',
        'GAVE', 'GOT', 'MET', 'SUES', 'SUE', 'WON', 'LOST', 'COST', 'VALUE', 
        'PRICE', 'RATE', 'YIELD', 'BOND', 'NOTE', 'BILL', 'CASH', 'GOLD', 'OIL',
        'GAS', 'DATA', 'TECH', 'SOFT', 'HARD', 'FIRM', 'BANK', 'FUND', 'USER',
        'ZERO', 'COST', 'HUNT', 'TEXT', 'PAGE', 'SITE', 'HOME', 'MENU',
        'MOXIE', 'TFEX', # Noise from log analysis
        # HTML/CSS Garbage
        'HTML', 'CSS', 'SRC', 'HREF', 'IMG', 'JPG', 'PNG', 'DIV', 'SPAN', 
        'CLASS', 'STYLE', 'WIDTH', 'HEIGHT', 'MARGIN', 'PADDING', 'FLOAT', 
        'ALT', 'TYPE', 'COM', 'HTTP', 'HTTPS', 'WWW', 'NET', 'ORG', 'GOV',
        # Common Prepositions/Verbs (Short Uppercase risks)
        'THE', 'AND', 'FOR', 'BUT', 'NOT', 'YOU', 'ARE', 'WAS', 'ITS', 'HAS',
        'HAD', 'CAN', 'GET', 'DID', 'WAY', 'TOO', 'USE', 'SEE', 'OWN', 'GOT',
        'MET', 'WON', 'LOST', 'RUN', 'SET', 'PUT', 'SAY', 'LET', 'BIG', 'OLD',
        'FULL', 'PART', 'TIME', 'YEAR', 'WEEK', 'DAY', 'HOUR', 'LIFE',
        # Noise Words Discovered in Verification
        'AGAINST', 'WELCOME', 'STREET', 'DONALD', 'FIRMS', 'LEVEL', 'BASED',
        'VETERAN', 'FIRST', 'SECOND', 'THIRD', 'LAST', 'NEXT', 'PAST',
        'NEAR', 'FAR', 'AWAY', 'BACK', 'LEFT', 'RIGHT', 'SIDE', 'WAYS',
        'THINGS', 'STUFF', 'LOOK', 'SEEM', 'MANY', 'MUCH', 'MOST', 'LEAST',
        'SUCH', 'SAME', 'DIFFERENT', 'INSIDE', 'OUTSIDE', 'UNDER', 'ABOVE',
        # Media/Generic Words (v3 updates)
        'CNBC', 'AHEAD', 'SPARKED', 'REVERSES', 'PLAN', 'SCHEME', 'USING',
        'BEING', 'DOING', 'GOING', 'HAVING', 'MAKING', 'SAYING', 'SEEING',
        'TAKING', 'WANT', 'NEED', 'LIKE', 'KNOW', 'THINK', 'WELL', 'GOOD',
        'EDITOR'
    }

    # Canonical Map for De-duplication (Still useful for unifying aliases)
    CANONICAL_MAP = {
        "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
        "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
        "SOL": "SOL-USD", "SOLANA": "SOL-USD",
        "RNDR": "RENDER-USD", "RENDER": "RENDER-USD",
        "RNDR-USD": "RENDER-USD",
        "BYD": "BYDDF", # Map BYD to USD OTC (prevents Boyd Gaming mixup)
        "BYD COMPANY": "BYDDF",
        "META": "META",
        "GOOG": "GOOGL",
        "GOOGL": "GOOGL"
    }

    def find_portfolio_entry(ticker, portfolio):
        if ticker in portfolio:
            return ticker, portfolio[ticker]
        base = ticker.replace('-USD', '').replace('-EUR', '')
        if base in portfolio:
            return base, portfolio[base]
        if f"{ticker}-USD" in portfolio:
            key = f"{ticker}-USD"
            return key, portfolio[key]
        return None, None

    # Signal Intelligence is already initialized at start
    pass

    # [PERFORMANCE] Explicit Local Caching to avoid repeated API/DB calls
    logger.info("Initializing Local Cache for unique tickers...")
    unique_tickers = set()
    
    # Identify all unique tickers mentioned in news (Dynamic Extraction)
    local_resolved_cache = {} # Cache for this run to avoid repeatedly resolving same text

    # Identifies all unique tickers mentioned in news (Dynamic Extraction)
    # [PERFORMANCE] Two-Pass Bulk Resolution Strategy
    
    def get_raw_candidates(text):
        """Extract potential candidates using Regex (No DB calls)"""
        candidates = set(re.findall(r'\b[A-Z0-9]{3,8}\b', text))
        
        # Explicit crypto mapping (Case sensitive)
        common_names = ["Bitcoin", "Ethereum", "Solana", "Ripple", "Cardano", "Dogecoin", "Polkadot", "Avalanche"]
        for name in common_names:
            if name in text:
                 for k, v in CANONICAL_MAP.items():
                    if k.upper() == name.upper() or v.replace('-USD','').upper() == name.upper():
                        if k in CANONICAL_MAP: candidates.add(k)
                        else: candidates.add(name.upper())
                        break
        
        # Local Filtering (CPU only)
        final = set()
        for cand in candidates:
            if cand in IGNORE_LIST: continue
            if cand.isdigit(): continue
            if re.match(r'^\d+[KMBXG]$', cand): continue
            final.add(cand)
        return final

    # PASS 1: Collect ALL raw candidates from ALL news
    logger.info("Scanning news for raw candidates...")
    all_raw_candidates = set()
    news_candidate_map = [] # Store (item, candidates) for Pass 2
    
    for item in news_items:
        text_content = (item.get('title', '') + " " + item.get('summary', ''))
        raws = get_raw_candidates(text_content)
        all_raw_candidates.update(raws)
        news_candidate_map.append((item, raws))
        
    # BULK RESOLUTION (One DB Call)
    logger.info(f"Resolving {len(all_raw_candidates)} unique candidates...")
    from ticker_resolver import resolve_tickers
    resolved_map = resolve_tickers(list(all_raw_candidates))
    
    # PASS 2: Count Frequencies using Resolved Data
    MAX_NEW_DISCOVERIES = 5
    from collections import Counter
    discovery_counter = Counter()
    portfolio_found = set()
    
    for item, raws in news_candidate_map:
        processed_for_item = set()
        for raw in raws:
            # Get resolved value from bulk map
            resolved = resolved_map.get(raw)
            if not resolved: continue # Was rejected or failed
            
            # Canonicalize
            norm_ticker = CANONICAL_MAP.get(resolved, resolved)
            
            # De-duplicate per article
            if norm_ticker in processed_for_item: continue
            processed_for_item.add(norm_ticker)
            
            # Normalize portfolio matches
            if f"{norm_ticker}-USD" in portfolio_map:
                norm_ticker = f"{norm_ticker}-USD"
                
            if norm_ticker in portfolio_map:
                portfolio_found.add(norm_ticker)
            else:
                discovery_counter[norm_ticker] += 1

    # Step 3: Prioritize Selection
    # Always add portfolio assets found
    for p_ticker in portfolio_found:
        unique_tickers.add(p_ticker)
        
    # Select Top N New Discoveries by Frequency
    if discovery_counter:
        top_discoveries = discovery_counter.most_common(MAX_NEW_DISCOVERIES)
        logger.info(f"Top Discoveries Candidates: {top_discoveries}")
        
        for ticker, count in top_discoveries:
            unique_tickers.add(ticker)
            logger.info(f"Discovery: Added {ticker} ({count} mentions) to hunt list.")
    else:
        logger.info("Discovery: No new tickers found.")
    
    # Add all portfolio tickers too (for context/synthetic checks)
    for p_ticker in portfolio_map.keys():
        unique_tickers.add(p_ticker)
    
    # Pre-fetch technicals into LOCAL MEMORY dictionary
    local_tech_cache = {}
    local_price_cache = {}
    
    logger.info(f"Batch Fetching Data for {len(unique_tickers)} unique tickers...")
    
    for ticker in unique_tickers:
        try:
            # 1. Technical Summary
            # This might hit DB/API initially, but we do it ONCE per ticker
            tech = market.get_technical_summary(ticker)
            local_tech_cache[ticker] = tech
            
            # 2. Smart Price (EUR)
            price, _ = market.get_smart_price_eur(ticker)
            local_price_cache[ticker] = price
            
            # logger.info(f"Cached {ticker}: {price} EUR")
            
        except Exception as e:
            logger.warning(f"Failed to cache data for {ticker}: {e}")
            local_tech_cache[ticker] = "Data Unavailable"
            local_price_cache[ticker] = 0.0
            # [PERFORMANCE] Register failure to filter noise in future runs
            db.register_ticker_failure(ticker)

    logger.info(f"Local Cache Ready. Enriched {len(unique_tickers)} tickers.")

    # --- NEW DEDUPLICATION & MERGING (Moved up to save resources) ---
    merged_map = {}
    for item in news_items:
        text_content = (item.get('title', '') + " " + item.get('summary', '')).upper()
        
        # Find first matching ticker (Dynamic)
        raws = get_raw_candidates(text_content)
        extracted = []
        for r in raws:
             res = resolved_map.get(r)
             if res: extracted.append(res)
        
        detected_ticker = None
        if extracted:
            best_match = extracted[0]
            for t in extracted:
                if t in portfolio_map or f"{t}-USD" in portfolio_map:
                    best_match = t
                    break
                if t in CANONICAL_MAP:
                    best_match = CANONICAL_MAP[t]
            detected_ticker = best_match
            
        if detected_ticker:
            if detected_ticker in CANONICAL_MAP:
                detected_ticker = CANONICAL_MAP[detected_ticker]
            elif f"{detected_ticker}-USD" in portfolio_map:
                detected_ticker = f"{detected_ticker}-USD"
            elif f"{detected_ticker}USD" in portfolio_map:
                 detected_ticker = f"{detected_ticker}USD"
            
            item['ticker'] = detected_ticker
            
            if detected_ticker in merged_map:
                existing = merged_map[detected_ticker]
                existing['summary'] += f"\n\n--- ADDITIONAL ARTICLE: {item.get('title', 'Untitled')} ---\n{item.get('summary', '')}"
                # Keep original source if not already set or prioritize better sources?
            else:
                merged_map[detected_ticker] = item

    # Add synthetic assets that have no news
    # (We'll handle this in the next block)
    # --- ENRICHMENT PREPARATION ---
    # We will enrich all items (RSS + Synthetic + Pulse) before sending to AI
    analysis_batch = []
    
    # 1. Add Deduplicated RSS News
    analysis_batch.extend(list(merged_map.values()))
    logger.info(f"Deduplicated news for analysis: {len(news_items)} -> {len(analysis_batch)}")

    # Enrich RSS news items with Technicals & Portfolio context
    for item in analysis_batch:
        detected_ticker = item.get('ticker')
        if detected_ticker:
            extras = []
            
            # 2. Technicals (FROM LOCAL CACHE)
            tech_summary = local_tech_cache.get(detected_ticker)
            if not tech_summary:
                tech_summary = market.get_technical_summary(detected_ticker)
            
            extras.append(f"Technical: {tech_summary}")
            
            # 2.5 Signal Intelligence Context
            if si:
                try:
                    p_context = list(portfolio_map.values()) if portfolio_map else []
                    si_context = si.generate_context_for_ai(detected_ticker, portfolio_context=p_context)
                    extras.append(si_context)
                except Exception as e:
                    logger.warning(f"Signal Intelligence failed for {detected_ticker}: {e}")
            
            # 3. Portfolio
            portfolio_ticker, holding = find_portfolio_entry(detected_ticker, portfolio_map)
            
            if holding:
                current_price_eur = local_price_cache.get(detected_ticker, 0.0)
                if current_price_eur == 0.0:
                     current_price_eur, _ = market.get_smart_price_eur(detected_ticker)

                pnl_str = ""
                if current_price_eur > 0 and holding['avg_price'] > 0:
                    pnl_pct = ((current_price_eur - holding['avg_price']) / holding['avg_price']) * 100
                    sign = "+" if pnl_pct >= 0 else ""
                    pnl_str = f" | PnL: {sign}{pnl_pct:.2f}%"
                    logger.info(f"PnL for {detected_ticker}: {pnl_pct:.2f}%")

                p_summary = f"OWNED {holding['quantity']} @ €{holding['avg_price']:.2f}{pnl_str}"
                extras.append(f"Portfolio: {p_summary}")
            else:
                extras.append("Portfolio: NOT OWNED (New Entry)")

            item['summary'] += "\n\n[" + " | ".join(extras) + "]"
            logger.debug(f"Enriched {detected_ticker} news.")

    # --- SYNTHETIC PORTFOLIO INJECTION ---
    # Ensure ALL portfolio assets are analyzed.
    found_tickers_in_rss = set()
    for i in analysis_batch:
        if i.get('ticker'):
            t = i['ticker']
            if t in CANONICAL_MAP: t = CANONICAL_MAP[t]
            found_tickers_in_rss.add(t)

    for p_ticker, holding in portfolio_map.items():
        norm_p_ticker = CANONICAL_MAP.get(p_ticker, p_ticker)
        
        if norm_p_ticker not in found_tickers_in_rss:
            logger.info(f"Hunter: Portfolio Asset {norm_p_ticker} not in news. Generating Synthetic Check...")
            try:
                fetch_ticker = norm_p_ticker
                tech_summary = market.get_technical_summary(fetch_ticker)
                
                # PnL Calculation for Synthetic
                pnl_str = ""
                current_price_eur, _ = market.get_smart_price_eur(fetch_ticker)
                
                if current_price_eur > 0 and holding['avg_price'] > 0:
                    pnl_pct = ((current_price_eur - holding['avg_price']) / holding['avg_price']) * 100
                    sign = "+" if pnl_pct >= 0 else ""
                    pnl_str = f" | PnL: {sign}{pnl_pct:.2f}%"
                    logger.info(f"Synthetic PnL for {fetch_ticker}: {pnl_pct:.2f}%")

                synthetic_item = {
                    "title": f"PORTFOLIO CHECK: {fetch_ticker}",
                    "link": f"https://finance.yahoo.com/quote/{fetch_ticker}",
                    "summary": f"Routine technical check for owned asset. {tech_summary}. [Portfolio: OWNED {holding['quantity']} @ €{holding['avg_price']:.2f}{pnl_str}]",
                    "published": "Just Now",
                    "ticker": fetch_ticker,
                    "synthetic": True,
                    "source": "Portfolio Technicals"
                }
                analysis_batch.append(synthetic_item)
                logger.debug(f"Injected Synthetic Item for {fetch_ticker}")
            except Exception as e:
                logger.error(f"Failed to generate synthetic item for {norm_p_ticker}: {e}")
                
    # --- PHASE: MARKET PULSE (PREDICTIVE QUANT) ---
    try:
        pulse_results = pulse.scan()
        for anomaly in pulse_results:
            ticker = anomaly['ticker']
            findings_str = "\n".join(anomaly['findings'])
            
            pulse_item = {
                "title": f"MARKET PULSE: {ticker} Technical Anomaly",
                "link": f"https://finance.yahoo.com/quote/{ticker}",
                "summary": f"QUANTITATIVE ALERT: Technical analysis detected following anomalies for {ticker}:\n{findings_str}\n[Metrics: Vol Ratio {anomaly['metrics']['vol_ratio']}x, RSI {anomaly['metrics']['rsi']}]",
                "published": "Pulse REAL-TIME",
                "ticker": ticker,
                "pulse": True,
                "confidence_modifier": anomaly['confidence_modifier'],
                "source": "Market Pulse"
            }
            analysis_batch.append(pulse_item)
            logger.info(f"Pulse: Injected anomaly for {ticker}")
    except Exception as e:
        logger.error(f"Market Pulse failed: {e}")

    # Final check on analysis batch
    logger.info(f"Total items for AI analysis: {len(analysis_batch)}")
    # ----------------------------------------------

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
    ins = Insider()
    market_mood = ins.get_market_mood()
    # Optimized: Fetching social headlines once. 
    # These headlines ALREADY contain velocity info (from Insider.get_social_sentiment)
    social_headlines = ins.get_social_sentiment()
    
    insider_context = None
    if market_mood or social_headlines:
        insider_context = market_mood if market_mood else {}
        if social_headlines:
            insider_context['social'] = social_headlines
            logger.info(f"Insider: Found {len(social_headlines)} trending social headlines.")
        
        if market_mood:
            logger.info(f"Insider: Market Mood is {market_mood.get('overall')} ({market_mood.get('crypto',{}).get('value')})")

    # [SOCIAL CONTEXT] Prepare for Brain - Use headlines for better context
    social_context = ""
    if social_headlines:
        social_context = "\n[THE ORACLE: SOCIAL TRENDING]\n" + "\n".join(social_headlines)

    # [ADVISOR] Portfolio Health Analysis
    # Advisor initialized at start: adv
    # Fetch current portfolio from DB for analysis
    # We use portfolio_map values (loaded earlier)
    portfolio_list = list(portfolio_map.values()) if portfolio_map else []
    advisor_analysis = adv.analyze_portfolio(portfolio_list)
    if advisor_analysis:
        logger.info(f"Advisor: Portfolio Value ${advisor_analysis['total_value']:.2f}. Tips: {len(advisor_analysis.get('tips', []))}")

    # [ECONOMIST] Macro Context (V4.0)
    macro_context = economist.get_macro_summary()
    logger.info(f"Economist: Macro Context Generated. ({len(macro_context)} chars)")

    # [SECTOR ANALYST] Phase 3 Rotation
    sector_signals = []
    try:
        from market_data import SectorAnalyst
        sa = SectorAnalyst(market_instance=market)
        sector_signals = sa.get_rotation_signals()
        if sector_signals:
            logger.info(f"Sector Analyst: Generated {len(sector_signals)} rotation signals.")
    except Exception as e:
        logger.warning(f"Sector Analyst failed: {e}")


    # [WHALE WATCHER] On-Chain Context (V4.0 Phase 11)
    whale = WhaleWatcher()
    whale_context = whale.analyze_flow()
    
    # Safe logging
    w_lines = whale_context.splitlines()
    if len(w_lines) > 2:
        log_hint = w_lines[6].strip() if len(w_lines) > 6 and 'strategy_hint' in whale_context else ''
        logger.info(f"WhaleWatcher: {w_lines[2].strip()} | {log_hint}")
    else:
        logger.info(f"WhaleWatcher: {whale_context}")

    # [SENTINEL] Strategic Portfolio Forecast (V4.1 Predittivo)
    sentinel = Sentinel(db_handler=db)
    strategic_forecast = sentinel.get_strategic_forecast(market_data=market)
    if "error" not in strategic_forecast:
        logger.info(f"Sentinel: Strategic Forecast generated for regime {strategic_forecast.get('regime')}")
    else:
        logger.warning(f"Sentinel: Forecast failed: {strategic_forecast.get('error')}")

    # Remove redundant deduplication block
    # -----------------------------------------

    # --- FETCH USER SETTINGS ---
    # Moved up for L2 Regime Logic dependence
    user_settings = db.get_settings()
    min_conf = float(user_settings.get("min_confidence", 0.70))
    only_portfolio = user_settings.get("only_portfolio", False)
    logger.info(f"Smart Filters Active: Min Confidence={min_conf}, Only Portfolio={only_portfolio}")

    # --- L2 PREDICTIVE: MARKET REGIME ---
    # Adjust strategy aggressiveness based on market regime
    market_regime_data = None
    regime = "NEUTRAL"
    market_regime_summary = "MARKET REGIME: UNKNOWN" # Default
    try:
        regime_classifier = MarketRegimeClassifier()
        market_regime_data = regime_classifier.classify()
        
        regime = market_regime_data.get("regime", "NEUTRAL")
        regime_conf = market_regime_data.get("confidence", 0.5)
        recommendation = market_regime_data.get("recommendation", "normal")
        
        market_regime_summary = f"MARKET REGIME: {regime} ({regime_conf:.0%}) - {recommendation.upper()} MODE"
        logger.info(f"L2 Regime Summary: {market_regime_summary}")

        # Adjust min_confidence based on regime
        original_min_conf = min_conf
        if recommendation == "aggressive" and regime in ["BULL", "ACCUMULATION"]:
            min_conf = max(0.40, min_conf - 0.20)  # [VOLUME FIX] Lowered from 0.60
            logger.info(f"L2 Regime [{regime}]: Aggressive mode - min_conf {original_min_conf} -> {min_conf}")
        elif recommendation == "defensive" and regime in ["BEAR", "DISTRIBUTION"]:
            min_conf = min(0.85, min_conf + 0.05)  
            logger.info(f"L2 Regime [{regime}]: Defensive mode - min_conf {original_min_conf} -> {min_conf}")
        else:
            # NEUTRAL: Lower slightly to catch evolving trends
            min_conf = max(0.45, min_conf - 0.10)
            logger.info(f"L2 Regime [{regime}]: Normal mode - min_conf {original_min_conf} -> {min_conf}")
    except Exception as e:
        logger.warning(f"L2 Market Regime failed: {e}")

    # [THE ORACLE] On-Chain Intelligence (V4.2 Phase B)
    onchain_context = ""
    try:
        if detected_tickers:
            main_ticker = list(detected_tickers)[0] # Focus on the most important one
            oc_data = onchain_watcher.get_onchain_context(main_ticker)
            onchain_context = oc_data
            logger.info(f"Oracle: On-chain data gathered for {main_ticker}.")
    except Exception as e:
        logger.warning(f"Oracle on-chain failed: {e}")
            
    except Exception as e:
        logger.warning(f"Oracle data collection failed: {e}")

    logger.info("Analyzing news with Gemini...")
    _ai_start_time = timing_module.time()
    try:
        predictions = brain.analyze_news_batch(
            analysis_batch, 
            performance_context=performance_context, 
            insider_context=insider_context,
            portfolio_context=advisor_analysis,
            macro_context=macro_context,
            whale_context=whale_context,
            market_regime_summary=market_regime_summary,
            social_context=social_context,
            onchain_context=onchain_context,
            strategic_forecast=strategic_forecast
        )
        _ai_time = timing_module.time() - _ai_start_time
        logger.info(f"Gemini analysis complete. Received {len(predictions)} predictions in {_ai_time:.1f}s.")
    except Exception as e:
        _ai_time = timing_module.time() - _ai_start_time
        logger.error(f"CRITICAL: Gemini analysis FAILED: {e}")
        predictions = []

    processed_count = 0
    
    # 4. Process Predictions
    for pred in predictions:
        ticker = pred.get("ticker")
        sentiment = pred.get("sentiment")
        reasoning = pred.get("reasoning")
        confidence = float(pred.get("confidence", 0.0))
        source = pred.get("source", "Unknown")
        
        # Initialize values that may be used later (prevent UnboundLocalError)
        target_price = pred.get("target_price")
        upside_percentage = float(pred.get("upside_percentage", 0)) if pred.get("upside_percentage") else 0
        risk_score = pred.get("risk_score", 5)  # Default medium risk
        
        # Critic Fields
        critic_verdict = pred.get("critic_verdict")
        critic_score = pred.get("critic_score")
        critic_reasoning = pred.get("critic_reasoning")
        
        # [PHASE C.3] Council Summary extraction
        council_summary = pred.get("council_summary")
        is_council_verified = pred.get("is_council_verified", False)
        
        # [FIX] Avoid variable leaking from previous loops (Discovery/Synthetic)
        portfolio_ticker, holding = find_portfolio_entry(ticker, portfolio_map)

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
        # Use flexible matching for currency variants (XRP-USD matches XRP-EUR, RENDER matches RENDER-USD)
        def is_owned(signal_ticker, pmap):
            """Enhanced ownership check - matches base ticker regardless of currency suffix."""
            signal_upper = signal_ticker.upper()
            
            # Exact match
            if signal_upper in pmap:
                return True
            
            # Extract base ticker (remove -USD, -EUR suffixes)
            base_signal = signal_upper.replace('-USD', '').replace('-EUR', '').replace('-GBP', '')
            
            # Check each portfolio ticker for a base match
            for portfolio_ticker in pmap.keys():
                portfolio_upper = portfolio_ticker.upper()
                base_portfolio = portfolio_upper.replace('-USD', '').replace('-EUR', '').replace('-GBP', '')
                
                # Base match (XRP == XRP regardless of suffix)
                if base_signal == base_portfolio:
                    logger.debug(f"Ownership match: {signal_ticker} ≈ {portfolio_ticker}")
                    return True
            
            return False
        
        if sentiment in ["SELL", "PANIC SELL", "HOLD", "ACCUMULATE"] and not is_owned(ticker, portfolio_map):
            logger.warning(f"Skipped {ticker}: Ignored {sentiment} signal for unowned asset.")
            continue

        # FILTER 4: Market Hours (hard gate for actionable stock signals)
        try:
            ticker_meta = {}
            if hasattr(db, "get_ticker_cache"):
                cached_meta = db.get_ticker_cache(ticker)
                if isinstance(cached_meta, dict):
                    ticker_meta = cached_meta

            market_status = economist.get_market_status()
            is_open, market_bucket, market_label = economist.get_trading_status_for_ticker(
                ticker=ticker,
                market_status=market_status,
                resolved_ticker=ticker_meta.get("resolved_ticker"),
                is_crypto=ticker_meta.get("is_crypto"),
                currency=ticker_meta.get("currency"),
            )

            if not is_open:
                logger.info(f"Skipped {ticker}: {market_bucket} market closed ({market_label}) - no stock signals when closed")
                continue
        except Exception as e:
            logger.warning(f"Market hours check failed for {ticker}: {e}")

        # Check if recently analyzed (Same Ticker + Same Sentiment = SPAM)
        if db.check_if_analyzed_recently(ticker, sentiment):
            continue
        
        # --- SMART SENTIMENT CONVERSION ---
        # Convert BUY → ACCUMULATE if asset is already owned (more appropriate action)
        if sentiment == "BUY" and is_owned(ticker, portfolio_map):
            sentiment = "ACCUMULATE"
            logger.info(f"Converted {ticker}: BUY → ACCUMULATE (already owned)")
        
        # --- SIGNAL INTELLIGENCE LAYER (NEW) ---
        # Apply advanced filtering and adjustments
        try:
            if not si: raise Exception("SignalIntelligence disabled or failed init")
            logger.info(f"Signal Intelligence: Analyzing {ticker} ({sentiment} @ {confidence:.2f})")
            
            p_context = list(portfolio_map.values()) if portfolio_map else []
            si_analysis = si.analyze_signal(ticker, sentiment, confidence, portfolio_context=p_context)
            
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
            
            # --- PREDICTIVE SYSTEM L1: TECHNICAL CONFLUENCE ---
            try:
                confluence = si.check_technical_confluence(ticker, sentiment)
                confluence_multiplier = confluence.get('multiplier', 1.0)
                confidence = min(1.0, confidence * confluence_multiplier)
                if confluence.get('alignment', 0) >= 2:
                    logger.info(f"L1 Confluence [{ticker}]: {confluence.get('alignment')}/3 aligned -> confidence boost {confluence_multiplier:.2f}")
                    reasoning += f"\n✅ Confluence: {confluence.get('reason', 'N/A')}"
            except Exception as e:
                logger.warning(f"L1 Confluence failed for {ticker}: {e}")
            
            # --- PREDICTIVE SYSTEM L1: MULTI-TIMEFRAME ---
            try:
                mtf = market.get_multi_timeframe_trend(ticker)
                mtf_boost = mtf.get('confidence_boost', 1.0)
                if mtf.get('direction') == 'mixed':
                    confidence *= mtf_boost  # Penalize mixed signals
                    logger.info(f"L1 MTF [{ticker}]: Mixed timeframes -> penalty {mtf_boost:.2f}")
                elif mtf.get('direction') in ['bullish', 'bearish']:
                    confidence = min(1.0, confidence * mtf_boost)
                    logger.info(f"L1 MTF [{ticker}]: {mtf.get('direction')} ({mtf.get('alignment')}/3) -> boost {mtf_boost:.2f}")
                    reasoning += f" [MTF: {mtf.get('direction')} {mtf.get('alignment')}/3]"
            except Exception as e:
                logger.warning(f"L1 MTF failed for {ticker}: {e}")
            
            # --- PREDICTIVE SYSTEM L1: SENTIMENT ADJUSTMENT ---
            try:
                from sentiment_aggregator import SentimentAggregator
                sentiment_agg = SentimentAggregator()
                adjusted_conf = sentiment_agg.adjust_confidence(confidence, sentiment)
                if adjusted_conf != confidence:
                    logger.info(f"L1 Sentiment [{ticker}]: Adjusted confidence {confidence:.2f} -> {adjusted_conf:.2f}")
                    confidence = adjusted_conf
                    mkt_score = sentiment_agg.get_score()
                    reasoning += f"\n📊 Mkt Sentiment: {mkt_score}"
            except Exception as e:
                logger.warning(f"L1 Sentiment failed for {ticker}: {e}")
            
            # --- PREDICTIVE SYSTEM L2: DIVERGENCE DETECTOR ---
            try:
                divergence = si.check_divergence(ticker)
                if divergence.get('has_divergence'):
                    div_type = divergence.get('type')
                    div_boost = divergence.get('confidence_boost', 1.0)
                    
                    # Bullish divergence on BUY signal = boost
                    if div_type == 'bullish' and sentiment in ['BUY', 'ACCUMULATE']:
                        confidence = min(1.0, confidence * div_boost)
                        logger.info(f"L2 Divergence [{ticker}]: Bullish divergence -> boost {div_boost:.2f}")
                        reasoning += f"\n📐 Divergence: Bullish ({divergence.get('strength'):.0%})"
                    # Bearish divergence on BUY signal = warning
                    elif div_type == 'bearish' and sentiment in ['BUY', 'ACCUMULATE']:
                        confidence *= 0.90  # Penalize
                        logger.info(f"L2 Divergence [{ticker}]: Bearish divergence on BUY -> penalty")
                        reasoning += "\n⚠️ Divergence: Bearish (Caution)"
                    # Bearish divergence on SELL signal = boost
                    elif div_type == 'bearish' and sentiment in ['SELL', 'TRIM']:
                        confidence = min(1.0, confidence * div_boost)
                        logger.info(f"L2 Divergence [{ticker}]: Bearish divergence on SELL -> boost")
            except Exception as e:
                logger.warning(f"L2 Divergence failed for {ticker}: {e}")
            
            # --- PREDICTIVE SYSTEM L4: MACHINE LEARNING ---
            try:
                # Quant Path: Inject Sentiment & Regime
                sentiment_score = int(confidence * 100)
                
                # Predict ONCE (Saves to DB with new context)
                ml_pred = ml_predictor.predict(ticker, sentiment_score, regime)
                
                # Get modifier from the prediction object
                ml_modifier = ml_predictor.get_confidence_modifier_from_pred(ml_pred, sentiment)
                
                original_conf = confidence
                confidence = min(1.0, confidence * ml_modifier)
                
                ml_type = "ML" if ml_pred.is_ml else "Rule"
                
                if ml_modifier != 1.0:
                    logger.info(f"L4 ML [{ticker}]: {ml_pred.direction} ({ml_pred.confidence:.0%}) -> modifier {ml_modifier:.2f}")
                    if ml_modifier > 1.0:
                        reasoning += f"\n🤖 ML Check: ✅ Confirmed (Confidence {ml_pred.confidence:.0%})"
                    else:
                        match = "Neutral" if ml_pred.direction == "HOLD" else "Divergence"
                        emoji = "⏸️" if ml_pred.direction == "HOLD" else "⚠️"
                        reasoning += f"\n🤖 ML Check: {emoji} {match}: {ml_pred.direction} ({ml_pred.confidence:.0%})"
            except Exception as e:
                logger.warning(f"L4 ML failed for {ticker}: {e}")
            
            # --- L5: RSI EXTREME FILTER (Align with Analyze) ---
            try:
                features = ml_predictor._get_features(ticker)
                if features:
                    rsi = features.get('rsi_14', 50)
                    # Penalize BUY when extremely overbought (RSI > 80)
                    if rsi > 80 and sentiment in ['BUY', 'ACCUMULATE', 'STRONG BUY']:
                        penalty = 0.75  # -25% confidence
                        confidence = min(1.0, confidence * penalty)
                        reasoning += f"\n⚠️ RSI: OVERBOUGHT ({rsi:.0f}) - caution"
                        logger.info(f"L5 RSI [{ticker}]: RSI {rsi:.0f} > 80 -> BUY penalty {penalty}")
                    # Penalize SELL when extremely oversold (RSI < 20)
                    elif rsi < 20 and sentiment in ['SELL', 'TRIM', 'PANIC SELL']:
                        penalty = 0.75
                        confidence = min(1.0, confidence * penalty)
                        reasoning += f"\n⚠️ RSI: OVERSOLD ({rsi:.0f}) - caution"
                        logger.info(f"L5 RSI [{ticker}]: RSI {rsi:.0f} < 20 -> SELL penalty {penalty}")
            except Exception as e:
                logger.warning(f"L5 RSI filter failed for {ticker}: {e}")
            
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

        stop_loss = pred.get("stop_loss")
        take_profit = pred.get("take_profit")
        
        # --- RISK MANAGEMENT: DYNAMIC ATR (Quant Path) ---
        try:
            # 1. Calculate ATR
            atr_data = market.calculate_atr(ticker)
            atr_val = atr_data.get("atr", 0)
            
            # 2. Extract specific price for calculation
            # Use current price from market data (already fetched or fetch fresh)
            calc_price, _ = market.get_smart_price_eur(ticker)
            
            if atr_val > 0 and calc_price > 0:
                # 3. Calculate Dynamic SL/TP
                # SL = Price - 2x ATR (Standard Swing)
                # TP = Price + 4x ATR (1:2 Risk/Reward)
                
                sl_atr = calc_price - (2.0 * atr_val)
                tp_atr = calc_price + (4.0 * atr_val)
                
                # Override AI Hallucinations
                old_sl = stop_loss
                stop_loss = round(max(0, sl_atr), 2)
                take_profit = round(tp_atr, 2)
                
                logger.info(f"Risk Mgmt [{ticker}]: ATR {atr_val:.2f} -> SL {stop_loss} (was {old_sl}), TP {take_profit}")
            else:
                 # FALLBACK: If ATR is 0, use a fixed percentage (5% SL, 10% TP)
                 if calc_price > 0:
                     old_sl = stop_loss
                     stop_loss = round(calc_price * 0.95, 2)
                     take_profit = round(calc_price * 1.10, 2)
                     logger.info(f"Risk Mgmt [{ticker}]: ATR 0 -> Using Fixed Fallback (5% SL, 10% TP): SL {stop_loss}, TP {take_profit}")
                 else:
                     logger.warning(f"Risk Mgmt [{ticker}]: ATR/Price invalid (ATR={atr_val}, Price={calc_price})")

        except Exception as e:
            logger.warning(f"Risk Mgmt failed for {ticker}: {e}")
        
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
        # 5. Log to DB and Notify
        signal_id = db.log_prediction(
            ticker, sentiment, reasoning, reasoning, confidence, source, 
            risk_score, target_price, upside_percentage, stop_loss, take_profit,
            critic_verdict=critic_verdict, critic_score=critic_score, critic_reasoning=critic_reasoning
        )
        
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
                         paper_trader.execute_trade(999999999, ticker, "BUY", qty, p_price, f"Auto-Signal: {confidence:.2f}", sl=stop_loss, tp=take_profit)
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

        # ----------------------------------------
        
        # Format Alert
        is_owned_asset = holding is not None
        consensus_data = consensus_engine.calculate_weighted_action(pred, is_owned=is_owned_asset)
        alert_msg = format_alert_msg(
            ticker, sentiment, confidence, reasoning, source, pred, 
            stop_loss, take_profit, critic_score, critic_reasoning, 
            council_summary, consensus_data=consensus_data,
            ai_footer=None
        )
        signal_cards.append(alert_msg)
        
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

    # --- PREDICTIVE SYSTEM L1: CORRELATION PROPAGATION ---
    # After processing all signals, propagate strong signals to correlated assets
    try:
        from correlation_engine import CorrelationEngine
        corr_engine = CorrelationEngine()
        
        # Collect tickers that already received signals to avoid duplicates
        already_signaled = [p.get('ticker') for p in predictions if p.get('confidence', 0) >= min_conf]
        
        propagated_count = 0
        for pred in predictions:
            ticker = pred.get('ticker')
            sentiment = pred.get('sentiment')
            confidence = float(pred.get('confidence', 0))
            
            # Only propagate strong actionable signals
            if confidence >= 0.80 and sentiment in ['BUY', 'ACCUMULATE', 'SELL']:
                propagated_signals = corr_engine.propagate(ticker, sentiment, confidence, already_signaled)
                
                for prop_signal in propagated_signals:
                    prop_ticker = prop_signal['ticker']
                    prop_sentiment = prop_signal['sentiment']
                    prop_conf = prop_signal['confidence']
                    prop_reasoning = prop_signal['reasoning']
                    
                    # Skip if below threshold
                    if prop_conf < min_conf:
                        continue
                    
                    # Check if we should notify (not in portfolio && not owned check)
                    already_signaled.append(prop_ticker)
                    
                    # Log propagated signal
                    logger.info(f"L1 Correlation: {ticker} → {prop_ticker} ({prop_sentiment} @ {prop_conf:.0%})")
                    
                    propagated_cards.append(
                        f"`{prop_ticker}` → {prop_sentiment} ({int(prop_conf * 100)}%) | {_truncate_text(prop_reasoning, 160)}"
                    )
                    propagated_count += 1
        
        if propagated_count > 0:
            logger.info(f"L1 Correlation: Generated {propagated_count} propagated signals")
    except Exception as e:
        logger.warning(f"L1 Correlation propagation failed: {e}")
    # -----------------------------------------------------

    # --- POSITION WATCHDOG: SELL SIGNALS FOR OWNED ASSETS ---
    # Scans portfolio for exit opportunities based on:
    # 1. Dynamic ATR thresholds (volatility-adjusted)
    # 2. Technical indicators (RSI, Momentum, Trend)
    # 3. ML predictions for trend reversal
    # 4. Tax-aware profit calculations (26% + €1 fee)
    try:
        from position_watchdog import PositionWatchdog
        
        watchdog = PositionWatchdog(
            db_handler=db,
            market_data=market,
            ml_predictor=ml_predictor
        )
        
        exit_signals = await watchdog.scan_portfolio()
        
        if exit_signals:
            logger.info(f"Position Watchdog: Generated {len(exit_signals)} exit signals")
            
            # Format and queue for single hunt digest
            watchdog_report = watchdog.format_telegram_report(exit_signals)
            
            # Log each signal for tracking
            for signal in exit_signals:
                logger.info(f"Exit Signal: {signal.ticker} → {signal.action} ({signal.confidence:.0%}) | Net: €{signal.net_profit:.2f}")
        else:
            logger.info("Position Watchdog: No exit signals generated (all positions healthy)")
    except Exception as e:
        logger.warning(f"Position Watchdog failed: {e}")

    # --- AUDIT PHASE ---
    logger.info("Running Auditor Checkup...")
    audit_results = await auditor.audit_open_signals()
    if audit_results:
        summary_audit = "\n".join([f"• **{r['ticker']}**: {r['pnl_percent']:+.2f}% ({r['status']})" for r in audit_results])
        audit_report = f"⚖️ **Aggiornamento Auditor**\n{summary_audit}"

    # --- MAINTENANCE PHASE (Storage Monitoring) ---
    try:
        from db_maintenance import DBMaintenance
        maint = DBMaintenance()
        health = maint.check_storage_health()
        
        if health["status"] == "critical":
            # Auto-cleanup and notify
            deleted = maint.cleanup_old_records(force=True)
            total_deleted = sum(v for v in deleted.values() if v > 0)
            maintenance_report = (
                f"⚠️ **Allerta Storage**\n{health['message']}\n🧹 Pulizia automatica: {total_deleted} record."
            )
        elif health["status"] == "warning":
            maintenance_report = f"⚡ **Warning Storage**\n{health['message']}"
        
        logger.info(f"Maintenance: {health['message']}")
    except Exception as e:
        logger.warning(f"Maintenance check failed: {e}")

    # --- API USAGE REPORT AT END ---
    try:
        api_usage = db.get_api_usage()
        run_stats = api_usage.get('runs', {}).get(run_id, {})
        openrouter_this_run = run_stats.get('openrouter', 0)
        last_model = api_usage.get('last_model', 'N/A')
        model_short = last_model.split('/')[-1].replace(':free', '') if last_model != 'N/A' else 'N/A'
        
        logger.info(f"API Usage Report [END]:")
        logger.info(f"   Run ID: {run_id}")
        logger.info(f"   Model Used: {model_short}")
        logger.info(f"   This Run: {openrouter_this_run} OpenRouter calls")
        logger.info(f"   Total Today: {api_usage.get('openrouter', 0)} OpenRouter calls")
        
        # Show per-model breakdown
        models_used = api_usage.get('models', {})
        if models_used:
            logger.info(f"   Per-Model Stats:")
            for model, count in list(models_used.items())[:5]:
                m_short = model.split('/')[-1].replace(':free', '')
                logger.info(f"      └ {m_short}: {count}")
        
        logger.info(f"   Reset in: {api_usage.get('hours_until_reset', 0):.1f}h ({api_usage.get('reset_at_local', 'N/A')})")
    except Exception as e:
        logger.warning(f"API usage report failed: {e}")

    db.log_system_event("INFO", "Hunter", "Pipeline Finished")
    _total_run_time = timing_module.time() - _run_start_time
    logger.info(f"Pipeline finished. Processed {processed_count} signals in {_total_run_time:.1f}s.")
    
    # Save run metrics to DB
    try:
        run_metrics = {
            "total_time": _total_run_time,
            "ai_time": _ai_time,
            "news_fetch_time": _news_fetch_time,
            "signals_count": processed_count,
            "model_used": brain.last_run_details.get("model", "unknown"),
            "json_repair_needed": brain.last_run_details.get("json_repair_needed", False),
            "repair_strategy": brain.last_run_details.get("repair_strategy", "none"),
            "retry_count": brain.last_run_details.get("retry_count", 0),
            "news_items_processed": len(news_items) if isinstance(news_items, list) else 0
        }
        db.save_run_metrics(run_metrics)
        
        # UPDATE LAST SUCCESSFUL RUN TIMESTAMP
        try:
             from datetime import datetime, timezone
             now_iso = datetime.now(timezone.utc).isoformat()
             db.update_settings_last_run(now_iso)
        except Exception as e:
            logger.warning(f"Failed to update last_successful_hunt_ts: {e}")

    except Exception as e:
        logger.warning(f"Failed to save run metrics: {e}")

    
    # --- FLASH REBALANCE CHECK (New L5 Feature) ---
    flash_tip = ""
    try:
        rb = Rebalancer()
        tip = rb.get_flash_recommendation()
        if tip:
            flash_tip = f"\n\n💡 {tip}"
    except Exception as e:
        logger.warning(f"Flash rebalance check failed: {e}")
    
    # Send single consolidated hunt message to Telegram
    try:
        try:
            details = brain.last_run_details
            if details:
                model_name = details.get('model', 'Unknown').split('/')[-1].replace(':free', '')
                usage = details.get("usage", {})
                total_tok = _safe_total_tokens_from_usage(usage)
                ai_footer = f"🤖 AI: {model_name} ({total_tok} token)"
        except Exception as e:
            logger.debug(f"Failed to build ai_footer: {e}")
            ai_footer = ""

        total_time = timing_module.time() - _run_start_time
        final_report = _build_hunt_digest_message(
            analyzed_items=len(analysis_batch),
            total_time=total_time,
            ai_footer=ai_footer,
            processed_count=processed_count,
            signal_cards=signal_cards,
            propagated_cards=propagated_cards,
            watchdog_report=watchdog_report,
            audit_report=audit_report,
            maintenance_report=maintenance_report,
            flash_tip=flash_tip,
            sector_signals=sector_signals,
            sentinel_cards=sentinel_cards,
            ml_training_note=ml_training_note,
        )

        target_chat = user_settings.get("telegram_chat_id") or notifier.chat_id
        parts = _split_telegram_message(final_report, max_len=3800)
        if len(parts) == 1:
            await notifier.send_message(target_chat, parts[0])
        else:
            total_parts = len(parts)
            for idx, part in enumerate(parts, start=1):
                prefix = f"📨 **Hunt report {idx}/{total_parts}**\n"
                await notifier.send_message(target_chat, prefix + part)
    except Exception as e:
        logger.warning(f"Failed to send completion notification: {e}")

    model_used = brain.last_run_details.get("model", "unknown") if isinstance(brain.last_run_details, dict) else "unknown"
    news_processed = len(news_items) if isinstance(news_items, list) else 0
    signal_yield = round((processed_count / news_processed), 4) if news_processed > 0 else 0.0
    run_observer.finalize(
        status="success",
        summary="Hunt pipeline completed.",
        kpis={
            "news_items_processed": news_processed,
            "signals_generated": processed_count,
            "signal_yield": signal_yield,
            "total_time_seconds": round(_total_run_time, 3),
            "ai_time_seconds": round(_ai_time, 3),
            "news_fetch_time_seconds": round(_news_fetch_time, 3),
            "openrouter_calls_this_run": int(openrouter_this_run or 0),
            "json_repair_needed": bool(brain.last_run_details.get("json_repair_needed", False)) if isinstance(brain.last_run_details, dict) else False,
            "retry_count": int(brain.last_run_details.get("retry_count", 0)) if isinstance(brain.last_run_details, dict) else 0,
        },
        context={
            "model_used": model_used,
            "run_id": run_id,
        },
    )

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
