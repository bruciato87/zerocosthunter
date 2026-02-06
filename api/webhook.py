import os
import logging
import asyncio
import json
from pathlib import Path
from flask import Flask, request, render_template, redirect, session, url_for, flash
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import yfinance as yf
import pandas as pd
from datetime import datetime
import sys
import httpx

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import Brain
from db_handler import DBHandler
from hunter import NewsHunter
from market_data import MarketData
from main import run_async_pipeline
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='../templates')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback_secret_key_change_in_prod")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
# Silence yfinance internal logging (prevents "possibly delisted" spam during suffix rotation)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logger = logging.getLogger("VercelWebhook")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "hunter")
RUN_REPORT_TYPES = ("hunt", "analyze", "rebalance", "trainml")
RUN_REPORT_KEY_PRIORITY = {
    "hunt": ("signals_generated", "news_items_processed", "signal_yield", "openrouter_calls_this_run", "total_time_seconds"),
    "analyze": ("verdict_score", "news_items_fetched", "technicals_loaded", "report_length_chars"),
    "rebalance": ("quant_order_count", "actionable_suggestions_count", "fallback_used", "portfolio_value_eur"),
    "trainml": ("promotions", "rollbacks", "usable_components", "training_samples", "classifier_accuracy"),
}


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dashboard_fast_mode_enabled(force_full: bool = False) -> bool:
    """
    Enable fast dashboard mode by default on Vercel to avoid request timeouts.
    Override with DASHBOARD_FAST_MODE env or `?full=1` query string.
    """
    if force_full:
        return False
    env_override = os.environ.get("DASHBOARD_FAST_MODE")
    if env_override is not None:
        return _is_truthy(env_override)
    return os.environ.get("VERCEL") == "1"


def _load_latest_run_report(run_type: str, reports_dir: str = "run_logs/latest") -> dict:
    """Load and normalize the latest observability report for one run type."""
    normalized = {
        "run_type": run_type,
        "status": "missing",
        "summary": "No report available yet.",
        "completed_at": None,
        "duration_seconds": None,
        "kpis": {},
        "context": {},
        "errors": [],
        "exists": False,
    }

    report_path = Path(reports_dir) / f"{run_type}_latest.json"
    if not report_path.exists():
        return normalized

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Dashboard observability: failed reading %s: %s", report_path, e)
        normalized.update(
            {
                "status": "invalid",
                "summary": f"Invalid report payload: {e}",
            }
        )
        return normalized

    normalized.update(
        {
            "run_type": str(payload.get("run_type", run_type)),
            "status": str(payload.get("status", "unknown")).lower(),
            "summary": str(payload.get("summary", "")),
            "completed_at": payload.get("completed_at"),
            "duration_seconds": payload.get("duration_seconds"),
            "kpis": payload.get("kpis") if isinstance(payload.get("kpis"), dict) else {},
            "context": payload.get("context") if isinstance(payload.get("context"), dict) else {},
            "errors": payload.get("errors") if isinstance(payload.get("errors"), list) else [],
            "exists": True,
        }
    )
    return normalized


def _top_kpis_for_dashboard(report: dict, limit: int = 5) -> list:
    """Select the most relevant KPI pairs for dashboard cards."""
    kpis = report.get("kpis") or {}
    if not isinstance(kpis, dict):
        return []

    run_type = report.get("run_type", "")
    selected = []
    used = set()

    for key in RUN_REPORT_KEY_PRIORITY.get(run_type, ()):
        if key in kpis:
            selected.append((key, kpis.get(key)))
            used.add(key)
        if len(selected) >= limit:
            return selected

    for key in sorted(kpis.keys()):
        if key in used:
            continue
        selected.append((key, kpis.get(key)))
        if len(selected) >= limit:
            break
    return selected


def build_observability_dashboard(reports_dir: str = "run_logs/latest") -> dict:
    """Aggregate latest run observability reports for the web dashboard."""
    runs = []
    for run_type in RUN_REPORT_TYPES:
        report = _load_latest_run_report(run_type=run_type, reports_dir=reports_dir)
        report["kpi_items"] = _top_kpis_for_dashboard(report)
        runs.append(report)
    return {"runs": runs, "reports_dir": reports_dir}

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == DASHBOARD_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            flash('❌ Password non valida.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/favicon.png')
def favicon_png():
    return "", 204

# --- Bot Logic ---



# Simple logic preventing double-execution on Retry
# Local (per-instance) lock + DB (distributed) lock
IS_HUNTING = False

# Deduplication Strategy: Distributed Lock via DB
# Prevents double execution across Vercel instances
DEBOUNCE_SECONDS = 60

def debounce_command(func):
    """Decorator to prevent double execution (Network retries & Manual double-clicks)."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            # 1. Identity & Update ID (Unique for retries)
            chat_id = update.effective_chat.id if update.effective_chat else 0
            update_id = update.update_id
            
            # 2. Content Hash (To block manual duplicates like sending same photo twice)
            content_val = "unknown"
            if update.message:
                if update.message.text: content_val = update.message.text
                elif update.message.document: content_val = update.message.document.file_unique_id
                elif update.message.photo: content_val = update.message.photo[-1].file_unique_id
            elif update.callback_query:
                content_val = update.callback_query.data
                
            import hashlib
            raw_string = f"{chat_id}:{content_val}"
            content_hash = hashlib.md5(raw_string.encode()).hexdigest()
            
            # Check DB Lock using TWO keys to be super safe
            db = DBHandler()
            
            # Layer 1: Block precise Telegram Retries
            if not db.check_and_lock_command(chat_id, str(update_id), DEBOUNCE_SECONDS):
                return
            
            # Layer 2: Block Manual Double-Clicks (same content within 15s)
            if not db.check_and_lock_command(chat_id, content_hash, DEBOUNCE_SECONDS):
                return
            
            # Execute
            await func(update, context)
            
        except Exception as e:
            logger.error(f"Debounce wrapper error: {e}")
            # Fallback: run anyway if lock logic fails to avoid blocking the user
            await func(update, context)
            
    return wrapper

async def hunt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger hunt via GitHub Actions to bypass Vercel timeout limits."""
    import httpx
    
    chat_id = update.effective_chat.id
    
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "bruciato87/zerocosthunter")

    if not github_token:
        # CRITICAL: Disable local fallback on Vercel to save CPU and avoid timeouts
        logger.error("GITHUB_TOKEN not set - Cannot trigger remote hunt.")
        await update.message.reply_text(
            "❌ **Errore Configurazione:**\n"
            "Manca il `GITHUB_TOKEN`. L'esecuzione locale è disabilitata per risparmiare risorse su Vercel.\n"
            "Contatta l'amministratore per configurare i segreti di GitHub."
        )
        return
    
    # Trigger GitHub Actions workflow
    await update.message.reply_text(
        "🏹 **Caccia Avviata!**\n"
        "Ho triggerato l'analisi completa su GitHub Actions.\n"
        "Riceverai i segnali entro 2-3 minuti. 🚀"
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/repos/{github_repo}/actions/workflows/market_scan.yml/dispatches",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                },
                json={
                    "ref": "main",
                    "inputs": {"job_type": "hunt"}
                },
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"GitHub Actions workflow triggered successfully for chat {chat_id}")
            else:
                logger.error(f"GitHub Actions trigger failed: {response.status_code} - {response.text}")
                await update.message.reply_text(f"⚠️ Trigger fallito: {response.status_code}")
                
    except Exception as e:
        logger.error(f"GitHub Actions trigger error: {e}")
        await update.message.reply_text(f"⚠️ Errore trigger: {str(e)[:100]}")

async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    app_url = os.environ.get("APP_URL", "https://zerocosthunter.vercel.app")
    dashboard_url = f"{app_url}/dashboard"
    keyboard = [[InlineKeyboardButton("🖥️ Apri Dashboard Web", url=dashboard_url)]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📊 **Zero-Cost Hunter Dashboard**\n\nClicca qui sotto per vedere i grafici e i segnali completi:",
        reply_markup=reply_markup
    )

@debounce_command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await setup_bot_commands(context.bot)
    await update.message.reply_text(
        "👋 **Benvenuto nel ZeroCostHunter Bot!** 🏹\n\n"
        "Carica lo screenshot del tuo portafoglio (Trade Republic/Fineco) per iniziare.\n"
        "Se il ticker non viene riconosciuto, puoi scriverlo nella didascalia della foto (es. 'ICGA.F').\n\n"
        "Usa /help per vedere la lista dei comandi."
    )

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"SETTINGS COMMAND CALLED. Args: {context.args}")
    args = context.args
    db = DBHandler()
    
    if not args:
        settings = db.get_settings()
        min_conf = settings.get("min_confidence", 0.70)
        only_port = settings.get("only_portfolio", False)
        risk = settings.get("risk_profile", "BALANCED")
        
        risk_icon = "🛡️" if risk == "CONSERVATIVE" else "⚖️" if risk == "BALANCED" else "💎"
        
        msg = (
            "⚙️ **Smart Filters Config**\n\n"
            f"🎯 **Min Confidence:** {int(min_conf * 100)}%\n"
            f"💼 **Portfolio Mode:** {'✅ ON' if only_port else '❌ OFF'}\n"
            f"{risk_icon} **Risk Profile:** `{risk}`\n\n"
            "**Comandi per modificare:**\n"
            "`/settings confidence=80` (Imposta al 80%)\n"
            "`/settings portfolio=on` (Solo asset posseduti)\n"
            "`/settings risk=conservative` (Tighter stops)\n"
            "`/settings risk=aggressive` (Diamond hands)"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    updated = False
    for arg in args:
        try:
            if "confidence=" in arg:
                val = int(arg.split("=")[1])
                if 0 <= val <= 100:
                    db.update_settings(min_confidence=val/100.0)
                    updated = True
            elif "portfolio=" in arg:
                val = arg.split("=")[1].lower()
                if val in ["on", "true", "1"]:
                    db.update_settings(only_portfolio=True)
                    updated = True
                elif val in ["off", "false", "0"]:
                    db.update_settings(only_portfolio=False)
                    updated = True
            elif "risk=" in arg:
                val = arg.split("=")[1].upper()
                if val in ["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]:
                    db.update_settings(risk_profile=val)
                    updated = True
        except Exception as e:
            logger.error(f"Error parsing setting {arg}: {e}")

    if updated:
        await update.message.reply_text("✅ Impostazioni aggiornate!")
    else:
        await update.message.reply_text("❌ Errore/Nessuna modifica. Usa il formato: `/settings confidence=80`")

async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch between PREPROD (Gemini Only) and PROD (Hybrid) modes."""
    db = DBHandler()
    current_settings = db.get_settings()
    current_mode = current_settings.get("app_mode", "PROD")
    
    new_mode = "PROD"
    if context.args:
        arg = context.args[0].upper()
        if arg in ["PREPROD", "GEMINI"]:
            new_mode = "PREPROD"
        elif arg in ["PROD", "HYBRID"]:
            new_mode = "PROD"
    else:
        # Toggle
        new_mode = "PREPROD" if current_mode == "PROD" else "PROD"
        
    db.update_settings(app_mode=new_mode)
    
    status_icon = "🔧" if new_mode == "PREPROD" else "🚀"
    mode_desc = "Gemini Direct ONLY" if new_mode == "PREPROD" else "Hybrid (OpenRouter + Fallback)"
    
    msg = (
        f"{status_icon} **Mode Switched: {new_mode}**\n\n"
        f"Logic: `{mode_desc}`\n"
        f"✅ Settings updated."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show API usage statistics."""
    db = DBHandler()
    usage = db.get_api_usage()
    
    openrouter_count = usage.get("openrouter", 0)
    fallback_count = usage.get("gemini_fallback", 0)
    models = usage.get("models", {})
    date = usage.get("date", "N/A")
    last_model = usage.get("last_model", "N/A")
    
    # Build per-model breakdown
    models_text = ""
    if models:
        for model, count in sorted(models.items(), key=lambda x: x[1], reverse=True)[:5]:
            model_short = model.split('/')[-1].replace(':free', '')
            models_text += f"  └ {model_short}: {count}\n"
    else:
        models_text = "  _Nessun modello usato oggi_\n"
    
    hours_left = usage.get("hours_until_reset", 0)
    reset_time = usage.get("reset_at_local", "01:00 Italy")
    
    msg = (
        f"📊 **API Usage Today** ({date})\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"🤖 **OpenRouter:** {openrouter_count} chiamate\n"
        f"{models_text}\n"
        f"🔄 **Gemini Fallback:** {fallback_count}\n\n"
        f"⏰ **Reset in:** {hours_left:.1f}h ({reset_time})"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

from whale_watcher import WhaleWatcher
from economist import Economist

# ... (Previous imports)

# ... (Previous imports)

async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Manually add an asset.
    Usage: /add TICKER QTY PRICE [SL] [TP]
    """
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "❌ **Uso:** `/add TICKER QTY PRICE [SL] [TP]`\n"
            "Es: `/add AAPL 10 150`\n"
            "Es: `/add NVDA 50 100 90 150` (con SL=90, TP=150)", 
            parse_mode="Markdown"
        )
        return

    try:
        ticker = args[0].upper()
        qty = float(args[1])
        price = float(args[2])
        sl = float(args[3]) if len(args) > 3 else 0
        tp = float(args[4]) if len(args) > 4 else 0

        db = DBHandler()
        db.add_to_portfolio(
            ticker=ticker,
            amount=qty,
            price=price,
            chat_id=chat_id,
            is_confirmed=True,
            stop_loss=sl,
            take_profit=tp
        )
        
        msg = f"✅ **Asset Aggiunto!**\n\n📌 {ticker}\n🔢 Qty: {qty}\n💰 Prezzo: €{price}"
        if sl > 0 or tp > 0:
            msg += f"\n\n🛡️ **Protezione Attiva:**\n🔴 SL: €{sl}\n🟢 TP: €{tp}"
            
        await update.message.reply_text(msg)
    except ValueError:
        await update.message.reply_text("❌ Errore: Assicurati che QTY e PRICE siano numeri validi.")
    except Exception as e:
        logger.error(f"Add command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def protect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Update SL/TP for an asset.
    Usage: /protect TICKER SL TP
    """
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "❌ **Uso:** `/protect TICKER SL TP`\n"
            "Es: `/protect NVDA 90 150` (SL=90, TP=150)\n"
            "Usa 0 per disabilitare un livello.",
            parse_mode="Markdown"
        )
        return

    try:
        ticker = args[0].upper()
        sl = float(args[1])
        tp = float(args[2])

        db = DBHandler()
        if db.update_asset_protection(chat_id, ticker, stop_loss=sl, take_profit=tp):
            await update.message.reply_text(
                f"🛡️ **Protezione Aggiornata per {ticker}**\n\n"
                f"🔴 Stop Loss: €{sl}\n"
                f"🟢 Take Profit: €{tp}"
            )
        else:
            await update.message.reply_text(f"❌ Asset {ticker} non trovato o errore DB.")
    except ValueError:
        await update.message.reply_text("❌ Errore: SL e TP devono essere numeri.")
    except Exception as e:
        logger.error(f"Protect command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def settarget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Update Strategic Targets (Phase 3) for an asset.
    Usage: /settarget TICKER TARGET STOP [HORIZON]
    """
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 3:
        await update.message.reply_text(
            "❌ **Uso:** `/settarget TICKER TARGET STOP [GIORNI]`\n"
            "Es: `/settarget AAPL 250 180 60` (Target €250, Stop €180, Orizzonte 60gg)\n"
            "I target vengono ricalcolati automaticamente dall'AI se non impostati.",
            parse_mode="Markdown"
        )
        return

    try:
        ticker = args[0].upper()
        target = float(args[1])
        stop = float(args[2])
        horizon = int(args[3]) if len(args) > 3 else 30

        db = DBHandler()
        if db.update_portfolio_targets(ticker, target_price=target, stop_loss_price=stop, horizon_days=horizon, target_type='MANUAL'):
            await update.message.reply_text(
                f"🎯 **Target Strategici Aggiornati per {ticker}**\n\n"
                f"💰 Target Profit: €{target}\n"
                f"🛑 Stop Loss: €{stop}\n"
                f"📅 Orizzonte: {horizon} giorni\n"
                f"📝 Tipo: Manuale"
            )
        else:
            await update.message.reply_text(f"❌ Asset {ticker} non trovato nel portfolio confirmed.")
    except ValueError:
        await update.message.reply_text("❌ Errore: Target, Stop e Giorni devono essere numeri.")
    except Exception as e:
        logger.error(f"SetTarget command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def core_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Toggle Core Asset status for an asset.
    Usage: /core TICKER ON/OFF
    """
    chat_id = update.effective_chat.id
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "❌ **Uso:** `/core TICKER ON/OFF`\n"
            "Es: `/core AAPL ON` (Protegge l'asset dalle vendite tecniche)\n"
            "Es: `/core AAPL OFF` (Ripristina gestione standard)",
            parse_mode="Markdown"
        )
        return

    try:
        ticker = args[0].upper()
        status = args[1].upper()
        is_core = status in ["ON", "TRUE", "YES"]

        db = DBHandler()
        if db.update_asset_core(chat_id, ticker, is_core):
            msg = f"🛡️ **Status Core Aggiornato per {ticker}**\n\n"
            msg += f"Status: `{'CORE (Protetto)' if is_core else 'STANDARD'}`\n"
            if is_core:
                msg += "⚠️ L'AI non suggerirà vendite tecniche per questo asset."
            await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Asset {ticker} non trovato o errore DB.")
    except Exception as e:
        logger.error(f"Core command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def macro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏛 Analizzo lo scenario Macro Economico... (VIX, Yields, FED)")
    try:
        from economist import Economist
        eco = Economist()
        summary = eco.get_macro_summary()
        await update.message.reply_text(f"```{summary}```", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Macro Command Fail: {e}")
        await update.message.reply_text("❌ Errore nel recupero dati Macro.")

async def whale_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🐋 Localizzo le Balene (Binance Real-Time)...")
    try:
        from whale_watcher import WhaleWatcher
        ww = WhaleWatcher()
        summary = ww.analyze_flow()
        # Clean up
        await update.message.reply_text(f"```{summary}```", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Whale Command Fail: {e}")
        await update.message.reply_text("❌ Errore Whale Watcher.")

async def harvest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check for Tax-Loss Harvesting opportunities."""
    chat_id = update.effective_chat.id
    try:
        db = DBHandler()
        portfolio = db.get_portfolio(chat_id)
        
        if not portfolio:
            await update.message.reply_text("📉 Il tuo portafoglio è vuoto. Nessuna minusvalenza da analizzare.")
            return
            
        await update.message.reply_text("💰 Analizzo opportunità di Tax-Loss Harvesting (Regime Italiano)...")
        
        from advisor import Advisor
        advisor = Advisor()
        
        # Analyze details
        analysis = advisor.analyze_portfolio(portfolio)
        opportunities = analysis.get("harvest_opportunities", [])
        
        if not opportunities:
            await update.message.reply_text("✅ **Ottime notizie!** Nessuna posizione in perdita significativa da 'rollare' per fini fiscali.")
        else:
            msg = "📉 **Opportunità Tax-Loss Harvesting**\n\n"
            for opp in opportunities:
                msg += f"• **{opp['ticker']}**: {opp['tip']}\n"
            
            msg += "\n⚠️ _Nota: Consultare sempre un commercialista. Le regole su ETF/Crypto sono in evoluzione._"
            await update.message.reply_text(msg, parse_mode="Markdown")
            
    except Exception as e:
        logger.error(f"Harvest Command Fail: {e}")
        await update.message.reply_text("❌ Errore durante l'analisi fiscale.")

async def sectors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze Sector Rotation momentum."""
    await update.message.reply_text("🔄 Analizzo la rotazione settoriale (Momentum)...")
    try:
        from market_data import MarketData, SectorAnalyst
        md = MarketData()
        sa = SectorAnalyst(md)
        
        signals = sa.get_rotation_signals()
        
        if not signals:
            await update.message.reply_text("⚠️ Dati settoriali insufficienti al momento.")
            return
            
        msg = "🔄 **Sector Rotation (Smart Money Flow)**\n\n"
        msg += "\n\n".join(signals)
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Sectors Command Fail: {e}")
        await update.message.reply_text("❌ Errore analisi settoriale.")

async def setup_bot_commands(bot):
    """Configures the menu button in Telegram UI."""
    commands = [
        # Core
        BotCommand("hunt", "🏹 Analisi News (Caccia Segnali)"),
        BotCommand("portfolio", "📊 Portfolio & Prezzi Live"),
        BotCommand("analyze", "🔬 Deep Dive Ticker"),
        BotCommand("rebalance", "⚖️ Suggerimenti Ribilanciamento"),
        BotCommand("harvest", "💰 Tax-Loss Harvesting"),
        BotCommand("sectors", "🔄 Rotazione Settoriale"),
        # Tools
        BotCommand("macro", "🏛 Scenario Macro (VIX/FED)"),
        BotCommand("whale", "🐋 Flussi Whale (Binance)"),
        BotCommand("benchmark", "📈 Confronto con S&P500/BTC"),
        BotCommand("report", "📑 Report Settimanale"),
        BotCommand("backtest", "📉 Backtest Storico Ticker"),
        # Alerts & Sim
        BotCommand("alert", "🔔 Imposta Alert Prezzo"),
        BotCommand("alerts", "📋 Lista Alert Attivi"),
        BotCommand("paper", "🧪 Lab (Trading Simulato)"),
        # ML
        BotCommand("trainml", "🧠 Avvia Training ML"),
        BotCommand("strategy", "🎯 Strategia ML Attuale"),
        # Management
        BotCommand("add", "➕ Aggiungi Asset Manuale"),
        BotCommand("sell", "💸 Registra Vendita"),
        BotCommand("settarget", "🎯 Imposta Target Profit/Stop"),
        BotCommand("protect", "🛡️ Imposta SL/TP (Alert)"),
        BotCommand("delete", "🗑️ Rimuovi Asset"),
        BotCommand("setprice", "💰 Modifica Prezzo Acquisto"),
        BotCommand("setticker", "🏷️ Modifica Ticker"),
        BotCommand("setqty", "🔢 Modifica Quantità"),
        # Core & Risk
        BotCommand("core", "🛡️ Proteggi come Core Asset"),
        # Memory
        BotCommand("recall", "💭 Ricorda Trade Passati"),
        BotCommand("learn", "📚 Insegna Lesson all'AI"),
        # Backtest
        BotCommand("portfolio_backtest", "📈 Backtest Portafoglio"),
        # Settings
        BotCommand("settings", "⚙️ Configura AI Hunter & Rischio"),
        BotCommand("usage", "📊 Statistiche API AI"),
        BotCommand("mode", "🔄 Cambia Modalità AI"),
        BotCommand("dashboard", "🖥️ Web Dashboard"),
        BotCommand("help", "❓ Guida ai Comandi"),
        BotCommand("reset", "☢️ Reset Totale Portafoglio"),
    ]
    await bot.set_my_commands(commands)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await setup_bot_commands(context.bot)
    msg = (
        "🚀 **Zero-Cost Hunter: Guida Completa**\n\n"
        "⚡ **MODI VELOCI PER GESTIRE I TRADE:**\n"
        "1. 📄 **PDF**: Inoltra il file 'Conferma d'ordine' di Trade Republic.\n"
        "2. 📊 **BOTTONI**: Digita `/portfolio` e clicca sui tasti sotto gli asset.\n"
        "3. ✍️ **TESTO**: Scrivi *'Venduti 10 BTC'* o *'Comprato 1 ETF World'*.\n"
        "4. 📷 **FOTO**: Invia uno screenshot del portafoglio.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🏹 **SEGNALI AI**\n"
        "• `/hunt` - Analisi news del giorno\n"
        "• `/analyze <TICKER>` - Deep dive sull'asset\n"
        "• `/rebalance` - Suggerimenti ribilanciamento\n"
        "• `/harvest` - 💰 Tax-Loss Harvesting (Minusvalenze)\n"
        "• `/sectors` - 🔄 Rotazione Settoriale (Momentum)\n\n"
        "📊 **PORTAFOGLIO**\n"
        "• `/portfolio` - Valore live + operazioni rapide\n"
        "• `/add TICKER QTY PRICE` - Aggiungi manualmente\n"
        "• `/sell TICKER QTY PRICE` - Registra vendita\n"
        "• `/settarget TICKER TARGET STOP` - 🎯 Target strategici AI/Manuali\n"
        "• `/protect TICKER SL TP` - Imposta Stop-Loss/Take-Profit (Alert)\n"
        "• `/delete TICKER` - Rimuovi un asset\n"
        "• `/setprice TICKER PRICE` - Modifica prezzo acquisto\n"
        "• `/setticker OLD NEW` - Modifica ticker\n"
        "• `/setqty TICKER QTY` - Modifica quantità\n"
        "• `/core TICKER ON/OFF` - 🛡️ Proteggi asset come 'Core'\n"
        "• `/reset` - ☢️ Cancella TUTTO\n\n"
        "🛠️ **STRUMENTI**\n"
        "• `/macro` - Scenario economico globale\n"
        "• `/whale` - 🐋 Flussi whale (Binance)\n"
        "• `/benchmark` - 📈 Confronto vs S&P500/BTC\n"
        "• `/report` - 📑 Report settimanale\n"
        "• `/backtest <TICKER>` - Backtest storico ticker\n"
        "• `/portfolio_backtest` - 📈 Backtest portafoglio\n"
        "• `/paper` - 🧪 Trading simulato (Lab)\n\n"
        "🔔 **ALERT**\n"
        "• `/alert BTC > 100000` - Imposta alert\n"
        "• `/alerts` - Lista alert attivi\n\n"
        "🧠 **MACHINE LEARNING**\n"
        "• `/trainml` - Avvia training del modello\n"
        "• `/strategy` - Strategia ML attuale\n\n"
        "💭 **MEMORIA AI**\n"
        "• `/recall <TICKER>` - Ricorda trade passati\n"
        "• `/learn <MESSAGGIO>` - Insegna lesson all'AI\n\n"
        "⚙️ **IMPOSTAZIONI & RISK**\n"
        "• `/settings risk=conservative` - 🛡️ Profilo Conservativo (Stop stretti)\n"
        "• `/settings risk=aggressive` - 💎 Profilo Aggressive (Diamond hands)\n"
        "• `/settings` - Configura filtri AI\n"
        "• `/usage` - Statistiche API AI\n"
        "• `/mode` - Cambia modalità (PROD/PREPROD)\n"
        "• `/dashboard` - 🖥️ Web Dashboard\n"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle generic text messages for quick interactions:
    - Quantity input for quick sell flow
    - Smart trade commands (e.g., "Venduto 10 BTC")
    """
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    user_data = context.user_data

    # 1. Quick Sell Flow (Expecting Qty after Clicking "Vendi")
    db = DBHandler()
    state_sell = db.get_user_state(chat_id, 'expecting_sell_qty')
    state_add = db.get_user_state(chat_id, 'expecting_add_qty')
    
    if state_sell:
        ticker = state_sell['ticker']
        db.save_user_state(chat_id, 'expecting_sell_qty', None) 
        try:
            qty = float(text.replace(',', '.'))
            market = MarketData()
            price, _ = await market.get_smart_price_eur_async(ticker)
            
            portfolio = db.get_portfolio(chat_id)
            asset = next((p for p in portfolio if p['ticker'].upper() == ticker.upper() or ticker in p['ticker']), None)
            avg_price = asset.get('avg_price', price) if asset else price

            pending_sell = {
                'ticker': ticker,
                'quantity': qty,
                'price': price,
                'avg_price': avg_price,
                'net_received': qty * price - 1.0,
                'profit': (price - avg_price) * qty,
                'tax': 0.0,
                'commission': 1.0,
                'profit_pct': ((price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            }
            db.save_user_state(chat_id, 'pending_sell', pending_sell)
            
            confirm_msg = (
                f"📋 **Conferma Vendita (Rapida)**\n\n"
                f"📊 **{ticker}**\n"
                f"├ Quantità: {qty}\n"
                f"├ Prezzo Attuale: €{price:.2f}\n"
                f"├ Stima Netto: €{pending_sell['net_received']:.2f}\n\n"
                f"{'🟢' if pending_sell['profit'] >= 0 else '🔴'} **P&L Est:** +€{pending_sell['profit']:.2f} ({pending_sell['profit_pct']:.1f}%)\n\n"
                f"⚠️ **Confermi?**"
            )
            keyboard = [[
                InlineKeyboardButton("✅ Conferma", callback_data="confirm_sell_manual"),
                InlineKeyboardButton("❌ Annulla", callback_data="cancel_sell")
            ]]
            await update.message.reply_text(confirm_msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
            return
        except Exception as e:
            await update.message.reply_text(f"❌ Errore: {e}. Inserisci un numero valido.")
            return

    if state_add:
        ticker = state_add['ticker']
        db.save_user_state(chat_id, 'expecting_add_qty', None)
        try:
            qty = float(text.replace(',', '.'))
            market = MarketData()
            price, _ = await market.get_smart_price_eur_async(ticker)
            
            pending_add = {
                'ticker': ticker,
                'quantity': qty,
                'price': price,
                'net_total': qty * price + 1.0 # Simple TR estimate
            }
            db.save_user_state(chat_id, 'pending_add', pending_add)
            
            confirm_msg = (
                f"📋 **Conferma Acquisto (Rapido)**\n\n"
                f"📊 **{ticker}**\n"
                f"├ Quantità: {qty}\n"
                f"├ Prezzo Attuale: €{price:.2f}\n"
                f"└ **Costo Totale Est: €{pending_add['net_total']:.2f}**\n\n"
                f"⚠️ **Confermi l'aggiunta?**"
            )
            keyboard = [[
                InlineKeyboardButton("✅ Conferma", callback_data="confirm_add_manual"),
                InlineKeyboardButton("❌ Annulla", callback_data="cancel_pdf_trade")
            ]]
            await update.message.reply_text(confirm_msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
            return
        except Exception as e:
            await update.message.reply_text(f"❌ Errore: {e}. Inserisci un numero valido.")
            return

    # 2. Smart Trade Parsing (Regex)
    import re
    sell_match = re.search(r'(?i)(vendut[oi]|sell)\s+([\d,.]+)\s+([a-zA-Z\d\-.=]+)', text)
    buy_match = re.search(r'(?i)(comprat[oi]|buy)\s+([\d,.]+)\s+([a-zA-Z\d\-.=]+)', text)
    
    if sell_match:
        qty = sell_match.group(2).replace(',', '.')
        ticker = sell_match.group(3).upper()
        # Automate the /sell logic
        await update.message.reply_text(f"🔍 **Rilevata vendita:** {qty} {ticker}. Elaboro...")
        # Reuse sell_command or direct logic
        # For simplicity, let's just trigger the expecting_sell_qty flow if price is missing
        # or better, fetch price and ask for confirm.
        pass # To be fully implemented if specific flow needed

    # If no trade detected, just handle normally (maybe ignore or echo)
    logger.debug(f"Unhandled text: {text}")

@debounce_command
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming PDF documents from Trade Republic."""
    doc = update.message.document
    if not doc.file_name.lower().endswith('.pdf'):
        return

    await update.message.reply_text("📄 **Analizzo il documento Trade Republic...**", parse_mode="Markdown")
    try:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        file_obj = await doc.get_file()
        file_path = f"/tmp/doc_{user_id}_{doc.file_name}"
        logger.info(f"Downloading PDF: {doc.file_name} for user {user_id}")
        await file_obj.download_to_drive(file_path)

        from brain import Brain
        brain = Brain()
        db = DBHandler()
        logger.info(f"Starting brain analysis for {file_path}")
        trade_data = brain.parse_trade_republic_pdf(file_path)
        logger.info(f"Brain analysis completed for {file_path}")

        if "error" in trade_data:
            await update.message.reply_text(f"❌ Errore durante l'analisi del PDF: {trade_data['error']}")
            return

        # Prepare confirmation
        ticker = trade_data.get('ticker', 'UNKNOWN')
        action = trade_data.get('action', 'BUY').upper()
        qty = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        net_total = trade_data.get('net_total', 0)
        asset_name = trade_data.get('asset_name', ticker)

        # Store in DB for confirmation (Vercel Stateless Fix)
        if action == "SELL":
            pending_sell = {
                'ticker': ticker,
                'quantity': qty,
                'price': price,
                'tax': trade_data.get('tax', 0),
                'commission': trade_data.get('commission', 1.0),
                'net_received': net_total,
                'profit': 0.0, # Will be calculated if possible
                'profit_pct': 0.0,
                'asset_name': trade_data.get('asset_name', ticker)
            }
            db.save_user_state(chat_id, 'pending_sell', pending_sell)
            confirm_msg = (
                f"📋 **Conferma VENDITA (da PDF)**\n\n"
                f"📊 **{asset_name}**\n"
                f"├ Quantità: {qty}\n"
                f"├ Prezzo: €{price:.2f}\n"
                f"└ **Netto Ricevuto: €{net_total:.2f}**\n\n"
                f"⚠️ **Confermi?**"
            )
            callback_confirm = "confirm_sell"
        else:
            pending_add = {
                'ticker': ticker,
                'quantity': qty,
                'price': price,
                'asset_name': asset_name
            }
            db.save_user_state(chat_id, 'pending_add', pending_add)
            confirm_msg = (
                f"📋 **Conferma ACQUISTO (da PDF)**\n\n"
                f"📊 **{asset_name}**\n"
                f"├ Quantità: {qty}\n"
                f"├ Prezzo: €{price:.2f}\n"
                f"└ **Costo Totale: €{net_total:.2f}**\n\n"
                f"⚠️ **Confermi l'aggiunta al portafoglio?**"
            )
            callback_confirm = "confirm_pdf_buy"

        keyboard = [[
            InlineKeyboardButton("✅ Conferma", callback_data=callback_confirm),
            InlineKeyboardButton("❌ Annulla", callback_data="cancel_pdf_trade")
        ]]
        await update.message.reply_text(confirm_msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Handle document error: {e}")
        await update.message.reply_text(f"❌ Errore sistema: {e}")

@debounce_command
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if we are waiting for a SELL screenshot
    if context.user_data.get('expecting_sell_screenshot'):
        await handle_sell_photo(update, context)
        # Clear flag after handling (or handle_sell_photo deals with state)
        # We'll let handle_sell_photo manage the flow
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    await update.message.reply_text("👀 Analizzo l'immagine...")
    try:
        photo = update.message.photo[-1]
        file_obj = await photo.get_file()
        file_path = f"/tmp/photo_{user_id}.jpg"
        await file_obj.download_to_drive(file_path)

        brain = Brain()
        holdings = brain.parse_portfolio_from_image(file_path)
        
        if not holdings:
            await update.message.reply_text("❌ Non ho trovato dati validi.")
            return

        caption = update.message.caption
        if caption and len(holdings) > 0:
            manual_ticker = caption.strip().upper()
            holdings[0]['ticker'] = manual_ticker
            logger.info(f"Manual Override: Set ticker to {manual_ticker} from caption.")
            await update.message.reply_text(f"✍️ **Override:** Uso il ticker manuale `{manual_ticker}`.")

        db = DBHandler()
        existing_drafts = db.get_drafts(chat_id)
        recent_confirmed = db.get_recent_confirmed_portfolio(chat_id, minutes=5)
        
        msg_text = "✅ **Dati Estratti (Bozza):**\n"
        show_confirm_button = True
        
        for item in holdings:
            merged = False
            new_ticker = item.get('ticker')
            new_qty = item.get('quantity')
            new_price = item.get('avg_price')
            
            def find_merge_candidate(candidate_list):
                 for c in candidate_list:
                      db_name = c.get('asset_name', '').lower()
                      new_name_chk = item.get('name', '').lower()
                      if db_name and new_name_chk:
                           if "china" in new_name_chk and "china" not in db_name: continue
                           if "world" in new_name_chk and "world" not in db_name: continue
                           if "sp500" in new_name_chk and "500" not in db_name: continue
                      if c['ticker'] == 'UNKNOWN' and c['quantity'] and c['quantity'] > 0 and new_ticker and new_ticker != "UNKNOWN":
                           return c, "update_ticker"
                      if new_ticker and c['ticker'] == new_ticker and new_qty and new_qty > 0:
                           return c, "update_qty"
                      if (not new_ticker or new_ticker == "UNKNOWN") and c['ticker'] != "UNKNOWN" and new_qty and new_qty > 0:
                           return c, "update_qty"
                 return None, None

            draft, action = find_merge_candidate(existing_drafts)
            if draft:
                if action == "update_qty":
                    db.update_draft_quantity(draft['id'], new_qty, new_price)
                    msg_text = "🧩 **Dati Integrati:**\n• " + f"{draft['ticker']}: Quantità **{new_qty}**\n_(Usa 'Conferma' precedente)_"
                elif action == "update_ticker":
                    db.update_draft_ticker(draft['id'], new_ticker, new_price)
                    msg_text = "🧩 **Dati Integrati:**\n• " + f"{new_ticker}: Quantità **{draft['quantity']}**\n_(Usa 'Conferma' precedente)_"
                merged = True
                show_confirm_button = False

            if not merged:
                conf_item, action = find_merge_candidate(recent_confirmed)
                if conf_item:
                    if action == "update_qty":
                        db.update_draft_quantity(conf_item['id'], new_qty, new_price)
                        msg_text = "♻️ **Aggiornato (Già Confermato):**\n• " + f"{conf_item['ticker']}: Quantità **{new_qty}**\n"
                    elif action == "update_ticker":
                        db.update_draft_ticker(conf_item['id'], new_ticker, new_price)
                        msg_text = "♻️ **Aggiornato (Già Confermato):**\n• " + f"{new_ticker}: Identificato asset.\n"
                    merged = True
                    show_confirm_button = False

            if not merged:
                db.add_to_portfolio(
                    ticker=new_ticker if new_ticker else "UNKNOWN", 
                    amount=new_qty, 
                    price=new_price, 
                    sector=item.get('sector', 'Unknown'),
                    asset_name=item.get('name'),
                    asset_type=item.get('asset_type', 'Unknown'),
                    is_confirmed=False, 
                    chat_id=chat_id
                )
                display_name = item.get('name') or new_ticker
                msg_text += f"• {display_name}: {new_qty} @ €{new_price}\n"

        if show_confirm_button:
            keyboard = [[InlineKeyboardButton("✅ Conferma e Salva", callback_data="confirm_save"), InlineKeyboardButton("❌ Annulla", callback_data="cancel_save")]]
            await update.message.reply_text(msg_text, reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            await update.message.reply_text(msg_text)

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Errore interno.")

@debounce_command
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    db = DBHandler()
    chat_id = update.effective_chat.id
    
    # --- QUICK ACTION HANDLERS ---
    if query.data == "start_quick_sell":
        await query.message.reply_text("💸 **Vendita Rapida**\nScrivi il ticker e la quantità (es: `AAPL 10`) o invia uno screenshot.")
        return
    elif query.data == "start_quick_add":
         await query.message.reply_text("➕ **Aggiunta Rapida**\nUsa `/add TICKER QTY PREZZO` o invia uno screenshot.")
         return
    
    # Handle Dynamic Asset Callbacks (e.g. sel_BTC-USD)
    if query.data.startswith("sel_"):
        ticker = query.data.replace("sel_", "")
        await query.message.reply_text(f"💸 **Vendi {ticker}**\nQuante quote vuoi vendere? Rispondi con il numero.")
        db.save_user_state(chat_id, 'expecting_sell_qty', {'ticker': ticker})
        return
    elif query.data.startswith("add_"):
        ticker = query.data.replace("add_", "")
        await query.message.reply_text(f"➕ **Aggiungi {ticker}**\nQuante quote vuoi aggiungere? Rispondi con il numero.")
        db.save_user_state(chat_id, 'expecting_add_qty', {'ticker': ticker})
        return
    elif query.data.startswith("prot_"):
        # Toggle Core Status
        ticker = query.data.replace("prot_", "")
        portfolio = db.get_portfolio(chat_id=chat_id)
        asset = next((a for a in portfolio if a['ticker'].upper() == ticker.upper()), None)
        if asset:
            new_status = not asset.get('is_core', False)
            if db.update_asset_core(chat_id, ticker, new_status):
                status_str = "PROTETTO 🛡️" if new_status else "STANDARD 🔓"
                await query.answer(f"{ticker}: {status_str}")
                # Refresh portfolio view? Or just update text. 
                # For now just notify and user can refresh with /portfolio
                await query.message.reply_text(f"🛡️ **{ticker}** impostato come `{status_str}`.")
            else:
                await query.answer("Errore aggiornamento.")
        return
    
    # --- SELL CALLBACKS ---
    elif query.data == "cancel_sell":
        db.save_user_state(chat_id, 'pending_sell', None)
        await query.edit_message_text("❌ Operazione di vendita annullata.")

    elif query.data == "confirm_sell" or query.data == "confirm_sell_manual":
        # [FIX] Persistent State from DB instead of context.user_data
        pending = db.get_user_state(chat_id, 'pending_sell')
        if not pending:
            await query.edit_message_text("❌ Nessuna vendita in attesa (Scaduta o Reset).")
            return
        
        # Execute the sale
        ticker = pending['ticker']
        quantity = pending['quantity']
        price = pending['price']
        
        # Determine values based on source (manual vs screenshot)
        if query.data == "confirm_sell_manual":
             # Calculate estimates for manual confirm
             avg_price = pending.get('avg_price', 0)
             gross_total = quantity * price
             realized_pnl = (price - avg_price) * quantity
             
             TR_COMMISSION = 1.00
             CRYPTO_TAX_RATE = 0.33
             crypto_tickers = ['BTC', 'ETH', 'SOL', 'XRP', 'RENDER', 'ADA', 'AVAX', 'DOT', 'LINK', 'DOGE']
             is_crypto = any(c in ticker.upper() for c in crypto_tickers)
             
             if realized_pnl > 0 and is_crypto:
                 tax = realized_pnl * CRYPTO_TAX_RATE
             else:
                 tax = 0.0
                 
             commission = TR_COMMISSION
             profit = realized_pnl
             net_received = gross_total - tax - commission
        else:
             # Use precise OCR values
             net_received = pending['net_received']
             profit = pending['profit']
             tax = pending['tax']
             tax = pending['tax']
             commission = pending['commission']
        
        # Log transaction with real net values
        result = db.log_transaction(
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price_per_unit=price,
            realized_pnl=profit
        )

        if result:
            # --- PORTFOLIO UPDATE LOGIC ---
            # Try to find existing asset to update quantity
            portfolio = db.get_portfolio(chat_id=chat_id)
            
            # Match priority: Ticker -> Exact Name -> Partial Name
            matched_asset = next((p for p in portfolio if p['ticker'].upper() == ticker.upper()), None)
            
            # Helper to get name safely
            def get_p_name(p):
                return p.get('asset_name') or p.get('name') or ''

            if not matched_asset and pending.get('asset_name'):
                name_to_match = pending['asset_name'].lower()
                # Check 1: Exact Name
                matched_asset = next((p for p in portfolio if get_p_name(p).lower() == name_to_match), None)
            
            if not matched_asset and pending.get('asset_name'):
                name_to_match = pending['asset_name'].lower()
                # Check 2: PDF Name contains DB Name
                matched_asset = next((p for p in portfolio if get_p_name(p) and get_p_name(p).lower() in name_to_match), None)
                
                # Check 3: DB Name contains PDF Name (Reverse)
                if not matched_asset:
                    matched_asset = next((p for p in portfolio if get_p_name(p) and name_to_match in get_p_name(p).lower()), None)

            if matched_asset:
                # Use the REAL ticker from the portfolio if we matched by name
                real_ticker = matched_asset['ticker']
                current_qty = float(matched_asset.get('quantity', 0))
                new_qty = current_qty - quantity
                
                if new_qty <= 0:
                    db.delete_asset(chat_id, real_ticker)
                    portfolio_msg = f"\n\n🗑️ Asset {real_ticker} rimosso (vendita totale)"
                else:
                    db.update_asset_quantity(chat_id, real_ticker, new_qty)
                    portfolio_msg = f"\n\n📉 {real_ticker} aggiornato: {new_qty:.6f} rimanenti"
            else:
                # Asset not found in portfolio (maybe already deleted or mapped wrong)
                logger.warning(f"Sold asset {ticker} ({pending.get('asset_name')}) not found in portfolio.")
                portfolio_msg = "\n\n⚠️ Vendita registrata, ma asset non trovato nel portfolio."
        else:
            portfolio_msg = ""
        
        pnl_emoji = "🟢" if profit >= 0 else "🔴"
        
        final_msg = (
            f"✅ **Vendita Confermata!**\n\n"
            f"📊 **{ticker}**\n"
            f"├ Quantità: {quantity}\n"
            f"├ Prezzo: €{price:.2f}\n"
            f"├ Imposta: -€{tax:.2f}\n"
            f"├ Commissione: -€{commission:.2f}\n"
            f"├ **Netto Ricevuto: €{net_received:.2f}**\n\n"
            f"{pnl_emoji} **P&L Netto:** +€{profit:.2f}"
            f"{portfolio_msg}"
        )
        
        context.user_data.pop('pending_sell', None)
        await query.edit_message_text(final_msg, parse_mode="Markdown")

    # --- PORTFOLIO CALLBACKS ---
    elif query.data == "confirm_save":
        try:
            db.confirm_portfolio(chat_id)
            await query.edit_message_text(text="🚀 **Portafoglio Aggiornato!**")
        except Exception as e:
            await query.edit_message_text(text=f"❌ Errore DB: {e}")
    elif query.data == "cancel_save":
        try:
            db.delete_drafts(chat_id)
            await query.edit_message_text(text="🗑️ Operazione annullata.")
        except Exception as e:
            await query.edit_message_text(text=f"❌ Errore: {e}")
    elif query.data == "confirm_pdf_buy" or query.data == "confirm_add_manual":
        try:
            # [FIX] Persistent State from DB
            pending = db.get_user_state(chat_id, 'pending_add')
            if not pending:
                await query.edit_message_text("❌ Nessun acquisto in attesa.")
                return
            
            ticker = pending['ticker']
            quantity = pending['quantity']
            price = pending['price']
            asset_name = pending.get('asset_name', ticker)
            
            # --- PORTFOLIO UPDATE LOGIC (Smart Match) ---
            portfolio = db.get_portfolio(chat_id=chat_id)
            
            # Match priority: Ticker -> Exact Name -> Partial Name
            matched_asset = next((p for p in portfolio if p['ticker'].upper() == ticker.upper()), None)
            
            # Helper to get name safely
            def get_p_name(p):
                return p.get('asset_name') or p.get('name') or ''

            if not matched_asset and asset_name:
                name_to_match = asset_name.lower()
                matched_asset = next((p for p in portfolio if get_p_name(p).lower() == name_to_match), None)

            if not matched_asset and asset_name:
                name_to_match = asset_name.lower()
                # PDF Name in DB Name
                matched_asset = next((p for p in portfolio if name_to_match in get_p_name(p).lower()), None)
                
                # DB Name in PDF Name (Reverse)
                if not matched_asset:
                    matched_asset = next((p for p in portfolio if get_p_name(p) and get_p_name(p).lower() in name_to_match), None)
            
            if not matched_asset:
                 # Ensure we don't duplicate simple tickers
                 matched_asset = next((p for p in portfolio if p['ticker'] == ticker), None)

            if matched_asset:
                real_ticker = matched_asset['ticker']
                current_qty = float(matched_asset.get('quantity', 0))
                new_qty = current_qty + quantity
                db.update_asset_quantity(chat_id, real_ticker, new_qty)
                msg_action = f"🔄 **{real_ticker}** Aggiornato (+{quantity})"
            else:
                 db.add_to_portfolio(
                    chat_id=chat_id,
                    ticker=ticker,
                    quantity=quantity,
                    purchase_price=price,
                    name=asset_name,
                    is_confirmed=True
                 )
                 msg_action = f"✅ **{ticker}** Aggiunto al Portafoglio!"
            
            db.save_user_state(chat_id, 'pending_add', None)
            await query.edit_message_text(f"🚀 **Acquisto Confermato!**\n\n{msg_action}")
        except Exception as e:
            await query.edit_message_text(f"❌ Errore DB: {e}")
            
    elif query.data == "cancel_pdf_trade":
        db.save_user_state(chat_id, 'pending_add', None)
        db.save_user_state(chat_id, 'pending_sell', None)
        await query.edit_message_text("🗑️ Operazione annullata.")

    elif query.data == "confirm_reset":
        try:
           if db.delete_portfolio(chat_id):
               await query.edit_message_text(text="☢️ **BOOM! Reset completato.**\nIl portafoglio è stato raso al suolo.")
           else:
               await query.edit_message_text(text="❌ Errore reset.")
        except Exception as e:
            await query.edit_message_text(text=f"❌ Errore: {e}")
    elif query.data == "cancel_reset":
        await query.edit_message_text(text="😮‍💨 **Reset Annullato.**\nI tuoi asset sono salvi.")



async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    db = DBHandler()
    portfolio = db.get_portfolio(chat_id=chat_id)
    if not portfolio:
        await update.message.reply_text("📂 Il tuo portafoglio è vuoto.")
        return
    await update.message.reply_text("⏳ **Recupero prezzi live...**")
    
    # MarketData handles all pricing and FX internally
    market = MarketData()
    
    msg = "📊 **Il tuo Portafoglio:**\n\n"
    total_val = 0.0
    # Note: eur_usd local is no longer needed here as market logic is centralized

    TICKER_FIX_MAP = {} # Centralized in MarketData



    # --- Optimized Parallel Price Fetching (Vercel Fix) ---
    async def fetch_item_data(item):
        ticker = item.get('ticker', 'N/A')
        search_ticker = TICKER_FIX_MAP.get(ticker, ticker)
        qty = item.get('quantity', 0)
        curr_val = 0.0
        
        if search_ticker and search_ticker != "UNKNOWN":
            try:
                # Use Async MarketData Logic for parallelism
                found_price, used_ticker = await market.get_smart_price_eur_async(search_ticker)
                if found_price > 0:
                    curr_val = qty * found_price
                    # Note: used_ticker ignored for display to keep original cleaned ticker
            except Exception as e:
                logger.error(f"Price error for {search_ticker}: {e}")
        
        return {**item, 'current_value': curr_val, 'display_ticker': ticker}

    # Gather all prices in parallel
    results = await asyncio.gather(*[fetch_item_data(item) for item in portfolio])
    total_val = sum(res['current_value'] for res in results)

    grouped_assets = {"Crypto": [], "Stock": [], "ETF": [], "Other": []}
    for res in results:
        # Determine Group
        a_type = res.get('asset_type', 'Unknown')
        if a_type in grouped_assets:
            grouped_assets[a_type].append(res)
        else:
            grouped_assets["Other"].append(res)

    # Build Message with Headers and Sorting
    # Order of Categories
    cat_order = ["Crypto", "Stock", "ETF", "Other"]
    
    for cat in cat_order:
        assets = grouped_assets.get(cat, [])
        if not assets: continue
        
        # Sort by Value Descending
        assets.sort(key=lambda x: x['current_value'], reverse=True)
        
        # Category Header
        cat_total = sum(a['current_value'] for a in assets)
        msg += f"\n━━━━━━━━━━━━━━━━━━━━\n📁 **{cat}** (Tot: €{cat_total:,.2f})\n"
        
        for a in assets:
            curr_val = a['current_value']
            qty = a['quantity']
            unit_price = curr_val / qty if qty > 0 else 0
            
            val_str = f"€{curr_val:,.2f}" if curr_val > 0 else "N/A"
            unit_price_str = f"€{unit_price:,.2f}" if unit_price > 0 else "N/A"
            icon = "🪙" if cat == "Crypto" else "📈" if cat == "Stock" else "📊"
            
            # Cleaner, less indented format with spacing
            msg += f"\n{icon} **{a.get('asset_name') or a['display_ticker']}**"
            msg += " 🛡️" if a.get('is_core') else ""
            msg += f"\n   `{a['display_ticker']}`  •  `{unit_price_str}`  •  {qty} pz  •  **{val_str}**\n"
            
            if a.get('stop_loss') or a.get('take_profit') or a.get('is_core'):
                sl_info = f"🛑 SL: €{a['stop_loss']:.2f}" if a.get('stop_loss') else "🛑 SL: N/A"
                tp_info = f"💰 TP: €{a['take_profit']:.2f}" if a.get('take_profit') else "💰 TP: N/A"
            # Action buttons for top assets 
            if len(results) <= 12:
                from telegram import InlineKeyboardButton
                is_core = a.get('is_core', False)
                prot_label = "🔓 Unprotect" if is_core else "🛡️ Protect"
                ticker_btn = a['display_ticker']
                # Add a row for this asset
                if 'asset_keyboards' not in locals():
                    asset_keyboards = []
                
                asset_keyboards.append([
                    InlineKeyboardButton(f"💸 {ticker_btn}", callback_data=f"sel_{ticker_btn}"),
                    InlineKeyboardButton(f"➕ Agg", callback_data=f"add_{ticker_btn}"),
                    InlineKeyboardButton(f"{prot_label}", callback_data=f"prot_{ticker_btn}")
                ])
            
    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n💰 **TOTALE PORTAFOGLIO:** `€{total_val:,.2f}`"
    
    # Simple strategy: Add a "Quick Sell" button that asks for ticker
    footer_keyboard = []
    if 'asset_keyboards' in locals() and asset_keyboards:
        footer_keyboard.extend(asset_keyboards)
    
    footer_keyboard.extend([
        [InlineKeyboardButton("💸 Vendita Rapida", callback_data="start_quick_sell")],
        [InlineKeyboardButton("➕ Aggiungi Asset", callback_data="start_quick_add")]
    ])
    
    await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(footer_keyboard), parse_mode="Markdown")

async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("⚠️ Uso: `/delete <TICKER>`")
        return
    ticker = context.args[0].upper()
    if DBHandler().delete_asset(chat_id, ticker):
        await update.message.reply_text(f"🗑️ Eliminato `{ticker}`.")
    else:
        await update.message.reply_text("❌ Errore o non trovato.")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔴 Conferma Reset (IRREVERSIBILE)", callback_data="confirm_reset")],
        [InlineKeyboardButton("🟢 ANNULLA", callback_data="cancel_reset")]
    ]
    await update.message.reply_text(
        "⚠️ **ATTENZIONE: RESET TOTALE** ⚠️\n\n"
        "Stai per cancellare **TUTTO** il portafoglio.\n"
        "Questa azione non può essere annullata.\n"
        "Sei sicuro?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def setprice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("⚠️ Uso: `/setprice <TICKER> <Price>`")
        return
    try:
        if DBHandler().update_asset_price(update.effective_chat.id, context.args[0].upper(), float(context.args[1].replace(',','.'))):
            await update.message.reply_text("✅ Aggiornato.")
        else:
            await update.message.reply_text("❌ Non trovato.")
    except Exception as e:
        logger.debug(f"setprice_command invalid input: {e}")
        await update.message.reply_text("⚠️ Errore formato.")

async def setqty_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("⚠️ Uso: `/setqty <TICKER> <QTY>`")
        return
    try:
        qty = float(context.args[1].replace(',', '.'))
        if DBHandler().update_asset_quantity(update.effective_chat.id, context.args[0].upper(), qty):
            await update.message.reply_text(f"✅ Quantità di {context.args[0].upper()} impostata a {qty}.")
        else:
            await update.message.reply_text("❌ Ticker non trovato in portafoglio.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Errore formato: {e}")

async def setticker_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("⚠️ Uso: `/setticker <OLD> <NEW>`")
        return
    if DBHandler().update_asset_ticker(update.effective_chat.id, context.args[0].upper(), context.args[1].upper()):
        await update.message.reply_text("✅ Aggiornato.")
    else:
        await update.message.reply_text("❌ Non trovato.")

# --- Webhook & Dashboard ---

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    force_full = _is_truthy(request.args.get("full", "0"))
    fast_mode = _dashboard_fast_mode_enabled(force_full=force_full)
    logger.info("Dashboard request mode: %s", "FAST" if fast_mode else "FULL")

    db = DBHandler()
    market = MarketData()
    
    
    # 1. Signals
    try:
        signals = db.supabase.table("predictions").select("*").order("created_at", desc=True).limit(25).execute().data
    except Exception as e:
        logger.warning(f"Dashboard: failed to fetch predictions: {e}")
        signals = []

    # 2. Portfolio Valuation with Forward-Fill Chart
    portfolio = db.get_portfolio()
    total_val, total_inv = 0.0, 0.0
    asset_trends = {}  # Track each asset's trend separately for forward-fill
    
    # FX & Tickers
    eur_usd = MarketData._eur_usd_rate_shared
    
    TICKER_FIX = {} # Centralized in MarketData

    for item in portfolio:
        try:
            qty = item.get('quantity', 0)
            price = item.get('avg_price', 0)
            cost = qty * price
            total_inv += cost
            
            ticker = item.get('ticker','UNKNOWN')
            search = TICKER_FIX.get(ticker, ticker)
            curr = 0.0
            
            if search != "UNKNOWN":
                # Use Global Smart Fetch via MarketData
                price_eur, used_ticker = market.get_smart_price_eur(search)
                if price_eur > 0:
                    curr = qty * price_eur
                    
                    # Fetch history for Chart using the CORRECT ticker
                    # We utilize the resolved 'used_ticker' (e.g. XRP-USD)
                    try:
                        t_obj = yf.Ticker(used_ticker if used_ticker else search)
                        hist = t_obj.history(period="5d")
                        
                        if not hist.empty:
                            asset_trends[ticker] = {}
                            for dt, val in hist['Close'].items():
                                d_str = dt.strftime("%Y-%m-%d")
                                # Convert history to EUR if needed
                                # formatting note: used_ticker usually has suffix if EUR, or is -USD.
                                # But we need to know if we divide by USD.
                                # Heuristic: If price_eur was derived via division, then history needs division.
                                # fetch_price_smart returns pure EUR.
                                # But here we have history objects.
                                # Re-deriving is_eu from used_ticker
                                is_usd_pair = used_ticker.endswith('-USD') or (used_ticker == search and '.' not in search)
                                is_eur_pair = used_ticker.endswith(('.DE', '.F', '.MI', '.PA')) or "EUR" in used_ticker
                                
                                # If it's a USD pair, convert. If EUR pair, keep.
                                val_eur = val
                                if is_usd_pair and not is_eur_pair:
                                     val_eur = val / eur_usd
                                
                                asset_trends[ticker][d_str] = qty * val_eur
                    except Exception as e:
                        logger.debug(f"Dashboard: history fetch failed for {ticker}: {e}")

            total_val += curr
            item['live_value_eur'] = round(curr, 2)
            item['pnl_eur'] = round(curr - cost, 2)
            item['pnl_percent'] = round(((curr - cost)/cost)*100, 2) if cost > 0 else 0
        except Exception as e:
            logger.debug(f"Dashboard: portfolio item processing failed for {item}: {e}")

    # Build chart with forward-fill to handle missing data
    all_dates = sorted(set(d for trends in asset_trends.values() for d in trends.keys()))
    daily_trend = {}
    
    for date in all_dates:
        total_for_date = 0.0
        for ticker, trends in asset_trends.items():
            if date in trends:
                total_for_date += trends[date]
            else:
                # Forward-fill: use last known value
                earlier_dates = [d for d in trends.keys() if d < date]
                if earlier_dates:
                    total_for_date += trends[max(earlier_dates)]
        daily_trend[date] = total_for_date

    # Calculate Total P&L (Unrealized + Realized)
    try:
        realized_pl = db.get_total_realized_pnl()
    except Exception as e:
        logger.warning(f"Dashboard: failed to fetch realized PnL: {e}")
        realized_pl = 0.0

    unrealized_pl = total_val - total_inv
    total_pl = unrealized_pl + realized_pl
    
    # ROI %: Based on currently invested capital (standard for active tracking)
    # Alternatively could be on Total Capital deployed ever, but current invested is better for 'Active' view.
    pl_pct = (total_pl / total_inv * 100) if total_inv > 0 else 0
    
    dates = sorted(daily_trend.keys())
    chart_d = [round(daily_trend[d],2) for d in dates]

    # 3. Analytics
    last_run = "Mai"
    last_run_iso = None
    
    # Last Run
    try:
        logs = db.supabase.table("logs").select("created_at").eq("module","Hunter").order("created_at",desc=True).limit(1).execute().data
        if logs: 
            last_run_iso = logs[0]['created_at']
            last_run = datetime.fromisoformat(last_run_iso.replace('Z','+00:00')).strftime("%d/%m/%Y %H:%M")
    except Exception as e:
        logger.debug(f"Dashboard: failed to fetch last run timestamp: {e}")

    # 4. Audit Stats
    audit_stats = db.get_audit_stats()

    # 5. Market Mood (Insider)
    from insider import Insider
    insider = Insider()
    market_mood = insider.get_market_mood()

    # 6. Advisor (Risk Manager)
    if fast_mode:
        advisor_analysis = {
            "tips": [],
            "note": "Fast dashboard mode: advisor deep analysis skipped to avoid timeout on Vercel.",
        }
    else:
        from advisor import Advisor
        adv = Advisor()
        # Convert row objects to dicts if needed, or pass as is (Advisor expects dict-like with 'ticker', 'quantity')
        # portfolio lines are rows from supabase.
        advisor_analysis = adv.analyze_portfolio(portfolio)

    # 7. Macro Strategist
    from economist import Economist # Fetch Macro Stats
    try:
        macro_stats = Economist().get_dashboard_stats()
    except Exception as e:
        logger.error(f"Macro Stats Error: {e}")
        macro_stats = None

    # Fetch Whale Stats
    if fast_mode:
        whale_stats = None
    else:
        from whale_watcher import WhaleWatcher
        try:
            whale_stats = WhaleWatcher().get_dashboard_stats()
        except Exception as e:
            logger.error(f"Whale Stats Error: {e}")
            whale_stats = None

    # 8. Trade History
    try:
        history = db.supabase.table("signal_tracking").select("*").in_("status", ["WIN", "LOSS", "EXPIRED"]).order("updated_at", desc=True).limit(50).execute().data
        # Parse dates to nicer string in python if needed, or do it in jinja
        for h in history:
            try:
                if h.get('created_at'):
                    h['created_at_fmt'] = datetime.fromisoformat(h['created_at'].replace('Z', '+00:00')).strftime("%d/%m/%Y")
                if h.get('updated_at'):
                    h['updated_at_fmt'] = datetime.fromisoformat(h['updated_at'].replace('Z', '+00:00')).strftime("%d/%m/%Y")
            except Exception as e:
                logger.debug(f"Dashboard: failed to format trade-history date for row {h.get('id')}: {e}")
    except Exception as e:
        logger.error(f"History Fetch Error: {e}")
        history = []

    # 9. Paper Trading (Lab)
    if fast_mode:
        paper_portfolio_enriched = []
        paper_total_value = 0.0
    else:
        try:
            from paper_trader import PaperTrader
            pt = PaperTrader()
            paper_raw = pt.get_portfolio(chat_id=None) # Fetch All
            
            paper_total_value = 0.0
            paper_portfolio_enriched = []
            
            for p in paper_raw:
                qty = p.get('quantity', 0)
                avg = p.get('avg_price', 0)
                ticker = p.get('ticker', 'UNKNOWN')
                
                # Use market data for price
                curr_p, _ = market.get_smart_price_eur(ticker)
                if curr_p <= 0: curr_p = avg # Fallback
                
                val = qty * curr_p
                paper_total_value += val
                
                cost = qty * avg
                pnl = val - cost
                pnl_pct = ((pnl) / cost * 100) if cost > 0 else 0.0
                
                paper_portfolio_enriched.append({
                    "ticker": ticker,
                    "quantity": qty,
                    "avg_price": avg,
                    "current_price": curr_p,
                    "pnl": pnl_pct,
                    "current_value": val
                })
                
        except Exception as e:
            logger.error(f"Paper Dashboard Error: {e}")
            paper_portfolio_enriched = []
            paper_total_value = 0.0

    # 10. Backtest Results (Lab)
    try:
        backtest_results = db.supabase.table("backtest_results") \
            .select("*") \
            .order("run_at", desc=True) \
            .limit(10) \
            .execute().data
    except Exception as e:
        logger.error(f"Backtest Fetch Error: {e}")
        backtest_results = []

    # 11. Benchmark Data (Phase 17)
    if fast_mode:
        benchmark_data = {}
    else:
        try:
            from benchmark import Benchmark
            bench = Benchmark()
            benchmark_data = bench.compare_vs_benchmarks(30)
        except Exception as e:
            logger.error(f"Benchmark Fetch Error: {e}")
            benchmark_data = {}

    # 12. Level 2 Predictive Data (Phase 4)
    # 12. Level 2 Predictive Data (Phase 4)
    try:
        from market_regime import MarketRegimeClassifier
        regime_classifier = MarketRegimeClassifier()
        market_regime = regime_classifier.classify()
    except Exception as e:
        logger.error(f"L2 Regime Fetch Error: {e}")
        market_regime = {}

    if fast_mode:
        sector_rotation = {"ranking": []}
    else:
        try:
            from market_data import SectorAnalyst
            sector_analyst = SectorAnalyst()
            # Adapt SectorAnalyst output for Dashboard
            ranking = sector_analyst.get_sector_ranking(limit=11)
            sector_rotation = {"ranking": ranking}
        except Exception as e:
            logger.error(f"L2 Sector Rotation Error: {e}")
            sector_rotation = {}

    # 13. Level 4 ML Predictor Stats
    try:
        from ml_predictor import MLPredictor
        ml = MLPredictor()
        ml_stats = ml.get_dashboard_stats()
    except Exception as e:
        logger.error(f"ML Stats Fetch Error: {e}")
        ml_stats = {}

    # 14. Self-Learning Stats (L13)
    try:
        ticker_cache_stats = db.get_ticker_cache_stats()
        rebalancer_learning = db.get_rebalancer_learning_stats()
    except Exception as e:
        logger.error(f"Self-Learning Stats Fetch Error: {e}")
        ticker_cache_stats = {}
        rebalancer_learning = {}

    # 15. Run Observability (Step 4 dashboard view)
    try:
        observability = build_observability_dashboard(
            reports_dir=os.environ.get("RUN_REPORT_DIR", "run_logs/latest")
        )
    except Exception as e:
        logger.error(f"Observability Dashboard Error: {e}")
        observability = {"runs": [], "reports_dir": os.environ.get("RUN_REPORT_DIR", "run_logs/latest")}

    return render_template('dashboard.html', 
                           signals=signals, 
                           portfolio=portfolio, 
                           history=history,
                           total_value_eur=total_val, 
                           total_invested_eur=total_inv, 
                           total_pl_eur=total_pl, 
                           total_pl_percent=pl_pct,
                           chart_labels=dates,
                           chart_data=json.dumps(chart_d),
                           last_run=last_run,
                           last_run_iso=last_run_iso,
                           now_iso=datetime.utcnow().isoformat(),
                           audit_stats=audit_stats,
                           market_mood=market_mood,
                           advisor_analysis=advisor_analysis,
                           macro_stats=macro_stats,
                           whale_stats=whale_stats,
                           paper_portfolio=paper_portfolio_enriched,
                           paper_total_value=paper_total_value,
                           market_regime=market_regime,
                           sector_rotation=sector_rotation,
                           backtest_results=backtest_results,
                           benchmark_data=benchmark_data,
                           ml_stats=ml_stats,
                           ticker_cache_stats=ticker_cache_stats,
                           rebalancer_learning=rebalancer_learning,
                           observability=observability,
                           dashboard_fast_mode=fast_mode)


@app.route('/api/webhook', methods=['POST'])
def webhook():
    if request.method == "POST":
        json_update = request.get_json()
        if json_update:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # FRESH APP INIT
                bot_app = ApplicationBuilder().token(TOKEN).build()
                
                # Register Handlers
                bot_app.add_handler(CommandHandler("start", start))
                bot_app.add_handler(CommandHandler("help", help_command))
                bot_app.add_handler(CommandHandler("hunt", hunt_command))
                bot_app.add_handler(CommandHandler("harvest", harvest_command))
                bot_app.add_handler(CommandHandler("sectors", sectors_command))
                bot_app.add_handler(CommandHandler("portfolio", show_portfolio))
                bot_app.add_handler(CommandHandler("dashboard", dashboard_command))
                bot_app.add_handler(CommandHandler("delete", delete_command))
                bot_app.add_handler(CommandHandler("reset", reset_command))
                bot_app.add_handler(CommandHandler("setprice", setprice_command))
                bot_app.add_handler(CommandHandler("setticker", setticker_command))
                bot_app.add_handler(CommandHandler("setqty", setqty_command))
                bot_app.add_handler(CommandHandler("alert", alert_command))
                bot_app.add_handler(CommandHandler("settarget", settarget_command))
                bot_app.add_handler(CommandHandler("alerts", my_alerts_command))
                bot_app.add_handler(CommandHandler("paper", paper_command))
                bot_app.add_handler(CommandHandler("recall", recall_command))
                bot_app.add_handler(CommandHandler("learn", learn_command))
                bot_app.add_handler(CommandHandler("dbstatus", dbstatus_command))
                bot_app.add_handler(CommandHandler("benchmark", benchmark_command))
                bot_app.add_handler(CommandHandler("report", report_command))
                bot_app.add_handler(CommandHandler("backtest", backtest_command))
                bot_app.add_handler(CommandHandler("analyze", analyze_command))
                bot_app.add_handler(CommandHandler("sell", sell_command))
                bot_app.add_handler(CommandHandler("settings", settings_command))
                bot_app.add_handler(CommandHandler("mode", mode_command))
                bot_app.add_handler(CommandHandler("usage", usage_command))
                bot_app.add_handler(CommandHandler("macro", macro_command))
                bot_app.add_handler(CommandHandler("whale", whale_command))
                bot_app.add_handler(CommandHandler("rebalance", rebalance_command))
                bot_app.add_handler(CommandHandler("trainml", trainml_command))
                bot_app.add_handler(CommandHandler("strategy", strategy_command))
                bot_app.add_handler(CommandHandler("add", add_command))
                bot_app.add_handler(CommandHandler("protect", protect_command))
                bot_app.add_handler(CommandHandler("core", core_command))
                bot_app.add_handler(CommandHandler("portfolio_backtest", portfolio_backtest_command))
                bot_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
                # Text handler for quick sells and smart input
                bot_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))
                # PDF handler for Trade Republic documents
                bot_app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
                bot_app.add_handler(CallbackQueryHandler(handle_callback))
                
                update = Update.de_json(json_update, bot_app.bot)
                
                loop.run_until_complete(bot_app.initialize())
                loop.run_until_complete(bot_app.process_update(update))
                loop.run_until_complete(bot_app.shutdown())
            except Exception as e:
                logger.error(f"Error: {e}")
                return "Error", 500
    return "OK", 200

async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Sets a price alert.
    Usage: /alert BTC > 80000  OR  /alert NVDA < 100
    """
    try:
        args = context.args
        if len(args) < 3:
            await update.message.reply_text("❌ **Uso Corretto:** `/alert BTC > 80000` oppure `/alert BTC < 50000`", parse_mode='Markdown')
            return

        ticker = args[0].upper()
        operator = args[1]
        try:
            price = float(args[2].replace(',', '.'))
        except ValueError:
             await update.message.reply_text("❌ Il prezzo deve essere un numero valido.")
             return

        condition = ""
        if operator in [">", "sopra", "above"]:
            condition = "ABOVE"
        elif operator in ["<", "sotto", "below"]:
            condition = "BELOW"
        else:
            await update.message.reply_text("❌ Operatore non riconosciuto. Usa `>` o `<`.")
            return

        db = DBHandler()
        # Normalization check logic could go here, but DBHandler saves as is. 
        # Sentinel will map logic.
        
        success = db.add_alert(
            chat_id=update.effective_chat.id,
            ticker=ticker,
            condition=condition,
            price_threshold=price
        )

        if success:
            arrow = "↗️" if condition == "ABOVE" else "↘️"
            await update.message.reply_text(f"✅ **Alert Impostato!**\nTi avviserò se **{ticker}** va **{condition}** €{price} (EUR) {arrow}")
        else:
            await update.message.reply_text("❌ Errore nel salvataggio dell'alert.")

    except Exception as e:
        logger.error(f"Alert command error: {e}")
        await update.message.reply_text("❌ Errore interno.")


async def my_alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show active alerts."""
    try:
        db = DBHandler()
        alerts = db.get_user_alerts(chat_id=update.effective_chat.id)
        
        if not alerts:
            await update.message.reply_text("🔕 Non hai allarmi attivi.")
            return

        msg = "🔔 **I tuoi Allarmi Attivi:**\n\n"
        for a in alerts:
            cond = ">" if a['condition'] == "ABOVE" else "<"
            msg += f"🔹 **{a['ticker']}** {cond} €{a['price_threshold']}\n"
        
        msg += "\n(Gli allarmi si disattivano automaticamente una volta scattati)"
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"My alerts error: {e}")
        await update.message.reply_text("❌ Errore nel recupero degli allarmi.")

async def paper_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows Paper Trading stats."""
    await update.message.reply_text("🧪 **Laboratorio Zero-Cost**\nCalcolo performance simulata...")
    try:
        from paper_trader import PaperTrader
        from market_data import MarketData
        
        pt = PaperTrader()
        market = MarketData()
        
        # Single-user mode: read all paper positions (chat_id not used)
        portfolio = pt.get_portfolio()
        
        if not portfolio:
             await update.message.reply_text("🧪 Il tuo portafoglio simulato è vuoto.\nAttendi i prossimi segnali automatici!")
             return

        total_value = 0.0
        msg = "🧪 **Paper Portfolio Holdings:**\n\n"
        
        for p in portfolio:
            price, _ = market.get_smart_price_eur(p['ticker'])
            val = p['quantity'] * price
            total_value += val
            
            # PnL
            cost = p['quantity'] * p['avg_price']
            pnl = val - cost
            pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
            
            icon = "🟢" if pnl >= 0 else "🔴"
            msg += f"{icon} **{p['ticker']}**: {p['quantity']:.4f} @ €{p['avg_price']:.2f}\n"
            msg += f"   Valore: €{val:.2f} ({pnl_pct:+.1f}%)\n"
            
        msg += f"\n💰 **Valore Totale Simulato:** €{total_value:.2f}"
        
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Paper command error: {e}")
        await update.message.reply_text("❌ Errore Paper Trader.")

async def recall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Recall historical AI decisions for a ticker."""
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Uso: `/recall <TICKER>` (es. `/recall BTC`)", parse_mode="Markdown")
        return
    
    ticker = args[0].upper()
    await update.message.reply_text(f"🧠 **Recupero memoria storica per {ticker}...**", parse_mode="Markdown")
    
    try:
        from memory import Memory
        mem = Memory()
        
        memories = mem.recall_memory(ticker, limit=5)
        
        if not memories:
            await update.message.reply_text(f"📭 Nessuna decisione storica trovata per **{ticker}**.\n\n_L'AI inizierà a ricordare dopo il prossimo /hunt o /analyze._", parse_mode="Markdown")
            return
        
        msg = f"🧠 **Memoria Storica: {ticker}**\n\n"
        for m in memories:
            date = m.get('event_date', '')[:10]
            sentiment = m.get('sentiment', 'N/A')
            reasoning = m.get('reasoning', 'N/A')[:200]
            outcome = m.get('actual_outcome')
            
            emoji = "🟢" if sentiment in ["BUY", "ACCUMULATE"] else "🔴" if sentiment in ["SELL", "PANIC SELL"] else "⚪"
            msg += f"{emoji} **{date}**: {sentiment}\n"
            msg += f"_{reasoning}_\n"
            
            if outcome is not None:
                outcome_emoji = "✅" if outcome > 0 else "❌"
                msg += f"{outcome_emoji} Outcome: {outcome:+.1f}%\n"
            
            msg += "\n"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Recall command error: {e}")
        await update.message.reply_text(f"❌ Errore nel recupero memoria: {e}")

async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show lessons learned from trading mistakes."""
    await update.message.reply_text("🎓 **Recupero lezioni apprese dagli errori...**", parse_mode="Markdown")
    
    try:
        from memory import Memory
        mem = Memory()
        
        lessons = mem.get_lessons_learned(limit=5)
        
        if not lessons:
            await update.message.reply_text("📚 Nessuna lezione ancora registrata.\n\n_Le lezioni vengono generate quando i trade si chiudono con perdite significative._", parse_mode="Markdown")
            return
        
        msg = "🎓 **Lezioni Apprese (Errori Recenti):**\n\n"
        for l in lessons:
            ticker = l.get('ticker', 'N/A')
            date = l.get('event_date', '')[:10]
            outcome = l.get('actual_outcome', 0)
            lesson = l.get('lessons_learned', 'N/A')
            
            emoji = "❌" if outcome < 0 else "⚠️"
            msg += f"{emoji} **{ticker}** ({date}): {outcome:+.1f}%\n"
            msg += f"📝 _{lesson}_\n\n"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Learn command error: {e}")
        await update.message.reply_text(f"❌ Errore nel recupero lezioni: {e}")

async def dbstatus_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show database storage status and table statistics."""
    await update.message.reply_text("📊 **Controllo stato database...**", parse_mode="Markdown")
    
    try:
        from db_maintenance import DBMaintenance
        maint = DBMaintenance()
        
        health = maint.check_storage_health()
        stats = maint.get_table_stats()
        
        # Status emoji
        status_emoji = "✅" if health["status"] == "healthy" else "⚡" if health["status"] == "warning" else "⚠️"
        
        msg = f"{status_emoji} **Database Status**\n\n"
        msg += f"📦 **Storage:** {health['size_mb']:.1f}MB / {health['limit_mb']}MB\n"
        msg += f"📊 **Utilizzo:** {health['usage_percent']:.1f}%\n\n"
        
        # Table stats
        msg += "📋 **Tabelle (righe):**\n"
        for table, count in sorted(stats.items(), key=lambda x: -x[1]):
            if count > 0:
                msg += f"• `{table}`: {count:,}\n"
        
        # Cleanup policies
        msg += "\n♻️ **Policy Cleanup Automatico:**\n"
        msg += "• Logs: 7 giorni\n"
        msg += "• Memory: 90 giorni\n"
        msg += "• Backtest: 60 giorni\n"
        msg += "• Signals: 180 giorni\n"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"DB Status command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def benchmark_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Compare portfolio performance against S&P500, BTC, and other benchmarks."""
    await update.message.reply_text("📊 **Calcolo performance vs benchmarks...**", parse_mode="Markdown")
    
    try:
        from benchmark import Benchmark
        bench = Benchmark()
        
        # Default 30 days, or parse from args
        period_days = 30
        if context.args:
            try:
                period_days = int(context.args[0])
            except Exception as e:
                logger.debug(f"Benchmark period parse failed for '{context.args[0]}': {e}")
        
        report = await bench.format_benchmark_report_async(period_days)
        await update.message.reply_text(report, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Benchmark command error: {e}")
        await update.message.reply_text(f"❌ Errore nel calcolo benchmark: {e}")

@debounce_command
async def rebalance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger Rebalancing Analysis via GitHub Actions (Async)."""
    import httpx
    
    chat_id = update.effective_chat.id
    # user_id unused here but available
    
    await update.message.reply_text(
        "📊 **Analisi Ribilanciamento Avviata!** 🚀\n\n"
        "⏳ Sto affidando il calcolo a **DeepSeek R1** (potente ma lento).\n"
        "Riceverai il report completo qui tra circa **2-3 minuti**.\n\n"
        "_(Il calcolo gira su GitHub per evitare timeout)_"
    )
    
    # Get GitHub token from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "bruciato87/zerocosthunter")
    
    if not github_token:
        # Fallback: Try to run locally if no token (will likely timeout on Vercel)
        logger.warning("GITHUB_TOKEN not set - running rebalance locally")
        try:
            from rebalancer import Rebalancer
            rebalancer = Rebalancer()
            report = rebalancer.format_rebalance_report()
            await update.message.reply_text(report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Errore locale: {e}")
        return
    
    # Trigger GitHub Actions workflow
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/repos/{github_repo}/actions/workflows/market_scan.yml/dispatches",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                },
                json={
                    "ref": "main",
                    "inputs": {
                        "job_type": "rebalance",
                        "target_chat_id": str(chat_id)
                    }
                },
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"GitHub Action 'rebalance' triggered for {chat_id}")
            else:
                logger.error(f"GitHub Action trigger failed: {response.status_code} - {response.text}")
                await update.message.reply_text(f"⚠️ Errore avvio task remoto: {response.status_code}")
                
    except Exception as e:
        logger.error(f"GitHub Trigger Error: {e}")
        await update.message.reply_text("⚠️ Impossibile avviare il task remoto.")

async def trainml_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show status or train the Pure Python ML model."""
    await update.message.reply_text("🤖 **ML Predictor Status...**", parse_mode="Markdown")
    
    try:
        from ml_predictor import MLPredictor
        ml = MLPredictor()
        
        # Get stats
        stats = ml.get_dashboard_stats()
        is_ready = stats.get('is_ml_ready', False)
        version = stats.get('model_version', 'N/A')
        accuracy = stats.get('accuracy')
        training_count = stats.get('available_samples', 0)
        predictions = stats.get('recent_predictions', [])
        
        # Check if user wants to train
        if context.args and context.args[0].lower() == 'train':
            if training_count >= ml.MIN_TRAINING_SAMPLES:
                await update.message.reply_text(
                    "⏳ **Avvio Training Remoto ML (Cloud)...**\n\n"
                    "L'operazione girerà su GitHub per salvare risorse.\n"
                    "Riceverai una conferma al termine.", 
                    parse_mode="Markdown"
                )
                
                # Trigger GitHub Action
                chat_id = update.effective_chat.id
                github_token = os.environ.get("GITHUB_TOKEN")
                repo_owner = "bruciato87"
                repo_name = "zerocosthunter"
                workflow_id = "market_scan.yml"
                
                if not github_token:
                    logger.warning("GITHUB_TOKEN not set - running locally as fallback")
                    success = ml.train()
                    if success:
                        await update.message.reply_text("✅ Training Locale Completato (Fallback).")
                    else:
                        await update.message.reply_text("❌ Training Locale Fallito.")
                    return

                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches",
                            headers={
                                "Authorization": f"Bearer {github_token}",
                                "Accept": "application/vnd.github.v3+json"
                            },
                            json={
                                "ref": "main",
                                "inputs": {
                                    "job_type": "trainml",
                                    "target_chat_id": str(chat_id)
                                }
                            },
                            timeout=10
                        )
                        if response.status_code == 204:
                            logger.info(f"GitHub Action 'trainml' triggered for {chat_id}")
                        else:
                            await update.message.reply_text(f"⚠️ Errore Cloud: {response.status_code}")
                except Exception as e:
                    logger.error(f"GitHub Trigger Error: {e}")
                    await update.message.reply_text("⚠️ Errore connessione GitHub.")
                return
            else:
                remaining = ml.MIN_TRAINING_SAMPLES - training_count
                await update.message.reply_text(f"⏳ Servono altri **{remaining}** segnali chiusi.", parse_mode="Markdown")
                return
        
        # Show status
        acc = stats.get('accuracy', 0) or 0
        r2 = stats.get('reg_r2', 0) or 0
        mse = stats.get('lstm_mse', 0) or 0
        
        msg = f"🤖 **ML Predictor Status**\n\n"
        msg += f"📦 **Modello:** `{version}`\n"
        msg += f"🎯 **ML Attivo:** {'✅ Pure Python GB' if is_ready else '❌ Rule-based'}\n"
        
        msg += f"\n📊 **Metrics:**\n"
        msg += f"├ Accuracy: {acc:.1%}\n"
        msg += f"├ Regressor (R²): {r2:.1%}\n"
        msg += f"└ LSTM (MSE): {mse:.4f}\n"
        
        msg += f"\n📈 **Segnali Disponibili:** {training_count}/{ml.MIN_TRAINING_SAMPLES}\n"
        
        if training_count >= ml.MIN_TRAINING_SAMPLES:
            if is_ready:
                msg += "✅ Modello addestrato e attivo!\n"
            else:
                msg += "💡 Usa `/trainml train` per addestrare il modello.\n"
        else:
            remaining = ml.MIN_TRAINING_SAMPLES - training_count
            msg += f"⏳ Servono altri {remaining} segnali per il training.\n"
        
        if predictions:
            msg += f"\n📊 **Ultime Predizioni:**\n"
            for pred in predictions[:3]:
                ticker = pred.get('ticker', 'N/A')
                direction = pred.get('predicted_direction', 'N/A')
                conf = pred.get('ml_confidence', 0)
                emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"
                msg += f"  {emoji} {ticker}: {direction} ({conf:.0%})\n"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"TrainML command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def strategy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    View or set strategy rules for assets.
    Usage:
        /strategy - List all rules
        /strategy set TICKER type=SWING target=10% tp=20% sl=-15%
    """
    try:
        from strategy_manager import StrategyManager
        sm = StrategyManager()
        
        args = context.args
        
        # No args: Show all rules
        if not args:
            report = sm.format_rules_report()
            await update.message.reply_text(report)
            return
        
        # /strategy set TICKER ...
        if args[0].lower() == 'set':
            if len(args) < 2:
                await update.message.reply_text(
                    "❌ **Formato:**\n"
                    "`/strategy set TICKER type=SWING target=10 cap=20 tp=25 sl=-15`\n\n"
                    "**Parametri:**\n"
                    "• `type`: ACCUMULATE, SWING, LONG_TERM\n"
                    "• `target`: % allocazione target\n"
                    "• `cap`: % max allocazione\n"
                    "• `tp`: % take profit (opzionale)\n"
                    "• `sl`: % stop loss (opzionale)\n"
                    "• `minprofit`: € minimo netto per vendere",
                    parse_mode="Markdown"
                )
                return
            
            ticker = args[1].upper()
            
            # Parse key=value pairs
            params = {
                'strategy_type': 'ACCUMULATE',
                'target_pct': 10.0,
                'max_cap': 20.0,
                'take_profit': None,
                'stop_loss': None,
                'min_profit': 50.0
            }
            
            for arg in args[2:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.lower().strip()
                    value = value.strip().replace('%', '')
                    
                    if key == 'type':
                        params['strategy_type'] = value.upper()
                    elif key == 'target':
                        params['target_pct'] = float(value)
                    elif key == 'cap':
                        params['max_cap'] = float(value)
                    elif key == 'tp':
                        params['take_profit'] = float(value)
                    elif key == 'sl':
                        params['stop_loss'] = float(value)
                    elif key == 'minprofit':
                        params['min_profit'] = float(value)
            
            # Set the rule
            success = sm.set_rule(
                ticker=ticker,
                strategy_type=params['strategy_type'],
                target_pct=params['target_pct'],
                max_cap=params['max_cap'],
                take_profit=params['take_profit'],
                stop_loss=params['stop_loss'],
                min_profit=params['min_profit']
            )
            
            if success:
                emoji = "🔵" if params['strategy_type'] == 'LONG_TERM' else "🟢" if params['strategy_type'] == 'ACCUMULATE' else "🟡"
                msg = (
                    f"✅ **Regola Salvata per {ticker}**\n\n"
                    f"{emoji} Tipo: `{params['strategy_type']}`\n"
                    f"🎯 Target: {params['target_pct']}%\n"
                    f"🚫 Cap Max: {params['max_cap']}%\n"
                )
                if params['take_profit']:
                    msg += f"💰 Take Profit: +{params['take_profit']}%\n"
                if params['stop_loss']:
                    msg += f"⚠️ Stop Loss: {params['stop_loss']}%\n"
                msg += f"📊 Min Profit: €{params['min_profit']}"
                
                await update.message.reply_text(msg, parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ Errore nel salvataggio della regola.")
            return
        
        # Unknown subcommand
        await update.message.reply_text(
            "❌ Comando non riconosciuto.\n\n"
            "**Uso:**\n"
            "`/strategy` - Mostra regole\n"
            "`/strategy set TICKER type=SWING ...` - Imposta regola",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Strategy command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def portfolio_backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Run portfolio backtest to calculate Sharpe Ratio, Max Drawdown, etc.
    TRIGGERS GITHUB ACTION (Async) to save Vercel CPU.
    Usage: /portfolio_backtest [days]
    """
    try:
        import httpx
        chat_id = update.effective_chat.id
        
        # Get period from args (default 180 days)
        period = 180
        if context.args and context.args[0].isdigit():
            period = int(context.args[0])
            period = min(365, max(30, period))  # Limit 30-365 days
            
        await update.message.reply_text(
            f"⏳ **Avvio Backtest Remoto ({period} giorni)...**\n"
            "☁️ Il calcolo avverrà sul Cloud (GitHub) per non sovraccaricare il bot.\n"
            "📨 Riceverai il report qui tra 1-2 minuti."
        )
        
        # Trigger GitHub Action
        # Requires GITHUB_TOKEN in env vars or hardcoded if necessary (here utilizing existing env approach)
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_owner = "bruciato87"
        repo_name = "zerocosthunter"
        workflow_id = "market_scan.yml"
        
        if not github_token:
            # Fallback for Vercel env if GITHUB_TOKEN not set
            # Assuming it might be set as env var, or log warning
            logger.warning("GITHUB_TOKEN not set, cannot trigger remote backtest.")
            await update.message.reply_text("⚠️ Token GitHub mancante. Contatta l'admin.")
            return

        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches",
                headers=headers,
                json={
                    "ref": "main",
                    "inputs": {
                        "job_type": "portfolio_backtest",
                        "target_chat_id": str(chat_id),
                        "backtest_period": str(period)
                    }
                },
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"GitHub Action 'portfolio_backtest' triggered for {chat_id}")
            else:
                logger.error(f"GitHub Action trigger failed: {response.status_code} - {response.text}")
                await update.message.reply_text(f"⚠️ Errore avvio task remoto: {response.status_code}")

    except Exception as e:
        logger.error(f"Portfolio backtest init error: {e}")
        await update.message.reply_text(f"❌ Errore init: {e}")

async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Track a sell transaction.
    Usage: /sell TICKER QUANTITY PRICE
    Example: /sell RENDER 100 2.50
    """
    try:
        args = context.args
        if len(args) < 3:
            await update.message.reply_text(
                "❌ **Formato corretto:**\n"
                "`/sell TICKER QTY PREZZO`\n\n"
                "**Esempio:**\n"
                "`/sell RENDER 100 2.50`\n"
                "(Venduto 100 RENDER a €2.50 ciascuno)",
                parse_mode="Markdown"
            )
            return
        
        ticker = args[0].upper()
        try:
            quantity = float(args[1])
            price = float(args[2])
        except ValueError:
            await update.message.reply_text("❌ Quantità e prezzo devono essere numeri validi.")
            return
        
        if quantity <= 0 or price <= 0:
            await update.message.reply_text("❌ Quantità e prezzo devono essere positivi.")
            return
        
        # Fetch portfolio to validate and calculate P&L
        from db_handler import DBHandler
        db = DBHandler()
        portfolio = db.get_portfolio()
        
        # Find the asset in portfolio
        asset = next((p for p in portfolio if p['ticker'].upper() == ticker or 
                      p['ticker'].upper().replace('-USD', '') == ticker or
                      ticker + '-USD' == p['ticker'].upper()), None)
        
        if not asset:
            await update.message.reply_text(
                f"⚠️ **{ticker}** non trovato nel portfolio.\n"
                f"Registro comunque la transazione.",
                parse_mode="Markdown"
            )
            avg_price = price  # Use sell price as reference if not in portfolio
        else:
            avg_price = asset.get('avg_price', price)
            
        total_value = quantity * price
        
        # Store pending sell data for confirmation handler
        context.user_data['pending_sell'] = {
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'avg_price': avg_price
        }
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [
            [InlineKeyboardButton("✅ Conferma Manuale", callback_data="confirm_sell_manual")],
            [InlineKeyboardButton("❌ Annulla", callback_data="cancel_sell")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg = (
            f"📊 **Vendita Iniziata: {ticker}**\n"
            f"├ Quantità: {quantity}\n"
            f"├ Prezzo: €{price:.4f}\n"
            f"└ Totale Stimato: €{total_value:.2f}\n\n"
            f"📸 **Invia ORA lo screenshot di Trade Republic** per calcolare Tasse e Commissioni esatte.\n"
            f"_(Oppure clicca Conferma per procedere con stime standard)_"
        )
        
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_markup)
            
    except Exception as e:
        logger.error(f"Sell command error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")

async def handle_sell_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle sale screenshot from Trade Republic.
    Usage: Send a photo with caption "/sell TICKER" or just "/sell"
    The system will extract: quantity, price, imposta, commissione, netto from OCR.
    """
    try:
        # Check if message has photo
        if not update.message.photo:
            return  # Not a photo message
        
        caption = update.message.caption or ""
        if not caption.lower().startswith("/sell"):
            return  # Not a sell command
        
        # Extract ticker from caption (e.g., "/sell RENDER")
        parts = caption.split()
        ticker = parts[1].upper() if len(parts) > 1 else None
        
        await update.message.reply_text("📸 **Analizzo lo screenshot di vendita...**", parse_mode="Markdown")
        
        # Download photo
        user_id = update.effective_user.id
        photo = update.message.photo[-1]
        file_obj = await photo.get_file()
        file_path = f"/tmp/sell_{user_id}.jpg"
        await file_obj.download_to_drive(file_path)
        
        # OCR extraction
        from brain import Brain
        brain = Brain()
        sale_data = brain.parse_sale_from_image(file_path)
        
        if "error" in sale_data:
            await update.message.reply_text(f"❌ Errore OCR: {sale_data['error']}")
            return
        
        # Use extracted data or override with ticker from caption
        asset_name = sale_data.get('asset_name', 'UNKNOWN')
        if ticker:
            asset_name = ticker
        
        quantity = sale_data.get('quantity', 0)
        price = sale_data.get('price_per_unit', 0)
        tax = sale_data.get('tax_amount', 0) or 0
        commission = sale_data.get('commission', 1.0) or 1.0
        net_received = sale_data.get('net_received', 0)
        profit = sale_data.get('profit_amount', 0) or 0
        profit_pct = sale_data.get('profit_percent', 0) or 0
        gross_total = sale_data.get('gross_total', quantity * price)
        
        # Store pending sale for confirmation
        context.user_data['pending_sell'] = {
            'ticker': asset_name,
            'quantity': quantity,
            'price': price,
            'tax': tax,
            'commission': commission,
            'net_received': net_received,
            'profit': profit,
            'profit_pct': profit_pct,
            'gross_total': gross_total
        }
        
        # Show confirmation
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        pnl_emoji = "🟢" if profit >= 0 else "🔴"
        
        confirm_msg = (
            f"📋 **Conferma Vendita (da Screenshot)**\n\n"
            f"📊 **{asset_name}**\n"
            f"├ Quantità: {quantity}\n"
            f"├ Prezzo: €{price:.2f}\n"
            f"├ Lordo: €{gross_total:.2f}\n"
            f"├ Imposta (TR): -€{tax:.2f}\n"
            f"├ Commissione: -€{commission:.2f}\n"
            f"├ **Netto Ricevuto: €{net_received:.2f}**\n\n"
            f"{pnl_emoji} P&L Netto: +€{profit:.2f} (+{profit_pct:.1f}%)\n\n"
            f"⚠️ **Confermi questa vendita?**"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Conferma", callback_data="confirm_sell"),
                InlineKeyboardButton("❌ Annulla", callback_data="cancel_sell")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(confirm_msg, parse_mode="Markdown", reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Sell photo handler error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a comprehensive weekly report with benchmarks, signals, and top movers."""
    await update.message.reply_text("📑 **Generazione Report Settimanale...**", parse_mode="Markdown")
    
    try:
        from benchmark import Benchmark
        from db_handler import DBHandler
        
        bench = Benchmark()
        db = DBHandler()
        
        # Get benchmark comparison (7 days for weekly) in parallel
        comparison_task = asyncio.to_thread(bench.compare_vs_benchmarks, 7)
        movers_task = asyncio.to_thread(bench.get_top_movers, 5)
        
        comparison, movers = await asyncio.gather(comparison_task, movers_task)
        
        # Get recent signals (use correct column names)
        signals = db.supabase.table("signal_tracking") \
            .select("ticker, status, pnl_percent, target_price, created_at") \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute().data or []
        
        # Build report
        report = "📑 **WEEKLY REPORT**\n"
        report += "━" * 20 + "\n\n"
        
        # Portfolio Summary
        if "portfolio" in comparison:
            p = comparison["portfolio"]
            port_emoji = "🟢" if p.get("return_pct", 0) >= 0 else "🔴"
            report += f"{port_emoji} **Portfolio:** {p.get('return_pct', 0):+.2f}%\n"
            report += f"💰 Valore: €{p.get('current_value', 0):,.2f}\n\n"
        
        # Benchmark Comparison
        report += "📈 **vs Benchmarks (7d):**\n"
        for name, data in comparison.get("benchmarks", {}).items():
            ret = data.get("return_pct", 0)
            emoji = "🟢" if ret >= 0 else "🔴"
            report += f"  {emoji} {name}: {ret:+.2f}%\n"
        
        beating = comparison.get("beating", [])
        if beating:
            report += f"\n🏆 **Batti:** {', '.join(beating)}\n"
        
        # Top Movers
        report += "\n🚀 **Top Gainers:**\n"
        for g in movers.get("gainers", [])[:3]:
            report += f"  • {g['ticker']}: {g['pnl_pct']:+.1f}%\n"
        
        report += "\n📉 **Top Losers:**\n"
        for l in movers.get("losers", [])[:3]:
            report += f"  • {l['ticker']}: {l['pnl_pct']:+.1f}%\n"
        
        # Recent Signals
        if signals:
            report += "\n📡 **Signal Tracking:**\n"
            for s in signals[:5]:
                status = s.get("status", "ACTIVE")
                ticker = s.get("ticker", "?")
                pnl = s.get("pnl_percent", 0) or 0
                emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                report += f"  {emoji} {ticker}: {status} ({pnl:+.1f}%)\n"
        
        report += "\n" + "━" * 20
        report += f"\n_Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        
        await update.message.reply_text(report, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Report command error: {e}")
        await update.message.reply_text(f"❌ Errore generazione report: {e}")

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Runs the best historical strategy backtest for a given ticker."""
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Uso: `/backtest <TICKER>` (es. `/backtest BTC-USD`)", parse_mode="Markdown")
        return
    
    ticker = args[0].upper()
    
    # Add -USD suffix for crypto if missing
    if ticker in ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", "AVAX", "LINK"]:
        ticker = f"{ticker}-USD"
    
    await update.message.reply_text(f"📈 **Esecuzione Backtest su {ticker}...**\n_(Usando la miglior strategia storica)_", parse_mode="Markdown")
    
    try:
        from backtester import Backtester
        bt = Backtester()
        
        result = bt.run_best_strategy(ticker, 90)
        
        if result:
            emoji = "🟢" if result['pnl_percent'] >= 0 else "🔴"
            msg = (
                f"📊 **Backtest Completato: {ticker}**\n\n"
                f"📅 Periodo: {result['period_days']} giorni\n"
                f"🎯 Strategia: `{result['strategy_version']}`\n\n"
                f"{emoji} **P/L: {result['pnl_percent']:+.2f}%**\n"
                f"📈 Trades: {result['total_trades']}\n"
                f"🏆 Win Rate: {result['win_rate']:.0f}%\n\n"
                f"💰 Bilancio Iniziale: €{result['starting_balance']:,.0f}\n"
                f"💵 Bilancio Finale: €{result['ending_balance']:,.0f}\n\n"
                f"_Risultato salvato in Dashboard → Laboratory_"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text(f"❌ Nessun dato trovato per **{ticker}**. Verifica il ticker.", parse_mode="Markdown")
    
    except Exception as e:
        logger.error(f"Backtest command error: {e}")
        await update.message.reply_text(f"❌ Errore durante il backtest: {e}")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger Deep Dive Analysis via GitHub Actions (Async)."""
    import httpx
    
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Uso: `/analyze <TICKER>` (es. `/analyze NVDA`)")
        return
    
    ticker = args[0].upper()
    chat_id = update.effective_chat.id
    
    await update.message.reply_text(
        f"🔬 **Analisi Deep Dive Avviata: {ticker}**\n\n"
        "⏳ Sto interrogando **DeepSeek R1** (ragionamento profondo).\n"
        "Riceverai il report completo qui tra circa **2-3 minuti**.\n\n"
        "_(Analisi remota su GitHub)_"
    )
    
    # Get GitHub token from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "bruciato87/zerocosthunter")
    
    if not github_token:
        # Fallback: Try to run locally if no token (might timeout)
        logger.warning("GITHUB_TOKEN not set - running analyze locally")
        try:
            # Replicate local logic briefly or just warn
            await update.message.reply_text("⚠️ GITHUB_TOKEN mancante. Esecuzione locale non supportata per R1.")
        except Exception as e:
            await update.message.reply_text(f"❌ Errore locale: {e}")
        return
    
    # Trigger GitHub Actions workflow
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/repos/{github_repo}/actions/workflows/market_scan.yml/dispatches",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                },
                json={
                    "ref": "main",
                    "inputs": {
                        "job_type": "analyze",
                        "target_chat_id": str(chat_id),
                        "target_ticker": ticker
                    }
                },
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"GitHub Action 'analyze' triggered for {ticker} by {chat_id}")
            else:
                logger.error(f"GitHub Action trigger failed: {response.status_code} - {response.text}")
                await update.message.reply_text(f"⚠️ Errore avvio task remoto: {response.status_code}")
                
    except Exception as e:
        logger.error(f"GitHub Trigger Error: {e}")
        await update.message.reply_text("⚠️ Impossibile avviare il task remoto.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
