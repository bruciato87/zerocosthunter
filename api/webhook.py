import os
import logging
import asyncio
import json
from flask import Flask, request, render_template, redirect, session, url_for, flash
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import yfinance as yf
import pandas as pd
from datetime import datetime
import sys

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

async def hunt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger hunt via GitHub Actions to bypass Vercel timeout limits."""
    import httpx
    
    chat_id = update.effective_chat.id
    
    # Get GitHub token from environment
    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO", "bruciato87/zerocosthunter")
    
    if not github_token:
        # Fallback: Try to run locally (will likely timeout)
        logger.warning("GITHUB_TOKEN not set - falling back to inline execution (may timeout)")
        await update.message.reply_text("🏹 **Caccia Iniziata!**\n⚠️ Esecuzione locale (potrebbe essere lenta)...")
        try:
            await run_async_pipeline()
            await update.message.reply_text("✅ **Caccia Completata.**")
        except Exception as e:
            await update.message.reply_text(f"❌ **Errore:** {str(e)[:200]}")
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
        msg = (
            "⚙️ **Smart Filters Config**\n\n"
            f"🎯 **Min Confidence:** {int(min_conf * 100)}%\n"
            f"💼 **Portfolio Mode:** {'✅ ON' if only_port else '❌ OFF'}\n\n"
            "**Comandi per modificare:**\n"
            "`/settings confidence=80` (Imposta al 80%)\n"
            "`/settings portfolio=on` (Solo asset posseduti)\n"
            "`/settings portfolio=off` (Tutti i segnali)"
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
        except Exception as e:
            logger.error(f"Error parsing setting {arg}: {e}")

    if updated:
        await update.message.reply_text("✅ Impostazioni aggiornate!")
    else:
        await update.message.reply_text("❌ Errore/Nessuna modifica. Usa il formato: `/settings confidence=80`")

async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch between PREPROD (Gemini only) and PROD (DeepSeek + Gemini) modes."""
    logger.info(f"MODE COMMAND CALLED. Args: {context.args}")
    db = DBHandler()
    args = context.args
    
    # Get current mode from settings
    settings = db.get_settings()
    current_mode = settings.get("app_mode", "PREPROD")
    
    if not args:
        # Show current mode
        mode_emoji = "🧪" if current_mode == "PREPROD" else "🚀"
        msg = (
            f"🔧 **Modalità Operativa**\n\n"
            f"Attuale: {mode_emoji} **{current_mode}**\n\n"
            f"**PREPROD** 🧪: Solo Gemini (gratis, 1000+ req/giorno)\n"
            f"**PROD** 🚀: DeepSeek primario + Gemini fallback\n\n"
            f"**Per cambiare:**\n"
            f"`/mode PREPROD` oppure `/mode PROD`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return
    
    new_mode = args[0].upper()
    if new_mode not in ["PREPROD", "PROD"]:
        await update.message.reply_text("❌ Modalità non valida. Usa: `/mode PREPROD` o `/mode PROD`", parse_mode="Markdown")
        return
    
    # Update mode in database
    db.update_settings(app_mode=new_mode)
    
    mode_emoji = "🧪" if new_mode == "PREPROD" else "🚀"
    await update.message.reply_text(
        f"✅ Modalità cambiata a {mode_emoji} **{new_mode}**\n\n"
        f"{'Ora usi solo Gemini (gratis).' if new_mode == 'PREPROD' else 'Ora usi DeepSeek + Gemini fallback.'}",
        parse_mode="Markdown"
    )

from whale_watcher import WhaleWatcher
from economist import Economist

# ... (Previous imports)

async def setup_bot_commands(bot):
    """Configures the menu button in Telegram UI."""
    commands = [
        BotCommand("hunt", "🏹 Caccia Manuale (Analisi News)"),
        BotCommand("analyze", "🔬 Deep Dive Ticker (es. /analyze NVDA)"),
        BotCommand("backtest", "📈 Backtest Storico (es. /backtest BTC)"),
        BotCommand("portfolio", "📊 Vedi Portafoglio & Valore Live"),
        BotCommand("dashboard", "🖥️ Web Dashboard"),
        BotCommand("macro", "🏛 Macro Context (FED/VIX)"),
        BotCommand("whale", "🐋 Whale Alert (On-Chain)"),
        BotCommand("alert", "🔔 Imposta Alert Prezzo"),
        BotCommand("alerts", "🔕 I tuoi Alert"),
        BotCommand("paper", "🧪 Lab / Paper Trading"),
        BotCommand("recall", "🧠 Memoria Storica (es. /recall BTC)"),
        BotCommand("learn", "🎓 Lezioni dagli Errori"),
        BotCommand("benchmark", "📊 Portfolio vs S&P500/BTC"),
        BotCommand("report", "📑 Weekly Report Completo"),
        BotCommand("rebalance", "⚖️ Analisi Ribilanciamento"),
        BotCommand("sell", "💸 Registra Vendita"),
        BotCommand("mode", "🔧 PREPROD/PROD Mode"),
        BotCommand("dbstatus", "📦 Stato Storage DB"),
        BotCommand("help", "❓ Lista Comandi"),
        BotCommand("setprice", "💶 Correggi Prezzo"),
        BotCommand("setticker", "🏷 Correggi Ticker"),
        BotCommand("setqty", "🔢 Imposta Quantità"),
        BotCommand("delete", "🗑 Elimina un Asset"),
        BotCommand("settings", "⚙️ Configura Smart Filters"),
        BotCommand("reset", "☢️ Reset Totale"),
        BotCommand("start", "🚀 Avvia"),
    ]
    await bot.set_my_commands(commands)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await setup_bot_commands(context.bot)
    msg = (
        "🛠 **Lista Comandi Disponibili:**\n\n"
        "🏹 `/hunt`\nAvvia la caccia ai segnali di trading (analisi news + AI).\n\n"
        "📊 `/portfolio`\nVisualizza il valore attuale del tuo portafoglio in tempo reale.\n\n"
        "📱 `/dashboard`\nApri la dashboard web interattiva.\n\n"
        "🔬 `/analyze <TICKER>`\nAnalisi AI approfondita con news, technicals, e backtest storici.\n\n"
        "📈 `/backtest <TICKER>`\nEsegue un backtest storico con la miglior strategia per l'asset.\n\n"
        "⚖️ `/rebalance`\nAnalisi ribilanciamento portafoglio con suggerimenti AI.\n\n"
        "💸 `/sell TICKER QTY PREZZO`\nRegistra una vendita (es. /sell RENDER 100 2.50).\n\n"
        "🧠 **Memory (Neuro-Link):**\n"
        "• `/recall <TICKER>`: Perché abbiamo comprato/venduto questo asset?\n"
        "• `/learn`: Lezioni apprese dagli errori recenti.\n\n"
        "🏛 `/macro`\nVisualizza il contesto Macro Economico (VIX, Tassi, FED).\n\n"
        "🐋 `/whale`\nVisualizza movimenti On-Chain (Balene).\n\n"
        "🔔 **Allarmi Prezzo:**\n"
        "• `/alert BTC > 100000`: Avvisa se BTC supera 100k.\n"
        "• `/alert NVDA < 90`: Avvisa se NVDA scende sotto 90.\n"
        "• `/alerts`: Lista dei tuoi allarmi attivi.\n\n"
        "🧪 **Laboratory (Paper Trading):**\n"
        "• `/paper`: Vedi le performance del tuo portafoglio simulato.\n\n"
        "✍️ **Correzioni Manuali:**\n"
        "• `/setprice <TICKER> <PREZZO>`: Imposta manualmente il prezzo medio.\n"
        "• `/setqty <TICKER> <QTY>`: Imposta manualmente la quantità.\n"
        "• `/setticker <OLD> <NEW>`: Rinomina un ticker errato.\n\n"
        "🗑 **Gestione:**\n"
        "• `/delete <TICKER>`: Elimina un singolo asset.\n"
        "• `/reset`: Cancella TUTTO il portafoglio.\n"
        "• `/dbstatus`: Stato storage database (500MB limit).\n\n"
        "📊 **Reports:**\n"
        "• `/benchmark`: Confronta portfolio vs S&P500, BTC.\n"
        "• `/report`: Report completo con top movers e signals.\n\n"
        "🤖 **Machine Learning:**\n"
        "• `/trainml`: Stato del modello ML (predizioni, accuracy).\n"
        "• `/trainml train`: Addestra il modello sui tuoi dati storici.\n\n"
        "⚙️ `/settings`: Configura filtri AI.\n"
        "🔧 `/mode`: Cambia modalità PREPROD/PROD.\n\n"
        "📸 **Caricamento:**\nBasta inviare una foto! Se vuoi forzare il ticker, scrivilo nella **didascalia**."
    )
    await update.message.reply_text(msg)

async def macro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏛 Analizzo lo scenario Macro Economico... (VIX, Yields, FED)")
    try:
        eco = Economist()
        summary = eco.get_macro_summary()
        await update.message.reply_text(f"```{summary}```", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Macro Command Fail: {e}")
        await update.message.reply_text("❌ Errore nel recupero dati Macro.")

async def whale_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🐋 Localizzo le Balene (Binance Real-Time)...")
    try:
        ww = WhaleWatcher()
        summary = ww.analyze_flow()
        # Clean up
        await update.message.reply_text(f"```{summary}```", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Whale Command Fail: {e}")
        await update.message.reply_text("❌ Errore Whale Watcher.")

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

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    db = DBHandler()
    chat_id = update.effective_chat.id
    
    # --- SELL CALLBACKS ---
    if query.data == "confirm_sell_manual":
        try:
            # Parse message to identify asset and quantity
            # Format: "📊 Vendita Iniziata: TICKER\n├ Quantità: 100..."
            lines = query.message.text.split('\n')
            ticker_line = next((l for l in lines if "Vendita Iniziata:" in l), None)
            qty_line = next((l for l in lines if "Quantità:" in l), None)
            
            if not ticker_line or not qty_line:
                await query.edit_message_text("❌ Errore parsing dati messaggio.")
                return
                
            ticker = ticker_line.split(":")[1].strip()
            quantity = float(qty_line.split(":")[1].strip())
            
            # Update Portfolio
            portfolio = db.get_portfolio(chat_id)
            asset = next((p for p in portfolio if p['ticker'] == ticker), None)
            
            if asset:
                current_qty = asset['quantity']
                new_qty = current_qty - quantity
                
                if new_qty > 0:
                    if db.update_asset_quantity(chat_id, ticker, new_qty):
                        await query.edit_message_text(f"✅ **Vendita Confermata!**\n\n📉 {ticker}: Quantità aggiornata a {new_qty:.4f}.")
                    else:
                        await query.edit_message_text("❌ Errore aggiornamento DB.")
                else:
                    # Fully sold
                    if db.delete_asset(chat_id, ticker):
                        await query.edit_message_text(f"✅ **Vendita Totale Confermata!**\n\n🗑️ {ticker} rimosso dal portafoglio.")
                    else:
                        await query.edit_message_text("❌ Errore eliminazione asset.")
            else:
                 await query.edit_message_text(f"⚠️ {ticker} non trovato in portafoglio (già venduto?).")
                 
        except Exception as e:
            logger.error(f"Sell Confirm Error: {e}")
            await query.edit_message_text(f"❌ Errore: {e}")

    elif query.data == "cancel_sell":
        await query.edit_message_text("❌ Operazione di vendita annullata.")

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
    
    # Instantiate MarketData for Centralized Pricing
    market = MarketData()
    
    msg = "📊 **Il tuo Portafoglio:**\n\n"
    total_val = 0.0
    eur_usd = 1.1
    try:
        t = yf.Ticker("EURUSD=X")
        eur_usd = t.history(period="1d")['Close'].iloc[-1]
    except: pass

    TICKER_FIX_MAP = {
        "RNDR-USD": "RENDER-USD", 
        "RENDER": "RENDER-USD", # Fix for naked ticker
        "3DJ.DE": "3CP.F", 
        "BYD": "BY6.F", 
        "ICGA.FRA": "IAG.MC", 
        "ICGA.DE": "IAG.MC", 
        "ICGA.F": "IAG.MC", 
        "3CP": "3CP.F",
        "TCT": "NNnD.F",  # Tencent Frankfurt
        "3XC": "3CP.F",   # Xiaomi Frankfurt (user variant)
        "NUKL": "NUKL.DE" # Global X Uranium
    }



    # Grouping Logic
    grouped_assets = {"Crypto": [], "Stock": [], "ETF": [], "Other": []}
    
    for item in portfolio:
        ticker = item.get('ticker', 'N/A')
        search = TICKER_FIX_MAP.get(ticker, ticker)
        qty = item.get('quantity', 0)
        curr_val = 0.0
        
        if search and search != "UNKNOWN":
            try:
                 # Use Centralized MarketData Logic
                 found_price, used_ticker = market.get_smart_price_eur(search)
                 if found_price > 0:
                      curr_val = qty * found_price
                      # Update ticker display distinctively if mapped
                      if used_ticker and used_ticker != search:
                          ticker = f"{ticker}" # Keep original cleaner, maybe put suffix in details if debug needed
            except Exception as e:
                 logger.error(f"Price error for {search}: {e}")
        
        total_val += curr_val
        
        # Determine Group
        a_type = item.get('asset_type', 'Unknown')
        if a_type not in grouped_assets:
            grouped_assets["Other"].append({**item, 'current_value': curr_val, 'display_ticker': ticker})
        else:
            grouped_assets[a_type].append({**item, 'current_value': curr_val, 'display_ticker': ticker})

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
            val_str = f"€{a['current_value']:,.2f}" if a['current_value'] > 0 else "N/A"
            icon = "🪙" if cat == "Crypto" else "📈" if cat == "Stock" else "📊"
            
            # Cleaner, less indented format with spacing
            msg += f"\n{icon} **{a.get('asset_name') or a['display_ticker']}**\n"
            msg += f"   `{a['display_ticker']}`  •  `{val_str}`  •  {a['quantity']} pz\n"

    msg += f"\n━━━━━━━━━━━━━━━━━━━━\n💰 **TOTALE PORTAFOGLIO:** `€{total_val:,.2f}`"
    await update.message.reply_text(msg)

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
    except:
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
    db = DBHandler()
    market = MarketData()
    
    
    # 1. Signals
    try:
        signals = db.supabase.table("predictions").select("*").order("created_at", desc=True).limit(25).execute().data
    except: signals = []

    # 2. Portfolio Valuation with Forward-Fill Chart
    portfolio = db.get_portfolio()
    total_val, total_inv = 0.0, 0.0
    asset_trends = {}  # Track each asset's trend separately for forward-fill
    
    # FX & Tickers
    eur_usd = 1.1
    try:
        hist = yf.Ticker("EURUSD=X").history(period="1mo")
        if not hist.empty: eur_usd = hist['Close'].iloc[-1]
    except: pass
    
    TICKER_FIX = {
        "RNDR-USD": "RENDER-USD", 
        "RENDER": "RENDER-USD", 
        "3DJ.DE": "3CP.F", 
        "BYD": "BY6.F", 
        "ICGA.FRA": "IAG.MC", 
        "ICGA.F": "IAG.MC", 
        "ICGA.DE": "IAG.MC", 
        "3CP": "3CP.F",
        "TCT": "NNnD.F",  # Tencent Frankfurt
        "3XC": "3CP.F",   # Xiaomi Frankfurt
        "NUKL": "NUKL.DE" # Global X Uranium
    }

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
                    except: pass

            total_val += curr
            item['live_value_eur'] = round(curr, 2)
            item['pnl_eur'] = round(curr - cost, 2)
            item['pnl_percent'] = round(((curr - cost)/cost)*100, 2) if cost > 0 else 0
        except: pass

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
    except: 
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
    except: pass

    # 4. Audit Stats
    audit_stats = db.get_audit_stats()

    # 5. Market Mood (Insider)
    from insider import Insider
    insider = Insider()
    market_mood = insider.get_market_mood()

    # 6. Advisor (Risk Manager)
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
            except: pass
    except Exception as e:
        logger.error(f"History Fetch Error: {e}")
        history = []

    # 9. Paper Trading (Lab)
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
    try:
        from benchmark import Benchmark
        bench = Benchmark()
        benchmark_data = bench.compare_vs_benchmarks(30)
    except Exception as e:
        logger.error(f"Benchmark Fetch Error: {e}")
        benchmark_data = {}

    # 12. Level 2 Predictive Data (Phase 4)
    try:
        from market_regime import MarketRegimeClassifier
        from sector_rotation import SectorRotationTracker
        
        regime_classifier = MarketRegimeClassifier()
        sector_tracker = SectorRotationTracker()
        
        market_regime = regime_classifier.classify()
        sector_rotation = sector_tracker.analyze()
    except Exception as e:
        logger.error(f"L2 Data Fetch Error: {e}")
        market_regime = {}
        sector_rotation = {}

    # 13. Level 4 ML Predictor Stats
    try:
        from ml_predictor import MLPredictor
        ml = MLPredictor()
        ml_stats = ml.get_dashboard_stats()
    except Exception as e:
        logger.error(f"ML Stats Fetch Error: {e}")
        ml_stats = {}

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
                           ml_stats=ml_stats)


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
                bot_app.add_handler(CommandHandler("portfolio", show_portfolio))
                bot_app.add_handler(CommandHandler("dashboard", dashboard_command))
                bot_app.add_handler(CommandHandler("delete", delete_command))
                bot_app.add_handler(CommandHandler("reset", reset_command))
                bot_app.add_handler(CommandHandler("setprice", setprice_command))
                bot_app.add_handler(CommandHandler("setticker", setticker_command))
                bot_app.add_handler(CommandHandler("setqty", setqty_command))
                bot_app.add_handler(CommandHandler("alert", alert_command))
                bot_app.add_handler(CommandHandler("alerts", my_alerts_command))
                bot_app.add_handler(CommandHandler("paper", paper_command))
                bot_app.add_handler(CommandHandler("recall", recall_command))
                bot_app.add_handler(CommandHandler("learn", learn_command))
                bot_app.add_handler(CommandHandler("dbstatus", dbstatus_command))
                bot_app.add_handler(CommandHandler("benchmark", benchmark_command))
                bot_app.add_handler(CommandHandler("report", report_command))
                bot_app.add_handler(CommandHandler("backtest", backtest_command))
                bot_app.add_handler(CommandHandler("analyze", analyze_command))
                bot_app.add_handler(CommandHandler("settings", settings_command))
                bot_app.add_handler(CommandHandler("mode", mode_command))
                bot_app.add_handler(CommandHandler("macro", macro_command))
                bot_app.add_handler(CommandHandler("whale", whale_command))
                bot_app.add_handler(CommandHandler("rebalance", rebalance_command))
                bot_app.add_handler(CommandHandler("trainml", trainml_command))
                bot_app.add_handler(CommandHandler("sell", sell_command))
                bot_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
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
            except:
                pass
        
        report = bench.format_benchmark_report(period_days)
        await update.message.reply_text(report, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Benchmark command error: {e}")
        await update.message.reply_text(f"❌ Errore nel calcolo benchmark: {e}")

async def rebalance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show portfolio rebalancing analysis and suggestions."""
    await update.message.reply_text("📊 **Analisi Ribilanciamento Portfolio...**", parse_mode="Markdown")
    
    try:
        from rebalancer import Rebalancer
        rebalancer = Rebalancer()
        
        report = rebalancer.format_rebalance_report()
        
        # Split if too long
        if len(report) > 4000:
            parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
            for part in parts:
                try:
                    await update.message.reply_text(part, parse_mode="Markdown")
                except Exception:
                    # Markdown parse error - fallback to plain text
                    await update.message.reply_text(part)
        else:
            try:
                await update.message.reply_text(report, parse_mode="Markdown")
            except Exception as md_err:
                # Markdown parse error - fallback to plain text
                logger.warning(f"Rebalance markdown failed, sending plain text: {md_err}")
                await update.message.reply_text(report)
        
    except Exception as e:
        logger.error(f"Rebalance command error: {e}")
        await update.message.reply_text(f"❌ Errore nel calcolo ribilanciamento: {e}")

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
                await update.message.reply_text("⏳ **Avvio training ML...**\n\nPotrebbe richiedere 1-2 minuti.", parse_mode="Markdown")
                
                success = ml.train()
                
                if success:
                    new_stats = ml.get_dashboard_stats()
                    await update.message.reply_text(
                        f"✅ **Training Completato!**\n\n"
                        f"📦 Modello: `{ml.model_version}`\n"
                        f"📊 Accuracy: {new_stats.get('accuracy', 0):.1%}\n"
                        f"📈 Samples: {new_stats.get('training_samples', 0)}\n\n"
                        f"💡 Il modello ora userà ML per le predizioni!",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text("❌ Training fallito. Controlla i logs.")
                return
            else:
                remaining = ml.MIN_TRAINING_SAMPLES - training_count
                await update.message.reply_text(f"⏳ Servono altri **{remaining}** segnali chiusi.", parse_mode="Markdown")
                return
        
        # Show status
        msg = f"🤖 **ML Predictor Status**\n\n"
        msg += f"📦 **Modello:** `{version}`\n"
        msg += f"🎯 **ML Attivo:** {'✅ Pure Python GB' if is_ready else '❌ Rule-based'}\n"
        
        if accuracy:
            msg += f"📊 **Accuracy:** {accuracy:.1%}\n"
        
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
            realized_pnl = 0
        else:
            avg_price = asset.get('avg_price', price)
            # Calculate realized P&L
            realized_pnl = (price - avg_price) * quantity
        
        total_value = quantity * price
        pnl_percent = ((price - avg_price) / avg_price * 100) if avg_price > 0 else 0
        
        # Log the transaction
        result = db.log_transaction(
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price_per_unit=price,
            realized_pnl=realized_pnl
        )
        
        
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

async def handle_sell_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle confirmation/cancellation of pending sell from screenshot."""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    
    if action == "cancel_sell":
        context.user_data.pop('pending_sell', None)
        await query.edit_message_text("❌ Vendita annullata.")
        return
    
    if action == "confirm_sell" or action == "confirm_sell_manual":
        pending = context.user_data.get('pending_sell')
        if not pending:
            await query.edit_message_text("❌ Nessuna vendita in attesa.")
            return
        
        # Execute the sale
        from db_handler import DBHandler
        db = DBHandler()
        
        ticker = pending['ticker']
        quantity = pending['quantity']
        price = pending['price']
        
        # Determine values based on source (manual vs screenshot)
        if action == "confirm_sell_manual":
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
             commission = pending['commission']
        
        # Fetch portfolio to update
        portfolio = db.get_portfolio()
        asset = next((p for p in portfolio if p['ticker'].upper() == ticker.upper() or 
                     ticker.upper() in p['ticker'].upper()), None)
        
        # Log transaction with real net values
        result = db.log_transaction(
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price_per_unit=price,
            realized_pnl=profit
        )
        
        if result and asset:
            # Update portfolio quantity
            current_qty = float(asset.get('quantity', 0))
            new_qty = current_qty - quantity
            
            if new_qty <= 0:
                db.delete_asset(query.from_user.id, asset['ticker'])
                portfolio_msg = "\n\n🗑️ Asset rimosso dal portfolio (vendita totale)"
            else:
                db.update_asset_quantity(query.from_user.id, asset['ticker'], new_qty)
                portfolio_msg = f"\n\n📉 Portfolio aggiornato: {new_qty:.6f} unità rimanenti"
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

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a comprehensive weekly report with benchmarks, signals, and top movers."""
    await update.message.reply_text("📑 **Generazione Report Settimanale...**", parse_mode="Markdown")
    
    try:
        from benchmark import Benchmark
        from db_handler import DBHandler
        
        bench = Benchmark()
        db = DBHandler()
        
        # Get benchmark comparison (7 days for weekly)
        comparison = bench.compare_vs_benchmarks(7)
        movers = bench.get_top_movers(5)
        
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
    args = context.args
    if not args:
        await update.message.reply_text("⚠️ Uso: `/analyze <TICKER>` (es. `/analyze NVDA`)")
        return
    
    ticker = args[0].upper()
    await update.message.reply_text(f"🔬 **Analisi Strategica in corso su {ticker}...**\n_(Analizzo news, grafici e contesto. Richiede ~10s)_")
    
    try:
        # Initialize modules
        brain = Brain()
        hunter = NewsHunter()
        market = MarketData()
        db = DBHandler()
        
        # 1. Fetch Deep News
        logger.info(f"Analyze: Fetching news for {ticker}")
        news_items = hunter.fetch_ticker_news(ticker, limit=3)
        
        # 2. Fetch Technicals
        technical_summary = market.get_technical_summary(ticker)
        
        # 3. Validation - Only block if technicals are unknown (invalid ticker)
        # News is now optional - AI can analyze based on technicals alone
        if "Unknown" in technical_summary:
             await update.message.reply_text(f"❌ Impossibile analizzare **{ticker}**. Ticker non valido o dati tecnici non disponibili.", parse_mode="Markdown")
             return
    
    

        # 4. Check Portfolio & Allocations
        portfolio = db.get_portfolio()
        total_invested = 0.0
        asset_data = None
        
        # Calculate Total Cost Basis (Approx Portfolio Size) for Context
        for p in portfolio:
            qty = p.get('quantity', 0)
            avg = p.get('avg_price', 0)
            if qty and avg:
                total_invested += qty * avg
            
            # Find the specific asset (Robust Match)
            p_ticker = p.get('ticker','').upper()
            
            # Normalize: Remove -USD suffix for comparison
            # e.g. Input "BTC" matches DB "BTC-USD"
            # e.g. Input "BTC-USD" matches DB "BTC"
            def normalize(t): return t.replace('-USD', '')
            
            if normalize(p_ticker) == normalize(ticker):
                 asset_data = p
        
        portfolio_context = "Not Owned"
        if asset_data:
            qty = asset_data.get('quantity', 0)
            avg = asset_data.get('avg_price', 0)
            
            # Calculate Live Value of this asset
            live_price = 0.0
            # Try to extract live price from technical summary output? No, separate logic.
            # We use 'technical_summary' string which might contain price, but cleaner to fetch.
            # Reuse logic from smart fetch? 
            # We already validated ticker via news, let's trust market data or just use cost basis if simplest.
            # User wants "Current Value".
            # Let's peek at technical_summary content? It's a string.
            
            # Re-fetch price quickly using MarketData logic
            price_eur, _ = market.get_smart_price_eur(ticker)
            live_val = qty * price_eur if price_eur > 0 else (qty * avg)
            
            alloc_pct = 0.0
            if total_invested > 0:
                # Allocation vs Cost Basis is a decent proxy for "Size"
                alloc_pct = (live_val / total_invested) * 100
            
            size_desc = "Tiny" if alloc_pct < 2 else "Small" if alloc_pct < 5 else "Medium" if alloc_pct < 15 else "Large" if alloc_pct < 30 else "Huge"
            
            portfolio_context = (
                f"OWNED: {qty} units. Live Value: €{live_val:.2f}. "
                f"Total Portfolio (Approx): €{total_invested:.0f}. "
                f"Allocation: {alloc_pct:.1f}% ({size_desc}). "
                f"Avg Buy Price: €{avg:.2f}. "
                f"PnL: €{(live_val - (qty*avg)):.2f}."
            )

        # 5. Backtest Context (Historical Strategy Performance)
        backtest_context = ""
        try:
            from backtester import Backtester
            bt = Backtester()
            result = bt.run_best_strategy(ticker, 90)
            if result:
                backtest_context = (
                    f"BACKTEST RESULTS (90d, best strategy): "
                    f"Strategy={result['strategy_version']}, "
                    f"PnL={result['pnl_percent']:+.2f}%, "
                    f"Trades={result['total_trades']}, "
                    f"WinRate={result['win_rate']:.0f}%. "
                    f"This historical simulation used the best-performing strategy for this asset type."
                )
                logger.info(f"Analyze: Backtest context added for {ticker}")
        except Exception as e:
            logger.warning(f"Analyze: Backtest failed for {ticker}: {e}")
            backtest_context = "BACKTEST: Not available for this ticker."

        # 6. Fetch Macro Context (VIX, Yields, DXY, Fear&Greed)
        macro_context = None
        try:
            from economist import Economist
            eco = Economist()
            macro_context = eco.get_macro_summary()
            logger.info(f"Analyze: Macro context fetched ({len(macro_context)} chars)")
        except Exception as e:
            logger.warning(f"Analyze: Macro context failed: {e}")

        # 7. Fetch Whale Context (On-Chain Flows)
        whale_context = None
        try:
            from whale_watcher import WhaleWatcher
            ww = WhaleWatcher()
            whale_context = ww.analyze_flow()
            logger.info(f"Analyze: Whale context fetched ({len(whale_context)} chars)")
        except Exception as e:
            logger.warning(f"Analyze: Whale context failed: {e}")

        # 8. L1 Predictive System - Unified with /hunt
        l1_context = ""
        try:
            from signal_intelligence import SignalIntelligence
            from sentiment_aggregator import SentimentAggregator
            
            si = SignalIntelligence()
            
            # 8.1 Technical Confluence (RSI + SMA50 + Volume)
            confluence = si.check_technical_confluence(ticker, "BUY")  # Check BUY alignment
            confluence_text = f"Confluence: {confluence.get('alignment', 0)}/3"
            if confluence.get('reasons'):
                confluence_text += f" ({', '.join(confluence.get('reasons', [])[:2])})"
            logger.info(f"Analyze L1: {confluence_text}")
            
            # 8.2 Multi-Timeframe Analysis
            mtf = market.get_multi_timeframe_trend(ticker)
            mtf_text = f"MTF: {mtf.get('direction', 'unknown')} ({mtf.get('alignment', 0)}/3 timeframes aligned)"
            logger.info(f"Analyze L1: {mtf_text}")
            
            # 8.3 Sentiment Aggregator
            sentiment_agg = SentimentAggregator()
            agg_result = sentiment_agg.get_aggregated_score()
            sentiment_text = f"Market Sentiment: {agg_result.get('score', 50)}/100 ({agg_result.get('label', 'Neutral')}) → {agg_result.get('recommendation', 'HOLD')}"
            logger.info(f"Analyze L1: {sentiment_text}")
            
            # 8.4 Correlated Assets Info
            from correlation_engine import CorrelationEngine
            corr_engine = CorrelationEngine()
            correlated = corr_engine.get_correlated_assets(ticker)
            corr_text = ""
            if correlated:
                top_corr = correlated[:3]
                corr_text = f"Correlated: " + ", ".join([f"{c[0]} ({c[1]:.0%})" for c in top_corr])
            
            l1_context = f"""
**L1 PREDICTIVE ANALYSIS:**
- {confluence_text}
- {mtf_text} (Timeframes: {mtf.get('timeframes', {})})
- {sentiment_text}
- {corr_text}
"""
            logger.info(f"Analyze: L1 context added")
        except Exception as e:
            logger.warning(f"Analyze: L1 context failed: {e}")
            l1_context = "L1 Predictive: Not available"

        # 9. L2 Predictive System - Support/Resistance + Divergence
        l2_context = ""
        try:
            from support_resistance import SupportResistanceAI
            from signal_intelligence import SignalIntelligence
            
            # 9.1 Support/Resistance Levels
            sr_ai = SupportResistanceAI()
            sr_context = sr_ai.format_for_ai(ticker)
            logger.info(f"Analyze L2: S/R levels added")
            
            # 9.2 Divergence Detection
            si = SignalIntelligence()
            divergence = si.check_divergence(ticker)
            div_context = ""
            if divergence.get('has_divergence'):
                div_context = f"\nDivergence: {divergence.get('type').upper()} divergence detected (strength: {divergence.get('strength'):.0%})"
                logger.info(f"Analyze L2: {divergence.get('type')} divergence detected")
            
            # 9.3 Market Regime (for context)
            from market_regime import MarketRegimeClassifier
            regime = MarketRegimeClassifier().classify()
            regime_context = f"\nMarket Regime: {regime.get('regime')} ({regime.get('confidence'):.0%}) - {regime.get('recommendation')}"
            
            l2_context = f"""
**L2 PREDICTIVE ANALYSIS:**
{sr_context}
{div_context}
{regime_context}
"""
            logger.info(f"Analyze: L2 context added")
        except Exception as e:
            logger.warning(f"Analyze: L2 context failed: {e}")
            l2_context = "L2 Predictive: Not available"

        # 10. L3 Pattern Recognition (NEW)
        l3_context = ""
        try:
            from pattern_recognition import PatternRecognizer
            pr = PatternRecognizer()
            pattern_summary = pr.get_pattern_summary(ticker)
            
            if "No significant" not in pattern_summary:
                l3_context = f"""
**L3 PATTERN RECOGNITION:**
{pattern_summary}
"""
                logger.info(f"Analyze: L3 pattern context added for {ticker}")
            else:
                l3_context = "L3 Pattern: No significant chart patterns detected."
                logger.info(f"Analyze: No patterns detected for {ticker}")
        except Exception as e:
            logger.warning(f"Analyze: L3 pattern context failed: {e}")
            l3_context = "L3 Pattern: Not available"

        # 11. Generate Report (with ALL context: backtest, macro, whale, L1, L2, L3)
        logger.info(f"Analyze: Generating AI Report for {ticker}...")
        report = brain.generate_deep_dive(
            ticker, 
            news_items, 
            technical_summary, 
            portfolio_context, 
            backtest_context,
            macro_context,
            whale_context,
            l1_context + "\n" + l2_context + "\n" + l3_context  # Combined L1+L2+L3 predictive context
        )
        
        # 6. Send Report (Split if too long, though unlikely for this prompt)
        # Telegram limit is 4096 chars.
        # 6. Send Report with Markdown Fallback
        async def send_safe(text):
            try:
                if len(text) > 4000:
                    for x in range(0, len(text), 4000):
                        await update.message.reply_text(text[x:x+4000], parse_mode="Markdown")
                else:
                    await update.message.reply_text(text, parse_mode="Markdown")
            except Exception as e:
                logger.warning(f"Markdown failed, sending plain text: {e}")
                # Fallback to plain text if Markdown fails
                if len(text) > 4000:
                    for x in range(0, len(text), 4000):
                        await update.message.reply_text(text[x:x+4000])
                else:
                    await update.message.reply_text(text)

        await send_safe(report)

    except Exception as e:
        logger.error(f"Analyze Error: {e}")
        await update.message.reply_text("❌ Errore durante l'analisi. Riprova più tardi.")
