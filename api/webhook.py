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

logging.basicConfig(level=logging.INFO)
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

async def setup_bot_commands(bot):
    """Configures the menu button in Telegram UI."""
    commands = [
        BotCommand("hunt", "🏹 Caccia Manuale (Analisi News)"),
        BotCommand("analyze", "🔬 Deep Dive Ticker (es. /analyze NVDA)"),
        BotCommand("portfolio", "📊 Vedi Portafoglio & Valore Live"),
        BotCommand("dashboard", "🖥️ Web Dashboard"),
        BotCommand("macro", "🏛 Macro Strategist Context"),
        BotCommand("help", "❓ Lista Comandi"),
        BotCommand("setprice", "💶 Correggi Prezzo (es. /setprice AAPL 150)"),
        BotCommand("setticker", "🏷 Correggi Ticker (es. /setticker OLD NEW)"),
        BotCommand("delete", "🗑 Elimina un Asset"),
        BotCommand("settings", "⚙️ Configura Smart Filters"),
        BotCommand("reset", "☢️ Reset Totale"),
        BotCommand("start", "🚀 Avvia"),
    ]
    await bot.set_my_commands(commands)

async def hunt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏹 **Caccia Iniziata!**\nAnalizzo le news... attendi qualche secondo.")
    try:
        await run_async_pipeline()
        await update.message.reply_text("✅ **Caccia Completata.**\nSe ho trovato segnali, te li ho inviati.")
    except Exception as e:
        logger.error(f"Manual hunt error: {e}")
        await update.message.reply_text(f"❌ Errore durante la caccia: {e}")

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

from whale_watcher import WhaleWatcher
from economist import Economist

# ... (Previous imports)

async def setup_bot_commands(bot):
    """Configures the menu button in Telegram UI."""
    commands = [
        BotCommand("hunt", "🏹 Caccia Manuale (Analisi News)"),
        BotCommand("analyze", "🔬 Deep Dive Ticker (es. /analyze NVDA)"),
        BotCommand("portfolio", "📊 Vedi Portafoglio & Valore Live"),
        BotCommand("dashboard", "🖥️ Web Dashboard"),
        BotCommand("macro", "🏛 Macro Context (FED/VIX)"),
        BotCommand("whale", "🐋 Whale Alert (On-Chain)"),
        BotCommand("help", "❓ Lista Comandi"),
        BotCommand("setprice", "💶 Correggi Prezzo (es. /setprice AAPL 150)"),
        BotCommand("setticker", "🏷 Correggi Ticker (es. /setticker OLD NEW)"),
        BotCommand("delete", "🗑 Elimina un Asset"),
        BotCommand("settings", "⚙️ Configura Smart Filters"),
        BotCommand("reset", "☢️ Reset Totale"),
        BotCommand("start", "🚀 Avvia"),
    ]
    await bot.set_my_commands(commands)

    await bot.set_my_commands(commands)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await setup_bot_commands(context.bot)
    msg = (
        "🛠 **Lista Comandi Disponibili:**\n\n"
        "📊 `/portfolio`\nVisualizza il valore attuale del tuo portafoglio in tempo reale.\n\n"
        "🏛 `/macro`\nVisualizza il contesto Macro Economico (VIX, Tassi, FED).\n\n"
        "🐋 `/whale`\nVisualizza movimenti On-Chain (Balene).\n\n"
        "✍️ **Correzioni Manuali:**\n"
        "• `/setprice <TICKER> <PREZZO>`: Imposta manualmente il prezzo medio.\n"
        "• `/setticker <OLD> <NEW>`: Rinomina un ticker errato.\n\n"
        "🗑 **Gestione:**\n"
        "• `/delete <TICKER>`: Elimina un singolo asset.\n"
        "• `/reset`: Cancella TUTTO il portafoglio.\n\n"
        "⚙️ `/settings`: Configura filtri AI.\n\n"
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
    if query.data == "confirm_save":
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
    
    msg = "📊 **Il tuo Portafoglio:**\n\n"
    total_val = 0.0
    eur_usd = 1.1
    try:
        t = yf.Ticker("EURUSD=X")
        eur_usd = t.history(period="1d")['Close'].iloc[-1]
    except: pass

    TICKER_FIX_MAP = {"RNDR-USD": "RENDER-USD", "3DJ.DE": "3CP.F", "BYD": "BY6.F", "ICGA.FRA": "IAG.MC", "ICGA.DE": "IAG.MC", "ICGA.F": "IAG.MC", "3CP": "3CP.F"}

    for item in portfolio:
        ticker = item.get('ticker', 'N/A')
        search = TICKER_FIX_MAP.get(ticker, ticker)
        qty = item.get('quantity', 0)
        curr_val = 0.0
        
        if search and search != "UNKNOWN":
            try:
                hist = yf.Ticker(search).history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    if search.endswith(('.DE','.F','.MI','.PA')):
                        curr_val = qty * price
                    else:
                        curr_val = qty * (price / eur_usd)
            except: pass
        
        total_val += curr_val
        val_str = f"€{curr_val:,.2f}" if curr_val > 0 else "N/A"
        msg += f"🔹 **{item.get('asset_name') or ticker}** ({item.get('asset_type','Unknown')})\n   Ticker: `{ticker}`\n   Qty: {qty} | Val: {val_str}\n\n"

    msg += f"-----------------------------\n💰 **Totale:** `€{total_val:,.2f}`"
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
    
    TICKER_FIX = {"RNDR-USD": "RENDER-USD", "3DJ.DE": "3CP.F", "BYD": "BY6.F", "ICGA.FRA": "IAG.MC", "ICGA.F": "IAG.MC", "ICGA.DE": "IAG.MC", "3CP": "3CP.F"}

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
                hist = yf.Ticker(search).history(period="5d")
                if not hist.empty:
                    close = hist['Close'].iloc[-1]
                    is_eu = search.endswith(('.DE','.F','.MI','.PA'))
                    curr = qty * close if is_eu else qty * (close / eur_usd)
                    
                    # Store per-asset trend
                    asset_trends[ticker] = {}
                    for dt, val in hist['Close'].items():
                        d_str = dt.strftime("%Y-%m-%d")
                        v_eu = qty * val if is_eu else qty * (val / eur_usd)
                        asset_trends[ticker][d_str] = v_eu

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

    total_pl = total_val - total_inv
    pl_pct = (total_pl/total_inv*100) if total_inv > 0 else 0
    
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

    return render_template('dashboard.html', 
                           signals=signals, 
                           portfolio=portfolio, 
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
                           whale_stats=whale_stats)

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
                bot_app.add_handler(CommandHandler("settings", settings_command))
                bot_app.add_handler(CommandHandler("analyze", analyze_command))
                bot_app.add_handler(CommandHandler("macro", macro_command))
                bot_app.add_handler(CommandHandler("whale", whale_command))
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
        
        # 3. Validation
        if not news_items and "Unknown" in technical_summary:
             await update.message.reply_text(f"❌ Impossibile analizzare **{ticker}**. Ticker non valido o nessuna news trovata.")
             return

        # 4. Check Portfolio
        portfolio_map = db.get_portfolio_map()
        portfolio_context = "Not Owned"
        if ticker in portfolio_map:
            p = portfolio_map[ticker]
            portfolio_context = f"OWNED: {p['quantity']} units @ €{p['avg_price']}. Make sure to suggest TAKING PROFIT or AVERAGING DOWN."

        # 5. Generate Report
        logger.info(f"Analyze: Generating AI Report for {ticker}...")
        report = brain.generate_deep_dive(ticker, news_items, technical_summary, portfolio_context)
        
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
