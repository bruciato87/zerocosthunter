import os
import logging
import asyncio
import json
from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import yfinance as yf

# Add parent directory to sys.path to import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import Brain
from db_handler import DBHandler
from dotenv import load_dotenv

# Load env vars
load_dotenv()

app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VercelWebhook")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Initialize Bot Application (Global)
# In serverless, we rebuild the app on requests, or cache it if the container stays warm.
bot_app = ApplicationBuilder().token(TOKEN).build()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **Zero-Cost Hunter (Cloud Mode)**\n\n"
        "Mandami uno screenshot del tuo portafoglio Trade Republic."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    await update.message.reply_text("👀 Analizzo l'immagine...")

    try:
        # 1. Get Photo File
        photo = update.message.photo[-1]
        file_obj = await photo.get_file()
        
        # In Vercel (read-only FS), we download to /tmp
        file_path = f"/tmp/photo_{user_id}.jpg"
        await file_obj.download_to_drive(file_path)

        # 2. Analyze
        brain = Brain()
        holdings = brain.parse_portfolio_from_image(file_path)
        
        if not holdings:
            await update.message.reply_text("❌ Non ho trovato dati validi.")
            return

        # 2.1 MANUAL OVERRIDE (Caption)
        # If user provided a caption (e.g. "ICGA.F"), use it as the ticker for the first asset.
        caption = update.message.caption
        if caption and len(holdings) > 0:
            manual_ticker = caption.strip().upper()
            holdings[0]['ticker'] = manual_ticker
            # Also reset any "N/A" price if we have a valid ticker now, 
            # though live price is fetched later in /portfolio, so this is fine.
            logger.info(f"Manual Override: Set ticker to {manual_ticker} from caption.")
            await update.message.reply_text(f"✍️ **Override:** Uso il ticker manuale `{manual_ticker}`.")

        # 3. Save as DRAFT (including MERGE LOGIC)
        # Strategy:
        # A. Get existing drafts for this user.
        # B. If new item is Partial (missing ticker OR missing qty), look for complementary draft.
        # C. If found, UPDATE draft. If not, INSERT new draft.
        
        db = DBHandler()
        # 3. Save as DRAFT (including MERGE LOGIC)
        # Strategy:
        # A. Look for incomplete DRAFTS.
        # B. Look for incomplete RECENT CONFIRMED items (Late Merge).
        # C. Merge if possible.
        
        db = DBHandler()
        existing_drafts = db.get_drafts(chat_id)
        # Fetch confirmed items from last 5 minutes to handle "User confirmed too early" case
        recent_confirmed = db.get_recent_confirmed_portfolio(chat_id, minutes=5)
        
        msg_text = "✅ **Dati Estratti (Bozza):**\n"
        show_confirm_button = True
        
        for item in holdings:
            merged = False
            already_confirmed_merge = False
            new_ticker = item.get('ticker')
            new_qty = item.get('quantity')
            new_price = item.get('avg_price')
            
            # Helper to find merge candidate in a list
            def find_merge_candidate(candidate_list):
                 # Sort candidates by updated_at DESC to merge into most recent
                 
                 for c in candidate_list:
                      # Check Name Mismatch Safety Guard
                      # If both have names, and they don't look alike, SKIP.
                      db_name = c.get('asset_name', '').lower()
                      new_name_chk = item.get('name', '').lower()
                      
                      # Simple keyword safety: if one says "china" and other "world", don't merge.
                      if db_name and new_name_chk:
                           # Very basic protection against "MSCI China" merging into "MSCI World"
                           if "china" in new_name_chk and "china" not in db_name: continue
                           if "world" in new_name_chk and "world" not in db_name: continue
                           if "sp500" in new_name_chk and "500" not in db_name: continue
                      
                      # Case 1: Match UNKNOWN ticker in DB with New Valid Ticker -> Update DB Ticker
                      if c['ticker'] == 'UNKNOWN' and c['quantity'] and c['quantity'] > 0 and new_ticker and new_ticker != "UNKNOWN":
                           return c, "update_ticker"
                      
                      # Case 2: Match Valid Ticker in DB with New Valid Ticker -> Update DB Qty
                      if new_ticker and c['ticker'] == new_ticker and new_qty and new_qty > 0:
                           return c, "update_qty"

                      # Case 3: Match Valid Ticker in DB with New UNKNOWN Ticker (Detail View) -> Update DB Qty
                      if (not new_ticker or new_ticker == "UNKNOWN") and c['ticker'] != "UNKNOWN" and new_qty and new_qty > 0:
                           # Only merge if names don't conflict (handled above)
                           return c, "update_qty"

                 return None, None

            # 1. Try merging into DRAFTS first
            draft, action = find_merge_candidate(existing_drafts)
            if draft:
                if action == "update_qty":
                    db.update_draft_quantity(draft['id'], new_qty, new_price)
                    msg_text = "🧩 **Dati Integrati (Multimodale):**\n"
                    msg_text += f"• {draft['ticker']}: Quantità: **{new_qty}** | Prezzo: **€{new_price or 'N/A'}**\n"
                    msg_text += "\n_(Usa il tasto 'Conferma' del messaggio precedente)_"
                elif action == "update_ticker":
                    db.update_draft_ticker(draft['id'], new_ticker, new_price)
                    msg_text = "🧩 **Dati Integrati (Multimodale):**\n"
                    msg_text += f"• {new_ticker}: Associato a Quantità **{draft['quantity']}**\n"
                    msg_text += "\n_(Usa il tasto 'Conferma' del messaggio precedente)_"
                merged = True
                show_confirm_button = False # Disable new buttons, rely on previous ones.

            # 2. If not merged, try merging into RECENT CONFIRMED (Late Merge)
            if not merged:
                conf_item, action = find_merge_candidate(recent_confirmed)
                if conf_item:
                    if action == "update_qty":
                        db.update_draft_quantity(conf_item['id'], new_qty, new_price)
                        msg_text = "♻️ **Dati Aggiornati (Già Confermato):**\n"
                        msg_text += f"• {conf_item['ticker']}: Quantità: **{new_qty}** | Prezzo: **€{new_price or 'N/A'}**\n"
                    elif action == "update_ticker":
                        # This implies we saved a Confirmed UNKNOWN item? Rare but possible.
                        db.update_draft_ticker(conf_item['id'], new_ticker, new_price)
                        msg_text = "♻️ **Dati Aggiornati (Già Confermato):**\n"
                        msg_text += f"• {new_ticker}: Identificato asset salvato come sconosciuto.\n"
                    merged = True
                    already_confirmed_merge = True
                    show_confirm_button = False # Don't ask to confirm again

            if not merged:
                # Standard Insert (Upsert)
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
                display_name = item.get('name') or display_ticker
                display_type = item.get('asset_type', '')
                
                # Format: "• NVIDIA (Stock): ..."
                type_badge = f" ({display_type})" if display_type and display_type != "Unknown" else ""
                msg_text += f"• {display_name}{type_badge}: {new_qty} @ €{new_price}\n"

        # 4. Ask Confirmation (Only if needed)
        if show_confirm_button:
            keyboard = [
                [
                    InlineKeyboardButton("✅ Conferma e Salva", callback_data="confirm_save"),
                    InlineKeyboardButton("❌ Annulla", callback_data="cancel_save")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(msg_text, reply_markup=reply_markup)
        else:
            # Likely merged into confirmed, just notify
            await update.message.reply_text(msg_text)

    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("❌ Errore interno.")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    chat_id = update.effective_chat.id
    db = DBHandler()

    if data == "confirm_save":
        # Update all drafts for this chat_id to confirmed
        try:
            db.confirm_portfolio(chat_id)
            await query.edit_message_text(text="🚀 **Portafoglio Aggiornato!**")
        except Exception as e:
            await query.edit_message_text(text=f"❌ Errore DB: {e}")

    elif data == "cancel_save":
        # Delete drafts
        try:
            db.delete_drafts(chat_id)
            await query.edit_message_text(text="🗑️ Operazione annullata.")
        except Exception as e:
            await query.edit_message_text(text=f"❌ Errore: {e}")

async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    db = DBHandler()
    portfolio = db.get_portfolio(chat_id=chat_id)
    
    if not portfolio:
        await update.message.reply_text("📂 Il tuo portafoglio è vuoto.")
        return

    await update.message.reply_text("⏳ **Recupero prezzi live...**")

    msg = "📊 **Il tuo Portafoglio Aggiornato:**\n\n"
    total_assets = 0
    total_value_eur = 0.0
    
    # 1. Get FX Rate (EUR/USD) once
    eur_usd_rate = 1.1 # Safe default
    try:
        fx = yf.Ticker("EURUSD=X")
        hist = fx.history(period="1d")
        if not hist.empty:
            eur_usd_rate = hist['Close'].iloc[-1]
    except:
        pass
        
    for item in portfolio:
        ticker = item.get('ticker', 'N/A')
        name = item.get('asset_name') or ticker
        asset_type = item.get('asset_type', 'Unknown')
        qty = item.get('quantity', 0)
        avg_price = item.get('avg_price', 0)
        
        # Format strings
        type_str = f" ({asset_type})" if asset_type and asset_type != "Unknown" else ""
        
        # FETCH LIVE PRICE
        current_val_eur = 0.0
        current_price_label = "N/A"
        
        # Runtime Ticker Fixes for Yahoo Finance
        # Some tickers from TR (Xetra) don't match Yahoo or have migrated.
        TICKER_FIX_MAP = {
            "RNDR-USD": "RENDER-USD",  # Token migration
            "3DJ.DE": "3CP.F",         # Xiaomi: Xetra (.DE) often missing on Yahoo, use Frankfurt (.F) substitute
            "BYD": "BY6.F",            # BYD: Force Frankfurt if generic
            "ICGA.FRA": "ICGA.F",      # Fix for user manual input (Yahoo uses .F)
            "ICGA.DE": "ICGA.F",       # Fallback for Xetra too
            "3CP": "3CP.F"             # Fix for manual input without suffix
        }
        
        search_ticker = TICKER_FIX_MAP.get(ticker, ticker)
        
        if search_ticker and search_ticker != "UNKNOWN":
            try:
                t = yf.Ticker(search_ticker)
                hist = t.history(period="1d")
                if not hist.empty:
                    live_price = hist['Close'].iloc[-1]
                    
                    # Normalize Currency
                    # Logic: If ticker ends in .DE/.MI/.PA/.F -> EUR. Else -> USD (convert to EUR).
                    is_eur_market = search_ticker.endswith('.DE') or search_ticker.endswith('.MI') or search_ticker.endswith('.PA') or search_ticker.endswith('.F')
                    
                    if is_eur_market:
                        current_val_eur = qty * live_price
                    else:
                        # Convert USD price to EUR (Price / Rate)
                        # e.g. Price $110 / 1.1 = €100
                        live_price_eur = live_price / eur_usd_rate
                        current_val_eur = qty * live_price_eur
                        
                current_price_label = f"€{current_val_eur:,.2f}"
                total_value_eur += current_val_eur
            except Exception as e:
                logger.error(f"Price error for {search_ticker} (orig: {ticker}): {e}")
        
        # Build Message
        msg += f"🔹 **{name}**{type_str}\n"
        msg += f"   `{ticker}`\n"
        msg += f"   Qty: `{qty}`\n"
        msg += f"   Avg: `€{avg_price}`\n"
        if current_val_eur > 0:
             msg += f"   Val: `{current_price_label}`\n\n"
        else:
             msg += f"   Val: `N/A`\n\n"

        total_assets += 1

    msg += f"-----------------------------\n"
    msg += f"💰 **Totale:** `€{total_value_eur:,.2f}`\n"
    msg += f"🔢 **Asset:** {total_assets}\n"
    
    await update.message.reply_text(msg)

async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /delete <ticker> command."""
    chat_id = update.effective_chat.id
    try:
        if not context.args:
            await update.message.reply_text("⚠️ Uso: `/delete <TICKER>` (es. `/delete AAPL`)")
            return

        ticker = context.args[0].upper()
        db = DBHandler()
        success = db.delete_asset(chat_id, ticker)
        
        if success:
            await update.message.reply_text(f"🗑️ Asset `{ticker}` eliminato dal portafoglio.")
        else:
            await update.message.reply_text(f"❌ Errore durante l'eliminazione di `{ticker}`. Forse non esiste?")
    except Exception as e:
        logger.error(f"Delete command error: {e}")
        await update.message.reply_text("❌ Errore nel comando.")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /reset command to clear entire portfolio."""
    chat_id = update.effective_chat.id
    db = DBHandler()
    if db.delete_portfolio(chat_id):
        await update.message.reply_text("☢️ **Portafoglio Azzerato.**\nTutti i dati sono stati cancellati.")
    else:
        await update.message.reply_text("❌ Errore durante il reset.")

async def setprice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /setprice <ticker> <price>."""
    chat_id = update.effective_chat.id
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text("⚠️ Uso: `/setprice <TICKER> <PREZZO>`\nEs: `/setprice AAPL 150.50`")
        return

    ticker = args[0].upper()
    try:
        new_price = float(args[1].replace(',', '.'))
        db = DBHandler()
        if db.update_asset_price(chat_id, ticker, new_price):
            await update.message.reply_text(f"✅ Prezzo di `{ticker}` aggiornato a **€{new_price}**.")
        else:
            await update.message.reply_text(f"❌ Asset `{ticker}` non trovato.")
    except ValueError:
        await update.message.reply_text("⚠️ Prezzo non valido.")

async def setticker_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /setticker <old> <new>."""
    chat_id = update.effective_chat.id
    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text("⚠️ Uso: `/setticker <VECCHIO> <NUOVO>`\nEs: `/setticker 3DJ.DE 3CP.F`")
        return

    old_ticker = args[0].upper()
    new_ticker = args[1].upper()
    
    db = DBHandler()
    if db.update_asset_ticker(chat_id, old_ticker, new_ticker):
        await update.message.reply_text(f"✅ Ticker aggiornato: `{old_ticker}` ➡️ `{new_ticker}`.")
    else:
        await update.message.reply_text(f"❌ Asset `{old_ticker}` non trovato.")

# Register Handlers
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(CommandHandler("help", help_command)) # NEW
bot_app.add_handler(CommandHandler("portfolio", show_portfolio))
bot_app.add_handler(CommandHandler("delete", delete_command))
bot_app.add_handler(CommandHandler("reset", reset_command))
bot_app.add_handler(CommandHandler("setprice", setprice_command))
bot_app.add_handler(CommandHandler("setticker", setticker_command))
bot_app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
bot_app.add_handler(CallbackQueryHandler(handle_callback))

@app.route('/api/webhook', methods=['POST'])
def webhook():
    if request.method == "POST":
        json_update = request.get_json()
        if json_update:
            # Run async loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                update = Update.de_json(json_update, bot_app.bot)
                
                # Check auth token manually to prevent spam if needed? 
                # Nope, simpler to just process.
                
                # Process update using Application.process_update
                # Note: initialize() is async
                loop.run_until_complete(bot_app.initialize())
                loop.run_until_complete(bot_app.process_update(update))
                loop.run_until_complete(bot_app.shutdown())
                
            except Exception as e:
                logger.error(f"Error in webhook: {e}")
                return "Error", 500
                
        return "OK", 200
    return "OK", 200
