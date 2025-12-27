import os
import logging
import asyncio
import json
from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

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
                 # Sort candidates by created_at DESC to merge into most recent
                 # (Assuming candidate_list has 'created_at', but it's a dict from Supabase)
                 # Supabase returns by insertion order usually, or we can rely on list order if it's implicitly ordered.
                 # Better to iterate reversed if we want most recent? 
                 # Let's just iterate as is, assuming list is recent-first or close enough.
                 
                 for c in candidate_list:
                      # Case 1: Match UNKNOWN ticker in DB with New Valid Ticker -> Update DB Ticker
                      if c['ticker'] == 'UNKNOWN' and c['quantity'] and c['quantity'] > 0 and new_ticker and new_ticker != "UNKNOWN":
                           return c, "update_ticker"
                      
                      # Case 2: Match Valid Ticker in DB with New Valid Ticker -> Update DB Qty
                      if new_ticker and c['ticker'] == new_ticker and new_qty and new_qty > 0:
                           return c, "update_qty"

                      # Case 3: Match Valid Ticker in DB with New UNKNOWN Ticker (Detail View) -> Update DB Qty
                      # This fixes the "EUNL.DE" vs "UNKNOWN" issue.
                      # We assume if user uploads a Detail View immediately after, it belongs to this Draft.
                      if (not new_ticker or new_ticker == "UNKNOWN") and c['ticker'] != "UNKNOWN" and new_qty and new_qty > 0:
                           # To be safe, maybe check if created_at is very recent?
                           # For now, trust the user workflow (Upload A -> Upload B).
                           return c, "update_qty"

                 return None, None

            # 1. Try merging into DRAFTS first
            draft, action = find_merge_candidate(existing_drafts)
            if draft:
                if action == "update_qty":
                    db.update_draft_quantity(draft['id'], new_qty)
                    msg_text = "🧩 **Dati Integrati (Multimodale):**\n"
                    msg_text += f"• {draft['ticker']}: Quantità corretta a **{new_qty}** (da dettaglio)\n"
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
                        db.update_draft_quantity(conf_item['id'], new_qty)
                        msg_text = "♻️ **Dati Aggiornati (Già Confermato):**\n"
                        msg_text += f"• {conf_item['ticker']}: Quantità corretta a **{new_qty}**\n"
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
                    is_confirmed=False, 
                    chat_id=chat_id
                )
                display_ticker = new_ticker if new_ticker else "⚠️ Sconosciuto (Carica dettaglio esteso)"
                msg_text += f"• {display_ticker}: {new_qty} @ ${new_price}\n"

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

    msg = "📊 **Il tuo Portafoglio:**\n\n"
    total_assets = 0
    
    for item in portfolio:
        ticker = item.get('ticker', 'N/A')
        qty = item.get('quantity', 0)
        avg_price = item.get('avg_price', 0)
        
        # Simple formatting
        msg += f"🔹 **{ticker}**\n"
        msg += f"   Qty: `{qty}`\n"
        msg += f"   Avg: `${avg_price}`\n\n"
        total_assets += 1

    msg += f"Totale Asset: {total_assets}"
    await update.message.reply_text(msg)

# Register Handlers
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(CommandHandler("portfolio", show_portfolio))
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
