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
        existing_drafts = db.get_drafts(chat_id)
        msg_text = "✅ **Dati Estratti (Bozza):**\n"
        
        for item in holdings:
            merged = False
            new_ticker = item.get('ticker')
            new_qty = item.get('quantity')
            new_price = item.get('avg_price')
            
            # Case 1: Partial - Quantity Found, Ticker UNKNOWN (Detail View)
            if (not new_ticker or new_ticker == "UNKNOWN") and new_qty and new_qty > 0:
                 # Find draft with Valid Ticker but Missing/Bad Qty
                 for draft in existing_drafts:
                      # Check if draft has valid ticker AND (qty is missing or suspect calc)
                      # We assume user uploads Ticker view then Detail view.
                      if draft['ticker'] != 'UNKNOWN':
                       # Merge! Update the draft with the precise quantity
                           db.update_draft_quantity(draft['id'], new_qty)
                           merged = True
                           msg_text = "🧩 **Dati Integrati (Multimodale):**\n"
                           msg_text += f"• {draft['ticker']}: Quantità corretta a **{new_qty}** (da dettaglio)\n"
                           break
            
            # Case 2: Partial - Ticker Found, Quantity Missing (Header View)
            elif new_ticker and new_ticker != "UNKNOWN" and (not new_qty or new_qty == 0):
                 # Find draft with UNKNOWN Ticker but Valid Qty
                 for draft in existing_drafts:
                      if draft['ticker'] == 'UNKNOWN' and draft['quantity'] and draft['quantity'] > 0:
                           # Merge! Set ticker for this quantity
                           db.update_draft_ticker(draft['id'], new_ticker, new_price)
                           merged = True
                           msg_text = "🧩 **Dati Integrati (Multimodale):**\n"
                           msg_text += f"• {new_ticker}: Associato a Quantità **{draft['quantity']}**\n"
                           break

            if not merged:
                # Standard Insert (Upsert)
                # If ticker is UNKNOWN, we still insert it if it has useful data (Qty), hoping for future merge.
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

        # 4. Ask Confirmation
        keyboard = [
            [
                InlineKeyboardButton("✅ Conferma e Salva", callback_data="confirm_save"),
                InlineKeyboardButton("❌ Annulla", callback_data="cancel_save")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(msg_text, reply_markup=reply_markup)

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

# Register Handlers
bot_app.add_handler(CommandHandler("start", start))
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
