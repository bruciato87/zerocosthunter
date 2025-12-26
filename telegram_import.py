import logging
import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
from brain import Brain
from db_handler import DBHandler

# Configure Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("TelegramImporter")

# Global temporary storage for confirmation (simplest approach for single user)
# In production, use context.user_data
PENDING_DATA = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **Ciao! Sono il tuo Portfolio Importer.**\n\n"
        "Mandami uno **screenshot** della tua app Trade Republic (o qualsiasi lista investimenti).\n"
        "Io estrarrò i dati e ti chiederò conferma prima di salvare.",
        parse_mode='Markdown'
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    await update.message.reply_text("👀 **Analisi in corso...** Dammi qualche secondo.")

    try:
        # 1. Download Photo
        photo_file = await update.message.photo[-1].get_file()
        file_path = f"temp_portfolio_{user_id}.jpg"
        await photo_file.download_to_drive(file_path)

        # 2. Analyze with Gemini
        brain = Brain()
        holdings = brain.parse_portfolio_from_image(file_path)
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        if not holdings:
            await update.message.reply_text("❌ Non sono riuscito a trovare dati leggibili. Riprova con uno screenshot più chiaro.")
            return

        # 3. Present Data
        msg = "✅ **Ho trovato questi investimenti:**\n\n"
        for i, item in enumerate(holdings):
            msg += f"{i+1}. **{item['ticker']}**: {item['quantity']} pz @ ${item['avg_price']} ({item['sector']})\n"
        
        msg += "\nScrivi **SI** per salvare nel database, o manda un'altra foto per riprovare."
        
        # Store for confirmation
        PENDING_DATA[user_id] = holdings
        
        await update.message.reply_text(msg, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await update.message.reply_text(f"❌ Errore interno: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip().upper()

    if text in ["SI", "SÌ", "YES", "OK", "CONFIRM"]:
        if user_id not in PENDING_DATA:
            await update.message.reply_text("⚠️ Nessun dato in attesa. Mandami prima una foto.")
            return
        
        holdings = PENDING_DATA[user_id]
        db = DBHandler()
        
        count = 0
        try:
            for item in holdings:
                db.add_to_portfolio(
                    ticker=item['ticker'],
                    amount=item['quantity'],
                    price=item['avg_price'],
                    sector=item['sector']
                )
                count += 1
            
            del PENDING_DATA[user_id]
            await update.message.reply_text(f"🚀 **Fatto!** Salvati {count} investimenti nel Database.\nIl bot ora li userà per le analisi.")
            
        except Exception as e:
            logger.error(f"DB Error: {e}")
            await update.message.reply_text("❌ Errore nel salvataggio su DB.")
            
    else:
        await update.message.reply_text("Scrivi **SI** per confermare, o manda una foto.")

if __name__ == '__main__':
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN not found!")
        exit(1)

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    logger.info("🤖 Telegram Importer is listening...")
    application.run_polling()
