import os
import logging
import asyncio
from telegram import Bot

# Configure logging
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if not self.token:
            logger.critical("TELEGRAM_BOT_TOKEN IS MISSING! Notifications will NOT be sent.")
        if not self.chat_id:
            logger.critical("TELEGRAM_CHAT_ID IS MISSING! Notifications will NOT be sent.")

    async def send_alert(self, message):
        """Send a formatted message to the default user."""
        await self.send_message(self.chat_id, message)
    
    def send_sync(self, message):
        """Synchronously send a message (helper for scripts)."""
        asyncio.run(self.send_alert(message))

    def escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2."""
        # Characters to escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        import re
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    async def send_message(self, chat_id, message):
        """Send a message to a specific chat_id with tiered fallbacks."""
        if not self.token:
            logger.info(f"Mock Alert (No Bot Configured): {message}")
            return
        
        if not chat_id:
            logger.error("Attempted to send message but Chat ID is None.")
            return

        try:
            async with Bot(token=self.token) as bot:
                try:
                    # Attempt 1: Markdown (legacy, more forgiving for simple bold/code)
                    await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                    logger.info(f"Telegram message sent to {chat_id} (Markdown).")
                except Exception:
                    try:
                        # Attempt 2: HTML (often more robust for AI text)
                        # We convert **bold** and `code` to HTML equivalents using regex for pairs
                        import re
                        html_msg = message
                        # Bold: **text** -> <b>text</b>
                        html_msg = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_msg)
                        # Code: `text` -> <code>text</code>
                        html_msg = re.sub(r'`(.*?)`', r'<code>\1</code>', html_msg)
                        
                        await bot.send_message(chat_id=chat_id, text=html_msg, parse_mode='HTML')
                        logger.info(f"Telegram message sent to {chat_id} (HTML Fallback).")
                    except Exception as html_err:
                        logger.warning(f"HTML fallback failed ({html_err}), retrying as plain text...")
                        # Attempt 3: Plain text (Guaranteed)
                        clean_msg = message.replace("**", "").replace("__", "").replace("`", "")
                        await bot.send_message(chat_id=chat_id, text=clean_msg)
                        logger.info(f"Telegram message sent to {chat_id} (Plain Text).")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t = TelegramNotifier()
    t.send_sync("Test message from Zero-Cost Hunter.")
