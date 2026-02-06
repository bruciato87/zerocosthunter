import os
import logging
import asyncio
import html
import re
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
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    @staticmethod
    def _markdownish_to_html(text: str) -> str:
        """
        Convert simple markdown-like formatting to safe HTML for Telegram.
        Supported:
        - **bold** -> <b>bold</b>
        - `code` -> <code>code</code>
        """
        safe = html.escape(text or "")
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe, flags=re.DOTALL)
        safe = re.sub(r"`([^`]+?)`", r"<code>\1</code>", safe)
        return safe

    async def send_message(self, chat_id, message):
        """Send a message to a specific chat_id with robust HTML-first fallback."""
        if not self.token:
            logger.info(f"Mock Alert (No Bot Configured): {message}")
            return
        
        if not chat_id:
            logger.error("Attempted to send message but Chat ID is None.")
            return

        try:
            async with Bot(token=self.token) as bot:
                try:
                    html_msg = self._markdownish_to_html(message)
                    await bot.send_message(chat_id=chat_id, text=html_msg, parse_mode='HTML')
                    logger.info(f"Telegram message sent to {chat_id} (HTML).")
                except Exception as html_err:
                    logger.warning(f"HTML send failed ({html_err}), retrying as plain text...")
                    clean_msg = (message or "").replace("**", "").replace("__", "").replace("`", "")
                    await bot.send_message(chat_id=chat_id, text=clean_msg)
                    logger.info(f"Telegram message sent to {chat_id} (Plain Text).")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t = TelegramNotifier()
    t.send_sync("Test message from Zero-Cost Hunter.")
