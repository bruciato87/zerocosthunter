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
            logger.critical("❌ TELEGRAM_BOT_TOKEN IS MISSING! Notifications will NOT be sent.")
        if not self.chat_id:
            logger.critical("❌ TELEGRAM_CHAT_ID IS MISSING! Notifications will NOT be sent.")

    async def send_alert(self, message):
        """Send a formatted message to the default user."""
        await self.send_message(self.chat_id, message)
    
    def send_sync(self, message):
        """Synchronously send a message (helper for scripts)."""
        asyncio.run(self.send_alert(message))

    def escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2."""
        # Characters to escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
        # Note: We don't escape everything if we want to allow SOME formatting.
        # But for AI-generated reasoning, it's safer to escape most.
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        import re
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    async def send_message(self, chat_id, message):
        """Send a message to a specific chat_id."""
        if not self.token:
            logger.info(f"Mock Alert (No Bot Configured): {message}")
            return
        
        if not chat_id:
            logger.error("❌ Attempted to send message but Chat ID is None.")
            return

        try:
            # Create a fresh Bot instance for each request to avoid event loop conflicts
            async with Bot(token=self.token) as bot:
                try:
                    # In MarkdownV2, we must be very careful with escaping.
                    # This implementation assumes the message is already formatted OR we want to escape everything.
                    # Since we use **bold** and `code`, we should only escape the rest.
                    # For now, we try 'Markdown' (not V2) which is simpler but less robust.
                    # Actually, V2 is needed for nested parers.
                    await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                    logger.info(f"Telegram message sent to {chat_id}.")
                except Exception as md_err:
                    logger.warning(f"Markdown failed ({md_err}), retrying as plain text...")
                    # Strip common markdown tokens if retrying as plain text to clean up
                    clean_msg = message.replace("**", "").replace("__", "").replace("`", "")
                    await bot.send_message(chat_id=chat_id, text=clean_msg)
                    logger.info(f"Telegram message sent (plain text) to {chat_id}.")
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t = TelegramNotifier()
    t.send_sync("Test message from Zero-Cost Hunter.")
