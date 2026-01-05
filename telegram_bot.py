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
        
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials missing. Notifications disabled.")

    async def send_alert(self, message):
        """Send a formatted message to the default user."""
        await self.send_message(self.chat_id, message)
    
    async def send_message(self, chat_id, message):
        """Send a message to a specific chat_id."""
        if not self.token:
            logger.info(f"Mock Alert (No Bot Configured): {message}")
            return
        
        if not chat_id:
            chat_id = self.chat_id

        try:
            # Create a fresh Bot instance for each request to avoid event loop conflicts
            async with Bot(token=self.token) as bot:
                await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                logger.info(f"Telegram message sent to {chat_id}.")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")



if __name__ == "__main__":
    t = TelegramNotifier()
    t.send_sync("Test message from Zero-Cost Hunter.")
