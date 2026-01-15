"""
Analyzer Module - Deep Dive Analysis via GitHub Actions
=======================================================
Runs comprehensive AI analysis on a specific ticker using DeepSeek R1.
Triggered asynchronously via GitHub Actions.
"""

import os
import asyncio
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Analyzer")

class Analyzer:
    def __init__(self):
        # Lazy imports to speed up startup
        from db_handler import DBHandler
        from market_data import MarketData
        from hunter import NewsHunter
        from brain import Brain
        from telegram_bot import TelegramNotifier
        
        self.db = DBHandler()
        self.market = MarketData()
        self.hunter = NewsHunter()
        self.brain = Brain()
        self.notifier = TelegramNotifier()
        
    async def analyze_ticker(self, ticker: str, target_chat_id: str):
        """
        Perform deep dive analysis on a ticker and send report to Telegram.
        """
        logger.info(f"🔬 Starting Deep Dive Analysis for {ticker}...")
        
        try:
            # 1. Fetch News (Top 3)
            logger.info("Fetching news...")
            news_items = self.hunter.fetch_ticker_news(ticker, limit=3)
            
            # 2. Fetch Technicals
            logger.info("Fetching technicals...")
            technical_summary = self.market.get_technical_summary(ticker)
            
            # 3. Check Portfolio Context
            portfolio = self.db.get_portfolio()
            portfolio_item = None
            total_invested = 0.0
            
            # Find asset and calc total invested
            for p in portfolio:
                qty = float(p.get('quantity', 0))
                price = float(p.get('avg_price', 0))
                total_invested += qty * price
                
                # Normalize ticker check
                p_ticker = p.get('ticker','').upper().replace('-USD','')
                t_norm = ticker.upper().replace('-USD','')
                
                if p_ticker == t_norm:
                    portfolio_item = p
            
            portfolio_context = "Not Owned"
            if portfolio_item:
                qty = float(portfolio_item.get('quantity', 0))
                avg_price = float(portfolio_item.get('avg_price', 0))
                curr_price, _ = self.market.get_smart_price_eur(ticker)
                
                pnl_pct = ((curr_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
                allocation = ((qty * curr_price) / total_invested * 100) if total_invested > 0 else 0
                
                portfolio_context = (
                    f"OWNED: {qty:.4f} units @ €{avg_price:.2f}. "
                    f"Current PnL: {pnl_pct:+.1f}%. Allocation: {allocation:.1f}% of total portfolio."
                )
            
            # 4. Generate AI Report (DeepSeek R1)
            logger.info("🧠 Generating AI Deep Dive (R1)...")
            
            analysis_report = self.brain.generate_deep_dive(
                ticker=ticker,
                news_list=news_items,
                technical_data=technical_summary,
                portfolio_context=portfolio_context
            )
            
            # 5. Send to Telegram
            logger.info(f"Sending report to {target_chat_id}...")
            await self.notifier.send_message(chat_id=target_chat_id, message=analysis_report)
            
            logger.info("✅ Analysis completed and sent.")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            error_msg = f"❌ **Errore Analisi {ticker}**: {str(e)[:200]}"
            await self.notifier.send_message(chat_id=target_chat_id, message=error_msg)
            sys.exit(1)

if __name__ == "__main__":
    # Input from Environment Variables (set by GitHub Action)
    target_ticker = os.environ.get("TARGET_TICKER")
    target_chat = os.environ.get("TARGET_CHAT_ID")
    
    if not target_ticker or not target_chat:
        logger.error("Missing TARGET_TICKER or TARGET_CHAT_ID env vars")
        sys.exit(1)
        
    analyzer = Analyzer()
    asyncio.run(analyzer.analyze_ticker(target_ticker, target_chat))
