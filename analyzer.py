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
from run_observability import RunObservability

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Analyzer")

def _env_flag(name: str) -> bool:
    """Parse common truthy env values."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

class Analyzer:
    def __init__(self):
        self.dry_run = _env_flag("DRY_RUN")
        # Lazy imports to speed up startup
        from db_handler import DBHandler
        from market_data import MarketData
        from hunter import NewsHunter
        from brain import Brain
        from market_regime import MarketRegimeClassifier
        from ml_predictor import MLPredictor
        from whale_watcher import WhaleWatcher
        from critic import Critic
        
        self.db = DBHandler()
        self.market = MarketData()
        self.hunter = NewsHunter()
        self.brain = Brain()
        self.notifier = None
        if not self.dry_run:
            from telegram_bot import TelegramNotifier
            self.notifier = TelegramNotifier()
        self.regime_classifier = MarketRegimeClassifier()
        self.ml_predictor = MLPredictor()
        self.whale_watcher = WhaleWatcher()
        self.critic = Critic()
        if self.dry_run:
            logger.info("Analyzer DRY_RUN enabled: AI calls, DB writes, and Telegram sends are disabled.")
        
    async def analyze_ticker(self, ticker: str, target_chat_id: str):
        """
        Perform deep dive analysis on a ticker and send report to Telegram.
        """
        logger.info(f"🔬 Starting Deep Dive Analysis for {ticker}...")
        observer = RunObservability(
            "analyze",
            dry_run=self.dry_run,
            context={"ticker": ticker, "target_chat_id": str(target_chat_id)},
        )
        verdict = "HOLD"
        sentiment_score = 50
        
        try:
            # 1. Fetch News (Top 3)
            logger.info("Fetching news...")
            news_items = self.hunter.fetch_ticker_news(ticker, limit=3)
            
            # 2. Fetch Technicals
            logger.info("Fetching technicals...")
            technical_summary = self.market.get_technical_summary(ticker)
            
            # 2b. Check Market Regime (Quant Path)
            regime = "NEUTRAL"
            try:
                regime_data = self.regime_classifier.classify()
                regime = regime_data.get("regime", "NEUTRAL")
                logger.info(f"Market Regime: {regime}")
            except Exception as e:
                logger.warning(f"Regime Check failed: {e}")
                
            # 2c. Check Whale Flow (Whale Watcher Integration)
            whale_context = "N/A"
            try:
                whale_context = self.whale_watcher.analyze_flow()
                logger.info(f"Whale Context: {whale_context.splitlines()[2].strip() if len(whale_context.splitlines()) > 2 else 'Loaded'}")
            except Exception as e:
                logger.warning(f"Whale Watcher failed: {e}")
            
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
            
            # 4. Generate report
            if self.dry_run:
                logger.info("DRY_RUN: skipping AI deep-dive generation.")
                analysis_report = (
                    f"🧪 **DRY RUN: Deep Dive Analysis**\n\n"
                    f"Ticker: `{ticker}`\n"
                    f"News fetched: `{len(news_items)}`\n"
                    f"Technicals loaded: `{'yes' if technical_summary else 'no'}`\n"
                    f"Market regime: `{regime}`\n"
                    f"Portfolio context: {portfolio_context}\n"
                    f"Whale context loaded: `{'yes' if whale_context and whale_context != 'N/A' else 'no'}`\n\n"
                    "No AI inference, Telegram send, or DB persistence has been executed."
                )
            else:
                logger.info("🧠 Generating AI Deep Dive (R1)...")
                
                # 4.0 Quick Win #2: Valuation Context
                valuation_context = ""
                try:
                    val_data = self.market.get_valuation_context(ticker)
                    valuation_context = val_data.get('summary', '')
                    logger.info(f"Valuation: {val_data.get('valuation_vs_sector', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Valuation Context failed: {e}")
                
                analysis_report = self.brain.generate_deep_dive(
                    ticker=ticker,
                    news_list=news_items,
                    technical_data=technical_summary,
                    portfolio_context=portfolio_context,
                    macro_context=f"Market Regime: {regime}\nValuation: {valuation_context}",
                    whale_context=whale_context
                )
                
                # 4x. CRITIC CHECK (New)
                try:
                    from economist import Economist
                    eco = Economist()
                    market_summary = eco.get_macro_summary()
                    critic_note = self.critic.critique_deep_dive(ticker, analysis_report, market_summary)
                    analysis_report += critic_note
                except Exception as e:
                    logger.error(f"Critic integration failed: {e}")
                
                # 4a. Risk Management (ATR Upgrade)
                try:
                    atr_data = self.market.calculate_atr(ticker)
                    atr_val = atr_data.get("atr", 0)
                    price_eur, _ = self.market.get_smart_price_eur(ticker)
                    
                    if atr_val > 0 and price_eur > 0:
                        sl_atr = price_eur - (2.0 * atr_val)
                        tp_atr = price_eur + (4.0 * atr_val)
                        
                        risk_section = (
                            f"\n\n🛡️ **Quant Risk (ATR-based)**\n"
                            f"- Stop Loss: €{sl_atr:.2f}\n"
                            f"- Take Profit: €{tp_atr:.2f}\n"
                            f"- Volatility: {atr_data.get('volatility', 'unknown').upper()}"
                        )
                        analysis_report += risk_section
                except Exception as e:
                    logger.warning(f"ATR Risk calc failed: {e}")
                
                # Add AI Model Footer
                try:
                    footer = self.brain.get_usage_summary()
                    analysis_report += f"\n\n{footer}"
                except:
                    pass
            
            # 4b. Quant Path: Save ML Prediction Data
            try:
                # Extract Sentiment Score (Rough approx for now or parse from report)
                # For Deep Dive, R1 output is text. We use mapping based on verdict.
                verdict_map = {
                    "BUY": 85, "ACCUMULATE": 75, "WATCH": 55, "HOLD": 50, 
                    "SELL": 20, "AVOID": 15, "TRIM": 30
                }
                # Simple extraction from report text (Looking for "Decisione: [VERDICT]")
                import re
                verdict_match = re.search(r"Decisione:\s*\*?\[?([A-Z]+)\]?", analysis_report)
                sentiment_score = 50
                verdict = "HOLD"
                
                if verdict_match:
                    verdict = verdict_match.group(1).upper()
                    sentiment_score = verdict_map.get(verdict, 50)
                
                logger.info(f"Quant Data: Verdict={verdict}, Score={sentiment_score}, Regime={regime}")
                
                # Predict & Save
                if self.dry_run:
                    logger.info("DRY_RUN: skipping ML prediction persistence (Quant Path).")
                else:
                    self.ml_predictor.predict(ticker, sentiment_score, regime)
                    logger.info("✅ ML Data Point saved (Quant Path)")
                
            except Exception as e:
                logger.warning(f"Quant Path save failed: {e}")
            
            # 5. Send to Telegram
            if self.dry_run:
                logger.info("DRY_RUN: report generated, Telegram send skipped.")
                logger.info("DRY_RUN report preview: %s", analysis_report[:500].replace("\n", " "))
            else:
                logger.info(f"Sending report to {target_chat_id}...")
                await self.notifier.send_message(chat_id=target_chat_id, message=analysis_report)
            
            logger.info("✅ Analysis completed.")
            observer.finalize(
                status="success",
                summary="Analyze completed.",
                kpis={
                    "news_items_fetched": len(news_items) if isinstance(news_items, list) else 0,
                    "technicals_loaded": bool(technical_summary),
                    "portfolio_assets_scanned": len(portfolio) if isinstance(portfolio, list) else 0,
                    "report_length_chars": len(analysis_report),
                    "verdict_score": sentiment_score,
                },
                context={
                    "market_regime": regime,
                    "verdict": verdict,
                    "portfolio_owned": bool(portfolio_item),
                },
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            observer.add_error("analyze_ticker", e)
            observer.finalize(
                status="error",
                summary=f"Analyze failed for {ticker}.",
                context={"verdict": verdict, "last_known_regime": locals().get("regime", "NEUTRAL")},
            )
            error_msg = f"❌ **Errore Analisi {ticker}**: {str(e)[:200]}"
            if self.dry_run:
                logger.error("DRY_RUN: error notification to Telegram skipped.")
            else:
                await self.notifier.send_message(chat_id=target_chat_id, message=error_msg)
            sys.exit(1)

if __name__ == "__main__":
    # Input from Environment Variables (set by GitHub Action)
    target_ticker = os.environ.get("TARGET_TICKER")
    target_chat = os.environ.get("TARGET_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    
    if not target_ticker or not target_chat:
        logger.error("Missing TARGET_TICKER or TARGET_CHAT_ID/TELEGRAM_CHAT_ID env vars")
        sys.exit(1)
        
    analyzer = Analyzer()
    asyncio.run(analyzer.analyze_ticker(target_ticker, target_chat))
