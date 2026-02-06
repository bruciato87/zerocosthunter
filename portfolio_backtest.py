"""
portfolio_backtest.py
Level 11: Portfolio Backtest - Historical Performance Analysis

Simulates the performance of the investment strategy over historical data
and calculates risk-adjusted metrics like Sharpe Ratio and Max Drawdown.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PortfolioBacktest:
    """
    Simulates historical portfolio performance based on strategy rules.
    """
    
    # Risk-free rate for Sharpe ratio (Italian BOT ~3.5% in 2026)
    RISK_FREE_RATE = 0.035
    
    def __init__(self):
        """Initialize the Portfolio Backtest module."""
        pass
    
    def run_backtest(self, portfolio: List[Dict], period_days: int = 365) -> Dict:
        """
        Run a historical backtest for the given portfolio.
        
        Args:
            portfolio: List of portfolio items [{"ticker": "BTC-USD", "qty": 0.5, "avg_price": 50000}, ...]
            period_days: Number of days to backtest (default 365)
        
        Returns:
            {
                "total_return_pct": 25.5,
                "annualized_return_pct": 28.3,
                "sharpe_ratio": 1.45,
                "max_drawdown_pct": -15.2,
                "volatility_pct": 18.5,
                "win_rate": 0.65,
                "best_performer": {"ticker": "BTC-USD", "return": 45.2},
                "worst_performer": {"ticker": "META", "return": -12.5},
                "daily_returns": [...],  # For charting
                "cumulative_returns": [...]  # For charting
            }
        """
        try:
            if not portfolio:
                return {"error": "Empty portfolio"}
            
            # Extract tickers and normalize
            tickers = []
            weights = {}
            total_value = 0
            
            for item in portfolio:
                ticker = item.get('ticker', '').upper()
                # Support both field names (qty/quantity, avg_price/avg_price_eur)
                qty = item.get('qty', item.get('quantity', 0))
                price = item.get('current_price', item.get('avg_price', item.get('avg_price_eur', 0)))
                value = float(qty) * float(price) if qty and price else 0
                
                if value > 0:
                    tickers.append(ticker)
                    total_value += value
            
            if total_value == 0:
                return {"error": "No portfolio value"}
            
            # Calculate weights
            for item in portfolio:
                ticker = item.get('ticker', '').upper()
                qty = item.get('qty', item.get('quantity', 0))
                price = item.get('current_price', item.get('avg_price', item.get('avg_price_eur', 0)))
                value = float(qty) * float(price) if qty and price else 0
                weights[ticker] = value / total_value if total_value > 0 else 0
            
            # Normalize tickers for yfinance using DB cache
            yf_tickers = []
            ticker_map = {}  # yf_ticker -> original_ticker
            
            # Try to use DB cache for resolution
            def resolve_ticker(t):
                """Resolve ticker using DB cache first, then fallback logic."""
                try:
                    from db_handler import DBHandler
                    db = DBHandler()
                    cached = db.get_ticker_cache(t.upper())
                    if cached:
                        return cached.get("resolved_ticker", t)
                except:
                    pass
                
                # Fallback logic for crypto
                crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'RENDER', 'DOGE', 'ADA', 'DOT', 'LINK', 'AVAX', 'MATIC']
                base = t.replace('-USD', '').replace('-EUR', '')
                if base.upper() in crypto_list and not t.endswith('-USD') and not t.endswith('-EUR'):
                    return f"{base}-USD"
                return t
            
            for t in tickers:
                yf_t = resolve_ticker(t)
                yf_tickers.append(yf_t)
                ticker_map[yf_t] = t
            
            # Download historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            data = yf.download(yf_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if data.empty:
                return {"error": "No historical data available"}
            
            # Extract Close prices
            if 'Close' in data.columns:
                if hasattr(data['Close'], 'columns'):
                    close_prices = data['Close']
                else:
                    close_prices = data[['Close']]
                    close_prices.columns = yf_tickers[:1]
            else:
                close_prices = data
            
            # Drop columns with too many NaN values (ticker not found)
            close_prices = close_prices.dropna(axis=1, thresh=int(len(close_prices) * 0.5))
            
            if close_prices.empty or len(close_prices.columns) == 0:
                return {"error": "No valid ticker data found"}
            
            # Calculate daily returns (without deprecated fill_method)
            daily_returns = close_prices.pct_change(fill_method=None).dropna()
            
            if daily_returns.empty or len(daily_returns) < 10:
                return {"error": f"Insufficient data: only {len(daily_returns)} days found"}
            
            # Calculate weighted portfolio returns
            portfolio_returns = pd.Series(0, index=daily_returns.index)
            
            for yf_t in yf_tickers:
                orig_t = ticker_map.get(yf_t, yf_t)
                w = weights.get(orig_t, 0)
                
                if yf_t in daily_returns.columns:
                    portfolio_returns += daily_returns[yf_t] * w
            
            # Calculate metrics
            cumulative_returns = (1 + portfolio_returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1
            
            # Annualized return
            days_in_period = len(portfolio_returns)
            annualized_return = ((1 + total_return) ** (365 / days_in_period)) - 1
            
            # Volatility (annualized)
            daily_volatility = portfolio_returns.std()
            annualized_volatility = daily_volatility * (252 ** 0.5)  # 252 trading days
            
            # Sharpe Ratio
            excess_return = annualized_return - self.RISK_FREE_RATE
            sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Max Drawdown
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win Rate
            winning_days = (portfolio_returns > 0).sum()
            total_days = len(portfolio_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            # Best/Worst performers
            asset_returns = {}
            for yf_t in yf_tickers:
                if yf_t in close_prices.columns:
                    orig_t = ticker_map.get(yf_t, yf_t)
                    first_price = close_prices[yf_t].dropna().iloc[0] if not close_prices[yf_t].dropna().empty else None
                    last_price = close_prices[yf_t].dropna().iloc[-1] if not close_prices[yf_t].dropna().empty else None
                    if first_price and last_price and first_price > 0:
                        asset_return = ((last_price - first_price) / first_price) * 100
                        if not np.isnan(asset_return):
                            asset_returns[orig_t] = round(float(asset_return), 2)
            
            # Filter valid returns for best/worst
            valid_returns = {k: v for k, v in asset_returns.items() if not np.isnan(v)}
            best = max(valid_returns.items(), key=lambda x: x[1]) if valid_returns else ("N/A", 0)
            worst = min(valid_returns.items(), key=lambda x: x[1]) if valid_returns else ("N/A", 0)
            
            return {
                "total_return_pct": round(float(total_return) * 100, 2),
                "annualized_return_pct": round(float(annualized_return) * 100, 2),
                "sharpe_ratio": round(float(sharpe_ratio), 2),
                "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
                "volatility_pct": round(float(annualized_volatility) * 100, 2),
                "win_rate": round(float(win_rate), 2),
                "best_performer": {"ticker": best[0], "return": best[1]},
                "worst_performer": {"ticker": worst[0], "return": worst[1]},
                "days_analyzed": days_in_period,
                "asset_returns": asset_returns
            }
            
        except Exception as e:
            logger.error(f"Portfolio backtest failed: {e}")
            return {"error": str(e)}
    
    def format_backtest_report(self, results: Dict) -> str:
        """Generate a formatted backtest report for Telegram."""
        if "error" in results:
            return f"❌ Backtest Error: {results['error']}"
        
        sharpe_emoji = "🟢" if results['sharpe_ratio'] >= 1.0 else "🟡" if results['sharpe_ratio'] >= 0.5 else "🔴"
        dd_emoji = "🟢" if results['max_drawdown_pct'] >= -15 else "🟡" if results['max_drawdown_pct'] >= -25 else "🔴"
        
        report = "📊 Portfolio Backtest Report\n"
        report += "━" * 24 + "\n\n"
        
        report += f"📈 Total Return: {results['total_return_pct']:+.1f}%\n"
        report += f"📊 Annualized: {results['annualized_return_pct']:+.1f}%\n"
        report += f"{sharpe_emoji} Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
        report += f"{dd_emoji} Max Drawdown: {results['max_drawdown_pct']:.1f}%\n"
        report += f"📉 Volatility: {results['volatility_pct']:.1f}%\n"
        report += f"🎯 Win Rate: {results['win_rate']:.0%}\n\n"
        
        report += f"🏆 Best: {results['best_performer']['ticker']} ({results['best_performer']['return']:+.1f}%)\n"
        report += f"💔 Worst: {results['worst_performer']['ticker']} ({results['worst_performer']['return']:+.1f}%)\n"
        
        report += "\n" + "━" * 24
        report += f"\nAnalyzed: {results['days_analyzed']} days"
        
        return report


# =============================================================================
# Standalone Execution (GitHub Actions)
# =============================================================================
if __name__ == "__main__":
    import os
    import asyncio
    from db_handler import DBHandler
    from market_data import MarketData
    from telegram_bot import TelegramNotifier
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Portfolio Backtest: Starting standalone execution...")
    
    # 1. Configuration
    target_chat_id = os.environ.get("TARGET_CHAT_ID")
    period = int(os.environ.get("BACKTEST_PERIOD", 180))
    
    if not target_chat_id:
        logger.warning("Target Chat ID not set, running in test mode with mock data")
        # Existing mock test...
        test_portfolio = [
            {"ticker": "BTC-USD", "quantity": 0.1, "avg_price": 40000, "current_price": 42000},
            {"ticker": "ETH-USD", "quantity": 1.0, "avg_price": 2000, "current_price": 2200}
        ]
        bt = PortfolioBacktest()
        results = bt.run_backtest(test_portfolio, period_days=period)
        print(bt.format_backtest_report(results))
    else:
        # 2. Production Execution
        async def main():
            try:
                db = DBHandler()
                md = MarketData()
                bot = TelegramNotifier()
                bt = PortfolioBacktest()
                
                await bot.send_message(target_chat_id, f"⏳ Avvio Backtest Remoto ({period} giorni)...")
                
                # Fetch Portfolio
                portfolio = db.get_portfolio()
                if not portfolio:
                    await bot.send_message(target_chat_id, "❌ Portafoglio vuoto.")
                    return

                # Add current prices
                for item in portfolio:
                    ticker = item.get('ticker', '')
                    price, _ = md.get_smart_price_eur(ticker)
                    item['current_price'] = price
                
                # Run Backtest
                results = bt.run_backtest(portfolio, period_days=period)
                report = bt.format_backtest_report(results)
                
                # Add Correlation (Standalone version needs this too)
                tickers = [item['ticker'] for item in portfolio][:10]
                if len(tickers) >= 2:
                    corr_data = md.calculate_correlation_matrix(tickers)
                    if corr_data.get('high_correlation_pairs'):
                        report += "\n\n⚠️ **Asset Correlati (>70%):**\n"
                        for t1, t2, corr in corr_data['high_correlation_pairs'][:3]:
                            report += f"• {t1} ↔ {t2}: {corr:.0%}\n"
                    report += f"\n🎯 Diversification Score: {corr_data['diversification_score']}/100"
                
                await bot.send_message(target_chat_id, report)
                await bot.send_message(target_chat_id, "✅ Backtest completato (Cloud Execution).")
                
            except Exception as e:
                logger.error(f"Backtest execution failed: {e}")
                if target_chat_id:
                    await bot.send_message(target_chat_id, f"❌ Errore Backtest Remoto: {e}")
        
        asyncio.run(main())
