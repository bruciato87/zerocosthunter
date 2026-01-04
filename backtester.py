
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from ta.momentum import RSIIndicator
from db_handler import DBHandler
from market_data import MarketData

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backtester")

class Backtester:
    def __init__(self, start_balance=10000.0):
        self.db = DBHandler()
        self.market = MarketData()
        self.start_balance = start_balance

    def run_rsi_backtest(self, ticker: str, period_days: int = 90, rsi_buy=30, rsi_sell=70):
        """
        Runs a standard RSI Mean Reversion backtest.
        Buy when RSI < 30. Sell when RSI > 70.
        """
        logger.info(f"🧪 Starting RSI Backtest for {ticker} ({period_days}d)...")
        
        # 1. Fetch History
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 30) # Buffer for MA/RSI calc
        
        # Handle ticker mapping (MarketData may have aliases, but for backtest use raw ticker)
        yf_ticker = ticker
             
        try:
             df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False)
             if df.empty:
                 logger.error(f"❌ No data found for {yf_ticker}")
                 return None
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return None

        # 2. Calculate Indicators
        # Ensure Close is flat series
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
            
        rsi_indicator = RSIIndicator(close=close_series, window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # Trim to requested period
        mask = (df.index >= pd.Timestamp(end_date - timedelta(days=period_days)).tz_localize(df.index.tz))
        sim_data = df.loc[mask].copy()
        
        if sim_data.empty:
            logger.error("❌ Not enough data after trimming.")
            return None
            
        # 3. Simulate
        cash = self.start_balance
        holdings = 0.0
        trades_count = 0
        wins = 0
        
        entry_price = 0.0
        
        for i in range(len(sim_data)):
            # Get row safely
            row = sim_data.iloc[i]
            price = float(row['Close'])
            rsi = float(row['RSI'])
            date = sim_data.index[i]
            
            # Logic
            if rsi < rsi_buy and cash > 0:
                # BUY ALL
                qty = cash / price
                holdings = qty
                cash = 0
                entry_price = price
                # logger.info(f"[{date.date()}] BUY  @ {price:.2f} (RSI: {rsi:.1f})")
                
            elif rsi > rsi_sell and holdings > 0:
                # SELL ALL
                sale_value = holdings * price
                pnl = (price - entry_price) / entry_price * 100
                
                cash = sale_value
                holdings = 0
                trades_count += 1
                if pnl > 0: wins += 1
                # logger.info(f"[{date.date()}] SELL @ {price:.2f} (RSI: {rsi:.1f}) -> PnL: {pnl:.2f}%")
        
        # Final Valuation
        final_price = float(sim_data.iloc[-1]['Close'])
        final_value = cash + (holdings * final_price)
        total_pnl = ((final_value - self.start_balance) / self.start_balance) * 100
        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0.0
        
        result = {
            "ticker": ticker,
            "period_days": period_days,
            "start_date": sim_data.index[0].strftime('%Y-%m-%d'),
            "end_date": sim_data.index[-1].strftime('%Y-%m-%d'),
            "starting_balance": self.start_balance,
            "ending_balance": round(final_value, 2),
            "pnl_percent": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": trades_count,
            "strategy_version": f"RSI_{rsi_buy}_{rsi_sell}"
        }
        
        # Save to DB
        self.db.save_backtest_result(result)
        logger.info(f"✅ Backtest Complete: {ticker} -> {total_pnl}% ({trades_count} trades)")
        return result

if __name__ == "__main__":
    bt = Backtester()
    # Test
    bt.run_rsi_backtest("BTC-USD", 90)
    bt.run_rsi_backtest("NVDA", 90)
