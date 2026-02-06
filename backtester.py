
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from db_handler import DBHandler
from market_data import MarketData
from ticker_resolver import resolve_ticker

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Backtester")

class Backtester:
    def __init__(self, start_balance=10000.0):
        self.db = DBHandler()
        self.market = MarketData()
        self.start_balance = start_balance

    def _fetch_data(self, ticker: str, period_days: int):
        """Helper to fetch and prepare historical data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 50)  # Buffer for indicator calc
        
        # Use centralized ticker_resolver for self-learning cache
        yf_ticker = resolve_ticker(ticker)
        logger.info(f"📊 Fetching data for {ticker} (resolved to: {yf_ticker})")
        
        try:
            df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if df.empty:
                logger.error(f"❌ No data found for {yf_ticker}")
                return None
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return None
        
        # Fix yfinance MultiIndex columns (returns MultiIndex with ticker as second level)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Ensure Close is truly 1D
        df['Close'] = df['Close'].values.flatten()
        
        # Trim to requested period
        mask = (df.index >= pd.Timestamp(end_date - timedelta(days=period_days)).tz_localize(df.index.tz))
        return df.loc[mask].copy()

    def _safe_float(self, val):
        """Safely convert pandas series element to float."""
        if hasattr(val, 'iloc'):
            return float(val.iloc[0])
        return float(val)

    def _save_result(self, ticker, period_days, sim_data, cash, holdings, trades_count, wins, strategy_version):
        """Calculate final metrics and save to DB."""
        final_price = self._safe_float(sim_data.iloc[-1]['Close'])
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
            "strategy_version": strategy_version
        }
        
        self.db.save_backtest_result(result)
        logger.info(f"✅ Backtest Complete: {ticker} [{strategy_version}] -> {total_pnl:.2f}% ({trades_count} trades)")
        return result

    def run_rsi_backtest(self, ticker: str, period_days: int = 90, rsi_buy=30, rsi_sell=70):
        """RSI Mean Reversion: Buy when RSI < 30, Sell when RSI > 70."""
        logger.info(f"🧪 Starting RSI Backtest for {ticker} ({period_days}d)...")
        
        df = self._fetch_data(ticker, period_days)
        if df is None or df.empty:
            return None
        
        # Calculate RSI
        rsi_indicator = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # Simulate
        cash, holdings, trades_count, wins, entry_price = self.start_balance, 0.0, 0, 0, 0.0
        
        for i in range(len(df)):
            price = self._safe_float(df.iloc[i]['Close'])
            rsi = self._safe_float(df.iloc[i]['RSI'])
            
            if rsi < rsi_buy and cash > 0:
                holdings = cash / price
                cash = 0
                entry_price = price
            elif rsi > rsi_sell and holdings > 0:
                pnl = (price - entry_price) / entry_price * 100
                cash = holdings * price
                holdings = 0
                trades_count += 1
                if pnl > 0: wins += 1
        
        return self._save_result(ticker, period_days, df, cash, holdings, trades_count, wins, f"RSI_{rsi_buy}_{rsi_sell}")

    def run_macd_backtest(self, ticker: str, period_days: int = 90):
        """MACD Crossover: Buy when MACD crosses above Signal, Sell when crosses below."""
        logger.info(f"🧪 Starting MACD Backtest for {ticker} ({period_days}d)...")
        
        df = self._fetch_data(ticker, period_days)
        if df is None or df.empty:
            return None
        
        # Calculate MACD
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Simulate
        cash, holdings, trades_count, wins, entry_price = self.start_balance, 0.0, 0, 0, 0.0
        prev_hist = 0.0
        
        for i in range(1, len(df)):
            price = self._safe_float(df.iloc[i]['Close'])
            hist = self._safe_float(df.iloc[i]['MACD_Hist'])
            prev = self._safe_float(df.iloc[i-1]['MACD_Hist'])
            
            # Buy: MACD histogram crosses from negative to positive
            if prev < 0 and hist > 0 and cash > 0:
                holdings = cash / price
                cash = 0
                entry_price = price
            # Sell: MACD histogram crosses from positive to negative
            elif prev > 0 and hist < 0 and holdings > 0:
                pnl = (price - entry_price) / entry_price * 100
                cash = holdings * price
                holdings = 0
                trades_count += 1
                if pnl > 0: wins += 1
        
        return self._save_result(ticker, period_days, df, cash, holdings, trades_count, wins, "MACD_12_26_9")

    def run_bollinger_backtest(self, ticker: str, period_days: int = 90, window=20, std_dev=2):
        """Bollinger Bands Mean Reversion: Buy at lower band, Sell at upper band."""
        logger.info(f"🧪 Starting Bollinger Bands Backtest for {ticker} ({period_days}d)...")
        
        df = self._fetch_data(ticker, period_days)
        if df is None or df.empty:
            return None
        
        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=window, window_dev=std_dev)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        
        # Simulate
        cash, holdings, trades_count, wins, entry_price = self.start_balance, 0.0, 0, 0, 0.0
        
        for i in range(len(df)):
            price = self._safe_float(df.iloc[i]['Close'])
            lower = self._safe_float(df.iloc[i]['BB_Lower'])
            upper = self._safe_float(df.iloc[i]['BB_Upper'])
            
            # Buy: Price touches lower band
            if price <= lower and cash > 0:
                holdings = cash / price
                cash = 0
                entry_price = price
            # Sell: Price touches upper band
            elif price >= upper and holdings > 0:
                pnl = (price - entry_price) / entry_price * 100
                cash = holdings * price
                holdings = 0
                trades_count += 1
                if pnl > 0: wins += 1
        
        return self._save_result(ticker, period_days, df, cash, holdings, trades_count, wins, f"BB_{window}_{std_dev}")

    def run_ema_crossover_backtest(self, ticker: str, period_days: int = 90, fast=9, slow=21):
        """EMA Crossover: Buy when fast EMA crosses above slow, Sell when crosses below."""
        logger.info(f"🧪 Starting EMA Crossover Backtest for {ticker} ({period_days}d)...")
        
        df = self._fetch_data(ticker, period_days)
        if df is None or df.empty:
            return None
        
        # Calculate EMAs
        df['EMA_Fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
        
        # Simulate
        cash, holdings, trades_count, wins, entry_price = self.start_balance, 0.0, 0, 0, 0.0
        
        for i in range(1, len(df)):
            price = self._safe_float(df.iloc[i]['Close'])
            fast_ema = self._safe_float(df.iloc[i]['EMA_Fast'])
            slow_ema = self._safe_float(df.iloc[i]['EMA_Slow'])
            prev_fast = self._safe_float(df.iloc[i-1]['EMA_Fast'])
            prev_slow = self._safe_float(df.iloc[i-1]['EMA_Slow'])
            
            # Golden Cross: Fast EMA crosses above Slow EMA
            if prev_fast <= prev_slow and fast_ema > slow_ema and cash > 0:
                holdings = cash / price
                cash = 0
                entry_price = price
            # Death Cross: Fast EMA crosses below Slow EMA
            elif prev_fast >= prev_slow and fast_ema < slow_ema and holdings > 0:
                pnl = (price - entry_price) / entry_price * 100
                cash = holdings * price
                holdings = 0
                trades_count += 1
                if pnl > 0: wins += 1
        
        return self._save_result(ticker, period_days, df, cash, holdings, trades_count, wins, f"EMA_{fast}_{slow}")

    def run_rsi_macd_confluence_backtest(self, ticker: str, period_days: int = 90, rsi_threshold=40):
        """
        RSI + MACD Confluence: Buy ONLY when BOTH conditions are met.
        - RSI < 40 (oversold-ish) AND MACD histogram crosses positive.
        Historically proven to reduce false positives.
        """
        logger.info(f"🧪 Starting RSI+MACD Confluence Backtest for {ticker} ({period_days}d)...")
        
        df = self._fetch_data(ticker, period_days)
        if df is None or df.empty:
            return None
        
        # Calculate RSI
        rsi_indicator = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # Calculate MACD
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD_Hist'] = macd.macd_diff()
        
        # Simulate
        cash, holdings, trades_count, wins, entry_price = self.start_balance, 0.0, 0, 0, 0.0
        
        for i in range(1, len(df)):
            price = self._safe_float(df.iloc[i]['Close'])
            rsi = self._safe_float(df.iloc[i]['RSI'])
            hist = self._safe_float(df.iloc[i]['MACD_Hist'])
            prev_hist = self._safe_float(df.iloc[i-1]['MACD_Hist'])
            
            # BUY: RSI is in oversold territory AND MACD confirms upward momentum
            if rsi < rsi_threshold and prev_hist < 0 and hist > 0 and cash > 0:
                holdings = cash / price
                cash = 0
                entry_price = price
            # SELL: RSI > 60 (overbought-ish) AND MACD confirms downward momentum
            elif rsi > 60 and prev_hist > 0 and hist < 0 and holdings > 0:
                pnl = (price - entry_price) / entry_price * 100
                cash = holdings * price
                holdings = 0
                trades_count += 1
                if pnl > 0: wins += 1
        
        return self._save_result(ticker, period_days, df, cash, holdings, trades_count, wins, f"RSI_MACD_Confluence")

    def run_best_strategy(self, ticker: str, period_days: int = 90):
        """
        Run ONLY the historically best performing strategy for this asset type.
        - Crypto (contains USD/EUR or SOL/BTC/ETH/XRP): RSI+MACD Confluence
        - Stocks/ETFs: Bollinger Bands
        """
        crypto_keywords = ["USD", "EUR", "BTC", "ETH", "SOL", "XRP", "RENDER"]
        is_crypto = any(kw in ticker.upper() for kw in crypto_keywords)
        
        if is_crypto:
            logger.info(f"🎯 {ticker} identified as CRYPTO -> Using RSI+MACD Confluence")
            return self.run_rsi_macd_confluence_backtest(ticker, period_days)
        else:
            logger.info(f"🎯 {ticker} identified as STOCK/ETF -> Using Bollinger Bands")
            return self.run_bollinger_backtest(ticker, period_days)

    def run_all_strategies(self, ticker: str, period_days: int = 90):
        """Run all available strategies for a ticker (for comparison/research only)."""
        logger.info(f"🚀 Running ALL strategies for {ticker}...")
        results = []
        results.append(self.run_rsi_backtest(ticker, period_days))
        results.append(self.run_macd_backtest(ticker, period_days))
        results.append(self.run_bollinger_backtest(ticker, period_days))
        results.append(self.run_ema_crossover_backtest(ticker, period_days))
        results.append(self.run_rsi_macd_confluence_backtest(ticker, period_days))
        return [r for r in results if r is not None]

if __name__ == "__main__":
    bt = Backtester()
    # Test best strategy on portfolio
    bt.run_best_strategy("BTC-USD", 90)
    bt.run_best_strategy("NVDA", 90)

