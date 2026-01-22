import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import argparse
from market_data import MarketData
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestEngine")

class BacktestEngine:
    def __init__(self, initial_capital=10000.0, commission=0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.holdings = {} # ticker -> {'qty': 0, 'avg_price': 0}
        self.trade_log = []
        self.market = MarketData()
        
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for the entire dataframe.
        In a real simulation, we'd do this step-by-step, but for vector speed we do it once
        and then iterate while respecting 'lookahead bias' (by only using row 'i').
        """
        if len(df) < 50: return df
        
        # RSI 14
        rsi_ind = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi_ind.rsi()
        
        # SMA 50/200
        sma_50 = SMAIndicator(close=df['Close'], window=50)
        df['SMA_50'] = sma_50.sma_indicator()
        
        sma_200 = SMAIndicator(close=df['Close'], window=200)
        df['SMA_200'] = sma_200.sma_indicator()
        
        return df

    def _generate_signal(self, row, prev_row):
        """
        Technical Proxy Strategy (Brain Simulation).
        Returns: 'BUY', 'SELL', 'HOLD'
        """
        # DEBUG (Sample first ticker occasionally)
        if hasattr(row, 'ticker') and row['ticker'] == 'BTC-USD':
             rsi_val = row.get('RSI', np.nan)
             sma200_val = row.get('SMA_200', np.nan)
             price_val = row.get('Close', np.nan)
             print(f"DEBUG: {row.name} Price:{price_val:.0f} SMA200:{sma200_val:.0f} RSI:{rsi_val:.1f}")

        if pd.isna(row['RSI']) or pd.isna(row['SMA_50']) or pd.isna(row['SMA_200']):
            return 'HOLD', 0.0, ""
            
        rsi = row['RSI']
        price = row['Close']
        sma50 = row['SMA_50']
        sma200 = row['SMA_200']
        
        signal = 'HOLD'
        confidence = 0.0
        reason = ""

        # --- CRITIC AGENT LOGIC (Trend Filter) ---
        # "Don't catch falling knives" -> Block BUYs if Price < SMA200
        is_uptrend = price > sma200
        
        # DEBUG (Sample first ticker occasionally)
        if row['ticker'] == 'BTC-USD':
             # Always print daily status for BTC to debug loop
             print(f"DEBUG: {date} Price:{price:.0f} SMA50:{sma50:.0f} SMA200:{sma200:.0f} RSI:{rsi:.1f} Up:{is_uptrend}")

        # BUY LOGIC
        if rsi < 30:
            if is_uptrend:
                # Golden Opportunity: Oversold + Uptrend
                signal = 'BUY'
                confidence = 0.90
                reason = "Oversold (RSI < 30) in Bull Trend (> SMA200)"
            else:
                # Bearish Regime: CRITIC BLOCKS THIS!
                # Only buy if deep oversold (< 25) for a bounce
                if rsi < 25:
                     signal = 'BUY'
                     confidence = 0.65 # Low confidence bounce play
                     reason = "Deep Oversold (RSI < 25) Bounce Play"
                else:
                     signal = 'HOLD' # Block standard oversold
                     reason = "Critic: Blocked BUY (Price < SMA200)"
        
        # SELL LOGIC
        elif rsi > 70:
            signal = 'SELL'
            confidence = 0.85
            reason = "Overbought (RSI > 70)"
        
        # SMA50 Breakout (Recovery Play)
        elif prev_row is not None and prev_row['Close'] < prev_row['SMA_50'] and price > sma50:
             # Only if not too far from SMA200 (don't buy top of bear rally?)
             # Relaxed for test
             signal = 'BUY'
             confidence = 0.70
             reason = "Price Breakout above SMA50"
            
        return signal, confidence, reason

    def execute_trade(self, date, ticker, action, price, confidence):
        """Execute Buy/Sell orders updating cash and holdings."""
        if action == 'HOLD': return
        
        if action == 'BUY':
            if self.cash < 100: return # Min cash
            
            # Allocation strategy: 20% of current cash per trade
            allocation = self.cash * 0.20
            qty = allocation / price
            cost = qty * price
            fee = cost * self.commission
            
            self.cash -= (cost + fee)
            
            # Update Holdings
            if ticker not in self.holdings:
                self.holdings[ticker] = {'qty': 0.0, 'cost_basis': 0.0}
            
            # Weighted average price
            old_qty = self.holdings[ticker]['qty']
            old_cost = self.holdings[ticker]['qty'] * self.holdings[ticker]['cost_basis']
            new_qty = old_qty + qty
            new_avg = (old_cost + cost) / new_qty
            
            self.holdings[ticker]['qty'] = new_qty
            self.holdings[ticker]['cost_basis'] = new_avg
            
            self.trade_log.append({
                'date': date, 'ticker': ticker, 'action': 'BUY', 
                'qty': qty, 'price': price, 'fee': fee, 'reason': f"Conf: {confidence:.2f}"
            })
            
        elif action == 'SELL':
            if ticker not in self.holdings or self.holdings[ticker]['qty'] <= 0: return
            
            qty = self.holdings[ticker]['qty']
            revenue = qty * price
            fee = revenue * self.commission
            
            # PnL
            cost_basis = self.holdings[ticker]['cost_basis']
            profit = (price - cost_basis) * qty
            
            self.cash += (revenue - fee)
            self.holdings[ticker]['qty'] = 0 # Full sell behavior for now
            
            self.trade_log.append({
                'date': date, 'ticker': ticker, 'action': 'SELL', 
                'qty': qty, 'price': price, 'fee': fee, 'pnl': profit, 'reason': f"Conf: {confidence:.2f}"
            })

    def check_sl_tp(self, date, ticker, row):
        """Check if intra-day Low/High hit SL or TP."""
        if ticker not in self.holdings or self.holdings[ticker]['qty'] <= 0: return

        pos = self.holdings[ticker]
        entry_price = pos['cost_basis']
        
        # Strategy Parameters
        SL_PCT = 0.05 # 5% Stop Loss
        TP_PCT = 0.15 # 15% Take Profit
        
        sl_price = entry_price * (1 - SL_PCT)
        tp_price = entry_price * (1 + TP_PCT)
        
        action = None
        price = row['Close']
        
        # Check Low for SL
        if row['Low'] <= sl_price:
            action = "STOP_LOSS"
            price = sl_price # Assume filled at SL
            
        # Check High for TP (Priority? Usually SL hit checks first for safety)
        elif row['High'] >= tp_price:
            action = "TAKE_PROFIT"
            price = tp_price
            
        if action:
            self.execute_trade(date, ticker, 'SELL', price, 1.0)
            # Update log reason
            self.trade_log[-1]['reason'] = f"{action} hit ({price:.2f})"

    def run_simulation(self, tickers, days=90):
        """Run the backtest loop."""
        logger.info(f"Starting Backtest for {tickers} over last {days} days (Enhanced Logic)...")
        
        start_capital = self.initial_capital
        
        # 1. Fetch Data
        merged_data = pd.DataFrame()
        
        for t in tickers:
            df = self.market.get_historical_data(t, days=days + 300) # Buffer for SMA200 (need >200 bars)
            if df.empty: continue
            
            df = self._calculate_indicators(df)
            
            # Trim to requested days
            start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df.index >= start_date]
            
            df['ticker'] = t
            merged_data = pd.concat([merged_data, df])
            
        if merged_data.empty:
            logger.error("No data fetched.")
            return

        merged_data.sort_index(inplace=True)
        
        logger.info(f"Simulating {len(merged_data)} candles...")
        
        prev_rows = {} 
        
        for date, row in merged_data.iterrows():
            ticker = row['ticker']
            
            # 1. RISK CHECK FIRST (Simulate intra-day hit)
            self.check_sl_tp(date, ticker, row)
            
            # 2. Generate Signal
            try:
                prev = prev_rows.get(ticker)
                signal, conf, reason = self._generate_signal(row, prev)
                
                if signal != 'HOLD':
                    pass
                    # logger.info(f"Signal: {ticker} {signal} {conf}")
                
                # Filter low confidence (LOWER threshold to 0.60 to be safe)
                if conf >= 0.60:
                    self.execute_trade(date, ticker, signal, row['Close'], conf)
            except Exception as e:
                logger.error(f"Error gen signal: {e}")
                pass
            
            prev_rows[ticker] = row

        # 3. Calculate Final Value
        final_value = self.cash
        equity = final_value
        
        print("\n" + "="*40)
        print(f"🏁 BACKTEST REPORT ({days} days)")
        print("="*40)
        
        # Liquidate remaining positions at last price
        for t, h in self.holdings.items():
            if h['qty'] > 0:
                # Find last price
                last_price = merged_data[merged_data['ticker'] == t].iloc[-1]['Close']
                val = h['qty'] * last_price
                equity += val
                print(f"Open Position: {t} | {h['qty']:.4f} units @ {last_price:.2f} (Val: €{val:.2f})")
        
        ret_pct = ((equity - start_capital) / start_capital) * 100
        
        print("-" * 40)
        print(f"Initial Capital: €{start_capital:,.2f}")
        print(f"Final Equity:    €{equity:,.2f}")
        print(f"Total Return:    {ret_pct:+.2f}%")
        print(f"Total Trades:    {len(self.trade_log)}")
        print("-" * 40)
        
        # Trade Log
        if self.trade_log:
             print("\nTrade History:")
             for t in self.trade_log:
                 pnl_str = f" | PnL: €{t.get('pnl', 0):+.2f}" if 'pnl' in t else ""
                 print(f"{t['date'].strftime('%Y-%m-%d')} {t['action']} {t['ticker']} @ {t['price']:.2f}{pnl_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*", default=["BTC-USD", "ETH-USD", "SOL-USD"], help="Tickers to test")
    parser.add_argument("--days", type=int, default=90, help="Days to look back")
    args = parser.parse_args()
    
    engine = BacktestEngine()
    engine.run_simulation(args.tickers, days=args.days)
