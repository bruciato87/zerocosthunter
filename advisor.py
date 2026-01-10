
import logging
import yfinance as yf

logger = logging.getLogger("Advisor")

class Advisor:
    def __init__(self, market_instance=None):
        # Cache for sector lookups to avoid slow yfinance calls every time
        # In a real app, this should be in DB (assets table)
        self.sector_cache = {
            "BTC-USD": "Crypto",
            "BTC": "Crypto",
            "ETH-USD": "Crypto",
            "ETH": "Crypto",
            "SOL-USD": "Crypto",
            "SOL": "Crypto",           # Solana (no suffix)
            "USDT-USD": "Crypto",
            "XRP": "Crypto",           # Ripple
            "XRP-USD": "Crypto",
            "RNDR": "Crypto",          # Render Token
            "RNDR-USD": "Crypto",
            "RENDER": "Crypto",
            "RENDER-USD": "Crypto",
            # ETFs
            "SPY": "ETF",
            "QQQ": "ETF",
            "EUNL.DE": "ETF",           # iShares Core MSCI World
            "RBOT.MI": "ETF",           # Automation & Robotics
            "AIAI.MI": "ETF",           # Artificial Intelligence ETF
            "NUKL": "ETF",              # Global X Uranium ETF
            "NUKL.DE": "ETF",
            "ICGA.FRA": "ETF",
            # Technology
            "NVDA": "Technology",
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "META": "Technology",
            "AMD": "Technology",
            "3CP": "Technology",        # Xiaomi (Frankfurt)
            "3XC": "Technology",        # Xiaomi (Frankfurt alt)
            "TCEHY": "Technology",      # Tencent ADR
            "TCT": "Technology",        # Tencent (Frankfurt)
            "NNnD.F": "Technology",     # Tencent Frankfurt
            # Consumer Cyclical
            "TSLA": "Consumer Cyclical",
            "AMZN": "Consumer Cyclical",
            "BYD": "Consumer Cyclical", # BYD Co
            "BYDDF": "Consumer Cyclical", # BYD OTC
            # Energy
            "XOM": "Energy",
            "CVX": "Energy",
        }
        from market_data import MarketData
        self.market = market_instance if market_instance else MarketData()

    def get_sector(self, ticker):
        """
        Returns the sector of the asset.
        """
        # 1. Crypto detection via suffix
        ticker_upper = ticker.upper()
        if "-USD" in ticker_upper or "-EUR" in ticker_upper:
            return "Crypto"
        
        # 2. Direct cache lookup
        if ticker_upper in self.sector_cache:
            return self.sector_cache[ticker_upper]
        
        # 3. Clean ticker lookup (strip suffix)
        clean_ticker = ticker_upper.replace("-USD", "").replace("-EUR", "")
        if clean_ticker in self.sector_cache:
            return self.sector_cache[clean_ticker]
        
        # 2. Fetch from Yahoo (Slow) - Use Alias from MarketData
        try:
            # Resolve alias using MarketData
            search_ticker = self.market.TICKER_ALIASES.get(ticker, ticker)
            
            t = yf.Ticker(search_ticker)
            info = t.info
            sector = info.get('sector', 'Unknown')
            self.sector_cache[ticker.upper()] = sector
            return sector
        except Exception as e:
            logger.warning(f"Advisor: Could not fetch sector for {ticker}: {e}")
            return "Unknown"

    def analyze_portfolio(self, portfolio_items):
        """
        Analyzes the portfolio and returns sector exposure + advanced metrics.
        portfolio_items: List of dicts from DB.
        """
        exposure = {}
        asset_performance = [] # New list for rebalancing logic
        total_value = 0.0
        
        # Helper to fetch price if missing (lazy load)
        from market_data import MarketData
        market = MarketData()

        for item in portfolio_items:
            ticker = item['ticker']
            qty = float(item.get('quantity', 0))
            
            # Use DB price if available, else fetch live
            price = float(item.get('current_price', 0) or 0)
            if price == 0:
                try:
                    price = market.get_market_price(ticker)
                    if price:
                        logger.info(f"Advisor: Fetched live price for {ticker}: ${price}")
                except Exception as e:
                    logger.warning(f"Advisor: Failed to fetch price for {ticker}: {e}")
            
            # Avg Price for PnL
            avg_price = float(item.get('avg_price', 0) or 0)
            
            if price:
                value = price * qty
                total_value += value

                sector = self.get_sector(ticker)
                exposure[sector] = exposure.get(sector, 0.0) + value
                
                # Check Performance
                pnl_pct = 0.0
                if avg_price > 0:
                    pnl_pct = ((price - avg_price) / avg_price) * 100
                
                asset_performance.append({
                    "ticker": ticker,
                    "value": value,
                    "pnl_pct": pnl_pct,
                    "sector": sector,
                    "avg_price": avg_price, # Include avg_price for tax harvesting logic
                    "quantity": qty
                })

        pct_exposure = {k: (v / total_value) * 100 for k, v in exposure.items()} if total_value > 0 else {}
        
        # Generate Tips internally or return data for generation
        tips = []

        if total_value == 0:
            return {
                "total_value": 0.0,
                "sector_value": {},
                "sector_percent": {},
                "tips": ["Portfolio is empty. Start accumulating assets."]
            }
        
        # 1. Sector Logic
        for sec, pct in pct_exposure.items():
            if pct > 40: tips.append(f"⚠️ High {sec} Exposure ({pct:.1f}%). Consider Diversifying.")
            if pct < 5 and sec != 'Cash': # Assuming 'Cash' might be a sector not needing diversification
                tips.append(f"Low exposure to {sec} ({pct:.1f}%). Look for opportunities.")
        
        # 3. Tax Harvesting & Profit Taking (Definitive Italy 2025/2026 Rules)
        # Based on 'Legge di Bilancio 2025' (Approved Dec 2024).
        
        for asset in asset_performance:
            asset_pct = (asset['value'] / total_value) * 100
            
            # Concentration Check
            if asset_pct > 20:
                tips.append(f"⚠️ Concentration Risk: {asset['ticker']} is {asset_pct:.1f}% of Portfolio.")
                
            # Tax Harvesting (IT Specifics)
            if asset['pnl_pct'] < -40:
                loss_val = asset['value'] - (asset['avg_price'] * asset['quantity'])
                
                if asset['sector'] == 'Crypto':
                    tips.append(f"📉 Tax Harvest (Crypto): {asset['ticker']} down {asset['pnl_pct']:.1f}%. Realize loss before 2026 (Tax rises to 33%).")
                elif asset['sector'] == 'ETF':
                     tips.append(f"📉 Tax Harvest (ETF): {asset['ticker']} down {asset['pnl_pct']:.1f}%. WARNING: Losses are SILOED until Aug 2026 (Unification pending).")
                else:
                     tips.append(f"📉 Tax Harvest: {asset['ticker']} down {asset['pnl_pct']:.1f}%. Sell to generate 'Minusvalenza' (4 years).")
            
            # Profit Taking
            if asset['pnl_pct'] > 50:
                 tips.append(f"💰 Take Profit: {asset['ticker']} is up {asset['pnl_pct']:.1f}%. Consider selling 20% to rebalance.")

        return {
            "total_value": total_value,
            "sector_value": exposure,
            "sector_percent": pct_exposure,
            "tips": tips,
            "note": "Tax tips based on definitive Legge di Bilancio 2025 (Crypto 26%->33%, ETF Siloed). Consult a commercialista."
        }

    def generate_tips(self, analysis):
        """
        Wrapper to return tips generated during analysis.
        """
        return analysis.get('tips', [])

if __name__ == "__main__":
    # Mock data
    mock_portfolio = [
        {"ticker": "NVDA", "quantity": 10, "current_price": 120},
        {"ticker": "BTC-USD", "quantity": 0.5, "current_price": 90000},
        {"ticker": "XOM", "quantity": 50, "current_price": 110}
    ]
    advisor = Advisor()
    analysis = advisor.analyze_portfolio(mock_portfolio)
    print("Analysis:", analysis)
    print("Tips:", advisor.generate_tips(analysis))
