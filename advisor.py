
import logging
import yfinance as yf

logger = logging.getLogger("Advisor")

class Advisor:
    def __init__(self):
        # Cache for sector lookups to avoid slow yfinance calls every time
        # In a real app, this should be in DB (assets table)
        self.sector_cache = {
            "BTC-USD": "Crypto",
            "ETH-USD": "Crypto",
            "SOL-USD": "Crypto",
            "USDT-USD": "Crypto",
            "SPY": "ETF",
            "QQQ": "ETF",
            "NVDA": "Technology",
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "TSLA": "Consumer Cyclical",
            "AMZN": "Consumer Cyclical",
            "XOM": "Energy",
            "CVX": "Energy"
        }

    def get_sector(self, ticker):
        """
        Returns the sector of the asset.
        """
        # 1. Check Cache
        clean_ticker = ticker.upper().replace("-USD", "")
        # Check known crypto variations
        if "-USD" in ticker.upper() or ticker.upper() in ["BTC", "ETH", "SOL"]:
            return "Crypto"
        
        if ticker.upper() in self.sector_cache:
            return self.sector_cache[ticker.upper()]
        
        # 2. Fetch from Yahoo (Slow)
        try:
            t = yf.Ticker(ticker)
            info = t.info
            sector = info.get('sector', 'Unknown')
            self.sector_cache[ticker.upper()] = sector
            return sector
        except Exception as e:
            logger.warning(f"Advisor: Could not fetch sector for {ticker}: {e}")
            return "Unknown"

    def analyze_portfolio(self, portfolio_items):
        """
        Analyzes the portfolio and returns sector exposure.
        portfolio_items: List of dicts from DB (must have 'ticker', 'quantity', 'current_price').
        """
        exposure = {}
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
            
            if price:
                value = price * qty
                total_value += value

                sector = self.get_sector(ticker)
                exposure[sector] = exposure.get(sector, 0.0) + value

        # specific check for overly dominating sectors
        pct_exposure = {}
        if total_value > 0:
            for sec, val in exposure.items():
                pct_exposure[sec] = round((val / total_value) * 100, 1)

        return {
            "total_value": total_value,
            "sector_value": exposure,
            "sector_percent": pct_exposure
        }

    def generate_tips(self, analysis):
        """
        Generates risk management tips based on analysis.
        """
        tips = []
        pcts = analysis['sector_percent']
        
        # Rule 1: Tech Overload
        if pcts.get('Technology', 0) > 40:
            tips.append(f"High Tech Exposure ({pcts['Technology']}%). Consider diversifying into Energy or Healthcare.")
        
        # Rule 2: Crypto Volatility
        if pcts.get('Crypto', 0) > 30:
            tips.append(f"High Crypto Exposure ({pcts['Crypto']}%). Ensure you are comfortable with high volatility.")

        # Rule 3: Single Asset Risk (Not implemented yet, needs per-asset calc)
        
        return tips

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
