
import unittest
from advisor import Advisor

class TestTaxHarvesting(unittest.TestCase):
    def setUp(self):
        self.advisor = Advisor()
        
    def test_crypto_harvest(self):
        # Mock portfolio item: Crypto with -30% loss
        items = [{
            "ticker": "BTC-USD",
            "quantity": 1.0,
            "current_price": 70000,
            "avg_price": 100000, # Loss of 30,000
            "sector": "Crypto"
        }]
        
        # Analyze
        # note: analyze_portfolio calls get_sector internally which might fail if we don't mock it or providing sector in items is not enough.
        # Actually analyze_portfolio recalculates sector. Let's rely on advisor.get_sector resolving "BTC-USD" to "Crypto" (which it does via cache/suffix).
        
        # We need to mock market data or ensure we provide current_price in the item so it doesn't try to fetch live.
        # analyze_portfolio uses item['current_price'] if available.
        
        result = self.advisor.analyze_portfolio(items)
        tips = result['tips']
        harvest = result['harvest_opportunities']
        
        self.assertTrue(any("Tax Harvest (Crypto)" in tip for tip in tips))
        self.assertEqual(len(harvest), 1)
        self.assertEqual(harvest[0]['ticker'], 'BTC-USD')
        self.assertAlmostEqual(harvest[0]['pnl_pct'], -30.0)

    def test_etf_harvest(self):
        # Mock portfolio item: ETF with -25% loss
        items = [{
            "ticker": "EUNL.DE",
            "quantity": 10.0,
            "current_price": 75.0,
            "avg_price": 100.0,
            "sector": "ETF"
        }]
        
        result = self.advisor.analyze_portfolio(items)
        tips = result['tips']
        
        self.assertTrue(any("Tax Harvest (ETF)" in tip for tip in tips))

    def test_stock_harvest(self):
        # Mock portfolio item: Stock with -25% loss
        items = [{
            "ticker": "TSLA",
            "quantity": 10.0,
            "current_price": 150.0,
            "avg_price": 200.0,
            "sector": "Consumer Cyclical" 
        }]
        
        result = self.advisor.analyze_portfolio(items)
        tips = result['tips']
        
        self.assertTrue(any("Tax Harvest" in tip and "Minusvalenza" in tip for tip in tips))

    def test_small_loss_ignored(self):
        # Mock portfolio item: Small loss (-5%)
        items = [{
            "ticker": "AAPL",
            "quantity": 10.0,
            "current_price": 190.0,
            "avg_price": 200.0, # -5%
            "sector": "Technology"
        }]
        
        result = self.advisor.analyze_portfolio(items)
        tips = result['tips']
        harvest = result['harvest_opportunities']
        
        # Should NOT trigger harvest tip
        self.assertFalse(any("Tax Harvest" in tip for tip in tips))
        self.assertEqual(len(harvest), 0)

if __name__ == '__main__':
    unittest.main()
