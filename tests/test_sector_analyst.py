
import unittest
from unittest.mock import MagicMock
import pandas as pd
from market_data import SectorAnalyst

class TestSectorAnalyst(unittest.TestCase):
    def setUp(self):
        self.mock_market = MagicMock()
        self.sector_analyst = SectorAnalyst(self.mock_market)

    def test_get_sector_ranking(self):
        # Mock historical data for a few sectors
        # XLK (Tech) -> uptrend (Leader)
        # XLE (Energy) -> downtrend (Laggard)
        # XLF (Financials) -> flat (Middle)
        
        def mock_get_history(ticker, days=200, force_refresh=False):
            dates = pd.date_range(end=pd.Timestamp.now(), periods=130)
            if ticker == "XLK":
                # Leader: Strong uptrend
                prices = [100 + i for i in range(130)] # Starts 100, ends 229
            elif ticker == "XLE":
                # Laggard: Strong downtrend
                prices = [200 - i for i in range(130)] # Starts 200, ends 71
            else:
                # Flat/Middle
                prices = [100] * 130
                
            df = pd.DataFrame({'Close': prices, 'Date': dates})
            df.set_index('Date', inplace=True)
            return df
            
        self.mock_market.get_historical_data.side_effect = mock_get_history
        
        # Only test a subset of tickers to speed up mocking logic (we mocked behavior by ticker name)
        # We need to ensure we don't crash on unmocked tickers.
        # But get_sector_ranking iterates over ALL sectors defined in SECTOR_ETFS.
        # Our mock handles "else" case so it's fine.
        
        ranking = self.sector_analyst.get_sector_ranking(limit=3)
        
        # XLK should be top because it rose ~129%
        # XLE should be bottom because it fell ~65%
        
        top_sector = ranking[0]
        self.assertEqual(top_sector['ticker'], 'XLK')
        self.assertTrue(top_sector['momentum_score'] > 0)
        
        # Find XLE in ranking
        xle_entry = next((x for x in ranking if x['ticker'] == 'XLE'), None)
        # Note: ranking is sorted descending. Since we mocked ALL sectors (via else), XLE should be last.
        # But limit=3 returns only top 3. So XLE might NOT be in result if we have many "else" sectors with 0 momentum?
        # Wait, "else" is flat (0% change). XLE is negative (-65%).
        # So XLE will definitely be lower than "else" (0%).
        # XLK > 0.
        # So ranking order: XLK (pos), Others (0), XLE (neg).
        
        # If we request full list (limit=100), XLE should be last.
        full_ranking = self.sector_analyst.get_sector_ranking(limit=100)
        self.assertEqual(full_ranking[0]['ticker'], 'XLK')
        self.assertEqual(full_ranking[-1]['ticker'], 'XLE')
        
    def test_rotation_signals(self):
        # Setup similar to above but verify string output
        
        def mock_get_history(ticker, days=200, force_refresh=False):
            dates = pd.date_range(end=pd.Timestamp.now(), periods=130)
            if ticker == "XLK":
                prices = [100 + i for i in range(130)] # +129%
            elif ticker == "XLE":
                prices = [200 - i for i in range(130)] # -65%
            else:
                prices = [100] * 130
            df = pd.DataFrame({'Close': prices, 'Date': dates})
            df.set_index('Date', inplace=True)
            return df
            
        self.mock_market.get_historical_data.side_effect = mock_get_history
        
        signals = self.sector_analyst.get_rotation_signals()
        
        self.assertTrue(len(signals) > 0)
        # Should recommend accumulating Top (Technology)
        self.assertTrue(any("Technology" in s for s in signals))
        # Should recommend avoiding Bottom (Energy)
        self.assertTrue(any("Energy" in s for s in signals))
        # Should suggest rotation (Gap > 10%)
        self.assertTrue(any("Rotation Idea" in s for s in signals))

if __name__ == '__main__':
    unittest.main()
