import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We need to test the logic block inside main.py explicitly. 
# Since main.py is a script with a loop, we can extract the risk logic into a function or test it by importing main (hard due to side effects).
# ALTERNATIVE: Test the logic conceptually by replicating it, or verifying market_data.calculate_atr behavior.
# BETTER: Verify market_data.calculate_atr works with mocks.

from market_data import MarketData

class TestRiskManagement(unittest.TestCase):
    
    @patch('market_data.yf.download')
    def test_atr_calculation(self, mock_download):
        # ERROR: yfinance returns a complex DataFrame structure.
        # We'll mock the DataFrame structure to match YF's output (MultiIndex columns usually in recent versions)
        import pandas as pd
        
        # Create a simple DataFrame (mimicking auto_adjust=True)
        # 15 days of data for ATR-14
        data = {
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
            'Low':  [98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        }
        df = pd.DataFrame(data)
        
        # Mock download return
        mock_download.return_value = df
        
        md = MarketData()
        
        # Test calculation
        result = md.calculate_atr("TEST", period=14)
        
        print("\nATR Result:", result)
        
        self.assertIn("atr", result)
        self.assertGreater(result["atr"], 0)
        self.assertIn("suggested_stop", result)
        
        # Verify 2x logic
        # ATR pct is approx (ATR / Price) * 100
        # Suggested stop should be approx 2 * ATR_pct
        self.assertAlmostEqual(result["suggested_stop"], result["atr_pct"] * 2, delta=0.5)

if __name__ == '__main__':
    unittest.main()
