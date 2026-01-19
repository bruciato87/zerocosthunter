import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_predictor import MLPredictor

class TestQuantFeatures(unittest.TestCase):
    @patch('db_handler.DBHandler')
    def test_predict_saves_context(self, MockDB):
        # Setup Mock
        mock_db_instance = MockDB.return_value
        mock_table = mock_db_instance.supabase.table.return_value
        mock_insert = mock_table.insert.return_value
        mock_insert.execute.return_value = None # Mock execution
        
        print("Initializing MLPredictor with Mock DB...")
        
        # Init MLPredictor (mock _load_model to prevent DB call during init)
        with patch.object(MLPredictor, '_load_model'): 
            ml = MLPredictor()
            
        # Mock _get_features to avoid Yahoo Finance calls and ensure valid features
        mock_features = {'rsi_14': 50.0, 'market_regime': 0}
        with patch.object(ml, '_get_features', return_value=mock_features):
            
            print("Calling predict with Quant features...")
            # Call predict with new args
            ml.predict("BTC-USD", sentiment_score=85, market_regime="BULL")
            
            # Verify insert called
            mock_table.insert.assert_called()
            
            # Get the args passed to insert
            call_args = mock_table.insert.call_args[0][0]
            
            print(f"Insert Args: {json.dumps(call_args, indent=2)}")
            
            # Assertions
            self.assertEqual(call_args['sentiment_score'], 85, "Sentiment Score should be passed to DB")
            self.assertEqual(call_args['market_regime'], "BULL", "Market Regime should be passed to DB")
            self.assertEqual(call_args['ticker'], "BTC-USD")
            self.assertIn('features', call_args)

if __name__ == '__main__':
    unittest.main()
