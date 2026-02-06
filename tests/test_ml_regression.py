"""Tests for PureGradientBoostingRegressor and predict_return()."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPureGradientBoostingRegressor:
    """Tests for the regressor class."""
    
    def test_fit_and_predict(self):
        """Test basic training and prediction."""
        from ml_predictor import PureGradientBoostingRegressor
        
        # Create simple linear data
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        model = PureGradientBoostingRegressor(n_estimators=10, learning_rate=0.1)
        model.fit(X, y)
        
        assert model.is_fitted
        predictions = model.predict(X)
        assert len(predictions) == 8
        assert predictions.dtype == np.float64
    
    def test_predict_not_fitted_raises(self):
        """Test that predicting before fitting raises error."""
        from ml_predictor import PureGradientBoostingRegressor
        
        model = PureGradientBoostingRegressor()
        X = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)
    
    def test_r2_score(self):
        """Test R² score calculation."""
        from ml_predictor import PureGradientBoostingRegressor
        
        # Perfect linear data should have high R²
        X = np.array([[i, i*2] for i in range(1, 21)])
        y = np.array([i * 2.0 for i in range(1, 21)])
        
        model = PureGradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        assert r2 > 0.5  # Should explain at least 50% of variance
    
    def test_to_json_from_json(self):
        """Test serialization round-trip."""
        from ml_predictor import PureGradientBoostingRegressor
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        
        model = PureGradientBoostingRegressor(n_estimators=5, learning_rate=0.1)
        model.fit(X, y)
        
        # Serialize
        json_str = model.to_json()
        assert 'model_type' in json_str
        assert 'regressor' in json_str
        
        # Deserialize
        loaded = PureGradientBoostingRegressor.from_json(json_str)
        
        assert loaded.is_fitted
        assert loaded.n_estimators == 5
        assert loaded.learning_rate == 0.1
        
        # Predictions should match
        np.testing.assert_array_almost_equal(
            model.predict(X), loaded.predict(X), decimal=5
        )
    
    def test_initial_prediction_is_mean(self):
        """Test that initial prediction equals target mean."""
        from ml_predictor import PureGradientBoostingRegressor
        
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # mean = 30
        
        model = PureGradientBoostingRegressor(n_estimators=5)
        model.fit(X, y)
        
        assert model.initial_prediction == 30.0


class TestReturnPrediction:
    """Tests for ReturnPrediction dataclass."""
    
    def test_return_prediction_attributes(self):
        """Test ReturnPrediction has all required attributes."""
        from ml_predictor import ReturnPrediction
        
        pred = ReturnPrediction(
            ticker="BTC",
            expected_return=5.5,
            action="BUY",
            confidence=0.75,
            is_regression=True
        )
        
        assert pred.ticker == "BTC"
        assert pred.expected_return == 5.5
        assert pred.action == "BUY"
        assert pred.confidence == 0.75
        assert pred.is_regression is True


class TestPredictReturn:
    """Tests for MLPredictor.predict_return method."""
    
    @patch('db_handler.DBHandler')
    def test_predict_return_buy_threshold(self, mock_db):
        """Test that high expected return triggers BUY."""
        from ml_predictor import MLPredictor, PureGradientBoostingRegressor
        
        # Mock DB to return no models
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        
        # Mock regression model that predicts +5%
        ml.regression_model = MagicMock()
        ml.regression_model.predict.return_value = np.array([5.0])
        
        # Mock _get_features to return valid features
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            result = ml.predict_return("BTC")
        
        assert result.action == "BUY"
        assert result.expected_return == 5.0
        assert result.is_regression is True
    
    @patch('db_handler.DBHandler')
    def test_predict_return_sell_threshold(self, mock_db):
        """Test that negative return triggers SELL."""
        from ml_predictor import MLPredictor
        
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        
        ml.regression_model = MagicMock()
        ml.regression_model.predict.return_value = np.array([-5.0])
        
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            result = ml.predict_return("AAPL")
        
        assert result.action == "SELL"
        assert result.expected_return == -5.0
        assert result.is_regression is True
    
    @patch('db_handler.DBHandler')
    def test_predict_return_hold_threshold(self, mock_db):
        """Test that small expected return triggers HOLD."""
        from ml_predictor import MLPredictor
        
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        
        ml.regression_model = MagicMock()
        ml.regression_model.predict.return_value = np.array([0.5])
        
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            result = ml.predict_return("ETH")
        
        assert result.action == "HOLD"
        assert result.expected_return == 0.5
    
    @patch('db_handler.DBHandler')
    def test_predict_return_boundary_values(self, mock_db):
        """Test exact threshold boundaries."""
        from ml_predictor import MLPredictor
        
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        ml.regression_model = MagicMock()
        
        # Test exactly at +2% (should be HOLD, not BUY - threshold is >)
        ml.regression_model.predict.return_value = np.array([2.0])
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            result = ml.predict_return("TEST")
        assert result.action == "HOLD"
        
        # Test exactly at -2% (should be HOLD, not SELL - threshold is <)
        ml.regression_model.predict.return_value = np.array([-2.0])
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            result = ml.predict_return("TEST2")
        assert result.action == "HOLD"
    
    @patch('db_handler.DBHandler')
    def test_predict_return_fallback_no_model(self, mock_db):
        """Test fallback when no regression model available."""
        from ml_predictor import MLPredictor
        
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = False
        
        # Oversold RSI should suggest UP/BUY
        with patch.object(ml, '_get_features', return_value={'rsi_14': 25, 'macd_hist': 1.0, 'bb_position': 0.1}):
            result = ml.predict_return("NVDA")
        
        assert result.is_regression is False
        assert result.action in ["BUY", "SELL", "HOLD"]
        assert result.expected_return in [5.0, -5.0, 0.0]  # Fallback values
    
    @patch('db_handler.DBHandler')
    def test_predict_return_no_features(self, mock_db):
        """Test fallback when feature extraction fails."""
        from ml_predictor import MLPredictor
        
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        ml.regression_model = MagicMock()
        
        # No features available
        with patch.object(ml, '_get_features', return_value=None):
            result = ml.predict_return("INVALID")
        
        assert result.is_regression is False
    
    @patch('db_handler.DBHandler')
    def test_predict_return_confidence_calculation(self, mock_db):
        """Test confidence scales with expected return magnitude."""
        from ml_predictor import MLPredictor
        
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        ml.regression_model = MagicMock()
        
        # Low return = lower confidence
        ml.regression_model.predict.return_value = np.array([1.0])
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            low_result = ml.predict_return("LOW")
        
        # High return = higher confidence
        ml.regression_model.predict.return_value = np.array([8.0])
        with patch.object(ml, '_get_features', return_value={'rsi_14': 50}):
            high_result = ml.predict_return("HIGH")
        
        assert high_result.confidence > low_result.confidence
        assert high_result.confidence <= 0.95  # Max cap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
