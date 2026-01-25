import pytest
from unittest.mock import MagicMock, patch
from ml_predictor import MLPredictor

def test_get_features_with_failed_resolution(mocker):
    """Test that MLPredictor handles tickers that fail resolution gracefully."""
    # 1. Mock resolve_ticker to return None
    mocker.patch("ticker_resolver.resolve_ticker", return_value=None)
    
    # 2. Initialize MLPredictor (mock _load_model to avoid DB)
    mocker.patch.object(MLPredictor, "_load_model")
    ml = MLPredictor()
    
    # 3. Call _get_features
    result = ml._get_features("INVALID")
    
    # 4. Assert result is None (handled gracefully)
    assert result is None

def test_predict_with_invalid_features(mocker):
    """Test that predict handles cases where features cannot be extracted."""
    # 1. Mock _get_features to return None
    mocker.patch.object(MLPredictor, "_load_model")
    ml = MLPredictor()
    mocker.patch.object(ml, "_get_features", return_value=None)
    
    # 2. Call predict
    prediction = ml.predict("INVALID")
    
    # 3. Assert fallback to HOLD
    assert prediction.direction == "HOLD"
    assert prediction.confidence == 0.5
    assert prediction.is_ml is False
