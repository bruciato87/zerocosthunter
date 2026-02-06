"""Tests for ensemble metrics reporting in MLPredictor."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@patch('db_handler.DBHandler')
def test_get_dashboard_stats_multi_model(mock_db):
    """Test that get_dashboard_stats correctly aggregates metrics from different model types."""
    from ml_predictor import MLPredictor
    
    # 1. Setup mocks to prevent DB calls during __init__
    mock_db_instance = mock_db.return_value
    
    with patch.object(MLPredictor, '_load_model'), \
         patch.object(MLPredictor, '_load_regression_model'), \
         patch.object(MLPredictor, '_load_lstm_model'):
        ml = MLPredictor()
        ml.is_ml_ready = True
        ml.model_version = "fallback_v1"

    # 2. Mock model data
    mock_model_data = [
        {
            "model_type": "classifier",
            "model_version": "clf_v1",
            "accuracy": 0.85,
            "samples_count": 100,
            "trained_at": "2026-02-04T12:00:00"
        },
        {
            "model_type": "regressor",
            "model_version": "reg_v1",
            "accuracy": 0.45, # R2
            "samples_count": 150,
            "trained_at": "2026-02-04T11:00:00"
        },
        {
            "model_type": "lstm",
            "model_version": "lstm_v1",
            "accuracy": 0.0123, # MSE
            "samples_count": 200,
            "trained_at": "2026-02-04T10:00:00"
        }
    ]
    
    # 3. Setup mock chain for Supabase calls
    # We need to distinguish between ml_model_state and ml_predictions
    def mock_table(table_name):
        mock_tbl = MagicMock()
        if table_name == "ml_model_state":
            mock_tbl.select.return_value.order.return_value.limit.return_value.execute.return_value.data = mock_model_data
        else:
            mock_tbl.select.return_value.order.return_value.limit.return_value.execute.return_value.data = []
        return mock_tbl

    mock_db_instance.supabase.table.side_effect = mock_table
    
    # Mock self methods
    with patch.object(ml, 'get_training_data_count', return_value=199):
        stats = ml.get_dashboard_stats()
    
    # 4. Assertions
    assert stats['model_version'] == "clf_v1"
    assert stats['accuracy'] == 0.85
    assert stats['reg_r2'] == 0.45
    assert stats['lstm_mse'] == 0.0123
    assert stats['training_samples'] == 100
    assert stats['available_samples'] == 199

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
