"""Tests for PureLSTM and ML Ensemble logic."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestPureLSTM:
    """Tests for the PureLSTM class."""

    def test_initialization(self):
        """Test LSTM initialization."""
        from ml_predictor import PureLSTM
        
        input_size = 5
        hidden_size = 10
        model = PureLSTM(input_size, hidden_size)
        
        assert model.input_size == input_size
        assert model.hidden_size == hidden_size
        # Check weights exist (stacked)
        assert model.W.shape == (hidden_size * 4, input_size)
        assert model.U.shape == (hidden_size * 4, hidden_size)
        assert model.b.shape == (hidden_size * 4, 1)

    def test_forward_pass(self):
        """Test forward pass output shape."""
        from ml_predictor import PureLSTM
        
        input_size = 3
        hidden_size = 4
        seq_len = 5
        batch_size = 2
        
        model = PureLSTM(input_size, hidden_size)
        
        # Input shape: (batch_size, seq_len, input_size)
        X = np.random.randn(batch_size, seq_len, input_size)
        
        # Forward
        output = model.forward(X)
        
        # Expect output to be (batch_size, 1) - currently predicting next step return
        assert output.shape == (batch_size, 1)

    def test_fit_simple(self):
        """Test simple training loop (smoke test)"""
        from ml_predictor import PureLSTM
        
        input_size = 2
        hidden_size = 4
        seq_len = 3
        
        model = PureLSTM(input_size, hidden_size)
        
        # Create dummy data: trend following
        # If input increases, target is positive
        # Shape: (2 samples, 3 steps, 2 features)
        X = np.array([
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], # Up trend
            [[0.3, 0.3], [0.2, 0.2], [0.1, 0.1]], # Down trend
        ])
        y = np.array([[1.0], [-1.0]]) # Targets (2, 1)
        
        initial_loss = model.compute_loss(X, y)
        
        # Train
        model.fit(X, y, epochs=10, learning_rate=0.1)
        
        final_loss = model.compute_loss(X, y)
        
        # Loss should decrease
        assert final_loss < initial_loss

class TestMLEnsemble:
    """Tests for Ensemble logic in MLPredictor."""
    
    @patch('db_handler.DBHandler')
    def test_ensemble_prediction(self, mock_db):
        """Test that predict_return uses both models."""
        from ml_predictor import MLPredictor
        
        # Mock DB
        mock_db.return_value.supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        mock_db.return_value.supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
        
        ml = MLPredictor()
        ml.is_regression_ready = True
        ml.is_lstm_ready = True
        
        # Mock models
        ml.regression_model = MagicMock()
        ml.regression_model.predict.return_value = np.array([2.0]) # GB says +2%
        
        ml.lstm_model = MagicMock()
        ml.lstm_model.forward.return_value = np.array([4.0]) # LSTM says +4%
        
        # Mock data fetching
        with patch.object(ml, '_get_features', return_value={'feature': 1}), \
             patch.object(ml, '_get_sequence_features', return_value=np.zeros((1, 10, 5))):
            
            result = ml.predict_return("BTC")
            
        # Expect average or weighted average
        # If 50/50, expected is 3.0
        assert 2.0 < result.expected_return < 4.0
