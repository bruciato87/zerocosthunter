"""
ML Predictor - Level 4 Machine Learning
========================================
Pure Python/NumPy Gradient Boosting + LSTM Ensemble for price prediction.
No heavy dependencies - works within Vercel 250MB limit.

This implements a real ML model that:
- Learns from historical signal data
- Trains directly on Vercel (no local setup)
- Stores model weights in Supabase
- Gets better with more data
"""

import logging
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from run_observability import RunObservability

import numpy as np
import pandas as pd

logger = logging.getLogger("MLPredictor")

def _env_flag(name: str) -> bool:
    """Parse common truthy env values."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class MLPrediction:
    """Structured ML prediction result."""
    ticker: str
    direction: str  # UP, DOWN, HOLD
    confidence: float  # 0.0 - 1.0
    features_used: Dict
    model_version: str
    is_ml: bool  # True if ML model, False if rule-based fallback


class DecisionStump:
    """Simple decision stump for gradient boosting."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_value = 0.0
        self.right_value = 0.0
    
    def fit(self, X: np.ndarray, residuals: np.ndarray):
        """Find best split for this stump."""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue
                
                left_value = residuals[left_mask].mean()
                right_value = residuals[right_mask].mean()
                
                # Calculate gain (reduction in variance)
                predictions = np.where(left_mask, left_value, right_value)
                gain = -np.mean((residuals - predictions) ** 2)
                
                if gain > best_gain:
                    best_gain = gain
                    self.feature_idx = feature_idx
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return np.where(
            X[:, self.feature_idx] <= self.threshold,
            self.left_value,
            self.right_value
        )
    
    def to_dict(self) -> Dict:
        return {
            'feature_idx': int(self.feature_idx),
            'threshold': float(self.threshold),
            'left_value': float(self.left_value),
            'right_value': float(self.right_value)
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DecisionStump':
        stump = cls()
        stump.feature_idx = d['feature_idx']
        stump.threshold = d['threshold']
        stump.left_value = d['left_value']
        stump.right_value = d['right_value']
        return stump


class PureGradientBoosting:
    """
    Pure Python/NumPy Gradient Boosting Classifier.
    Lightweight implementation for serverless deployment.
    """
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps: List[List[DecisionStump]] = []  # One list per class
        self.classes_ = None
        self.initial_predictions = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)
        
        # One-vs-all encoding
        Y = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes_):
            Y[:, i] = (y == c).astype(float)
        
        # Initial predictions (log-odds)
        self.initial_predictions = np.log(Y.mean(axis=0) + 1e-10)
        
        # Initialize predictions
        F = np.tile(self.initial_predictions, (n_samples, 1))
        
        self.stumps = [[] for _ in range(n_classes)]
        
        for _ in range(self.n_estimators):
            # Compute probabilities via softmax
            exp_F = np.exp(F - F.max(axis=1, keepdims=True))
            probs = exp_F / exp_F.sum(axis=1, keepdims=True)
            
            # Compute residuals for each class
            residuals = Y - probs
            
            for class_idx in range(n_classes):
                stump = DecisionStump()
                stump.fit(X, residuals[:, class_idx])
                self.stumps[class_idx].append(stump)
                
                # Update predictions
                F[:, class_idx] += self.learning_rate * stump.predict(X)
        
        self.is_fitted = True
        logger.info(f"Model trained with {self.n_estimators} estimators")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Start with initial predictions
        F = np.tile(self.initial_predictions, (n_samples, 1))
        
        # Add contributions from all stumps
        for class_idx in range(n_classes):
            for stump in self.stumps[class_idx]:
                F[:, class_idx] += self.learning_rate * stump.predict(X)
        
        # Softmax
        exp_F = np.exp(F - F.max(axis=1, keepdims=True))
        probs = exp_F / exp_F.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy."""
        predictions = self.predict(X)
        return (predictions == y).mean()
    
    def to_json(self) -> str:
        """Serialize model to JSON for storage."""
        return json.dumps({
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'classes': self.classes_.tolist() if self.classes_ is not None else None,
            'initial_predictions': self.initial_predictions.tolist() if self.initial_predictions is not None else None,
            'stumps': [[s.to_dict() for s in class_stumps] for class_stumps in self.stumps],
            'is_fitted': self.is_fitted
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PureGradientBoosting':
        """Deserialize model from JSON."""
        d = json.loads(json_str)
        model = cls(n_estimators=d['n_estimators'], learning_rate=d['learning_rate'])
        model.classes_ = np.array(d['classes']) if d['classes'] else None
        model.initial_predictions = np.array(d['initial_predictions']) if d['initial_predictions'] else None
        model.stumps = [[DecisionStump.from_dict(s) for s in class_stumps] for class_stumps in d['stumps']]
        model.is_fitted = d['is_fitted']
        return model


class PureGradientBoostingRegressor:
    """
    Pure Python/NumPy Gradient Boosting Regressor.
    Predicts continuous returns (5-day expected %).
    """
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps: List[DecisionStump] = []
        self.initial_prediction = 0.0
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train regressor with MSE loss."""
        n_samples = len(y)
        
        # Initial prediction (mean of targets)
        self.initial_prediction = float(np.mean(y))
        
        # Initialize residuals
        residuals = y - self.initial_prediction
        
        self.stumps = []
        
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.stumps.append(stump)
            
            # Update residuals (gradient descent step)
            predictions = stump.predict(X)
            residuals = residuals - self.learning_rate * predictions
        
        self.is_fitted = True
        logger.info(f"Regression model trained with {self.n_estimators} estimators")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous return %."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction)
        
        for stump in self.stumps:
            predictions += self.learning_rate * stump.predict(X)
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def to_json(self) -> str:
        """Serialize model to JSON."""
        return json.dumps({
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'initial_prediction': float(self.initial_prediction),
            'stumps': [s.to_dict() for s in self.stumps],
            'is_fitted': self.is_fitted,
            'model_type': 'regressor'
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PureGradientBoostingRegressor':
        """Deserialize model from JSON."""
        d = json.loads(json_str)
        model = cls(n_estimators=d['n_estimators'], learning_rate=d['learning_rate'])
        model.initial_prediction = d['initial_prediction']
        model.stumps = [DecisionStump.from_dict(s) for s in d['stumps']]
        model.is_fitted = d['is_fitted']
        return model


@dataclass
class ReturnPrediction:
    """Regression prediction result."""
    ticker: str
    expected_return: float  # Expected 5-day return %
    action: str  # BUY, SELL, HOLD based on thresholds
    confidence: float  # R² based
    is_regression: bool  # True if real model, False if fallback


class PureLSTM:
    """
    Pure Python/NumPy LSTM Implementation.
    Lightweight Recurrent Neural Network for time-series forecasting.
    Uses BPTT (Backpropagation Through Time) for training.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier Initialization
        limit = np.sqrt(6 / (input_size + hidden_size))
        
        # Input Gates (i, f, o, g) - Stacked for efficiency
        # Order: Input, Forget, Output, Gate (cell update)
        self.W = np.random.uniform(-limit, limit, (hidden_size * 4, input_size))
        self.U = np.random.uniform(-limit, limit, (hidden_size * 4, hidden_size))
        self.b = np.zeros((hidden_size * 4, 1))
        
        # Forget gate bias initialization to 1.0 (helps gradient flow)
        self.b[hidden_size:2*hidden_size] = 1.0
        
        # Output Linear Layer (Regression to scalar)
        self.Why = np.random.uniform(-limit, limit, (1, hidden_size))
        self.by = np.zeros((1, 1))
        
        # Cache for Backprop
        self.cache = None
        self.is_fitted = False
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    
    def d_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
        
    def tanh(self, x):
        return np.tanh(x)
        
    def d_tanh(self, x):
        t = np.tanh(x)
        return 1 - t ** 2
        
    def forward(self, X):
        """
        X shape: (batch_size, seq_len, input_size)
        Returns: last_output (batch_size, 1)
        """
        batch_size, seq_len, _ = X.shape
        hidden_size = self.hidden_size
        
        # States
        h = np.zeros((batch_size, hidden_size))
        c = np.zeros((batch_size, hidden_size))
        
        # Cache for backprop
        self.cache = {
            'X': X,
            'h_states': [h], # h_prev at t=0
            'c_states': [c],
            'gates': []
        }
        
        for t in range(seq_len):
            x_t = X[:, t, :].T # (input, batch)
            
            # Linear transformations
            # z = Wx + Uh + b
            # shape: (4*hidden, batch)
            gates = np.dot(self.W, x_t) + np.dot(self.U, h.T) + self.b 
            
            # Split gates
            # i: input gate, f: forget gate, o: output gate, g: cell candidate
            i_gate = self.sigmoid(gates[0:hidden_size, :])
            f_gate = self.sigmoid(gates[hidden_size:2*hidden_size, :])
            o_gate = self.sigmoid(gates[2*hidden_size:3*hidden_size, :])
            g_gate = self.tanh(gates[3*hidden_size:4*hidden_size, :])
            
            # Update Cell
            c_next = f_gate * c.T + i_gate * g_gate
            c_next = c_next.T # back to (batch, hidden)
            
            # Update Hidden
            h_next = o_gate.T * self.tanh(c_next)
            
            # Store states
            self.cache['h_states'].append(h_next)
            self.cache['c_states'].append(c_next)
            self.cache['gates'].append((i_gate, f_gate, o_gate, g_gate))
            
            h = h_next
            c = c_next
            
        # Readout Layer (Last time step only for Many-to-One)
        y = np.dot(self.Why, h.T) + self.by
        
        return y.T # (batch, 1)

    def compute_loss(self, X, y_target):
        # MSE Loss
        y_pred = self.forward(X)
        return np.mean((y_pred - y_target) ** 2)

    def fit(self, X, y, epochs=50, learning_rate=0.01):
        """Train sequence model with basic SGD on Vercel."""
        batch_size = len(y)
        if batch_size == 0: return

        # Reduce epochs/LR for stability if needed
        clip_value = 5.0
        
        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)
            
            # Loss Gradient (MSE) -> dL/dy
            dy = 2 * (y_pred - y) / batch_size # (batch, 1)
            
            # Readout Gradients
            # y = Why * h + by
            # dy/dWhy = dy * h
            # dy/dbh = dy
            h_last = self.cache['h_states'][-1] # (batch, hidden)
            
            dWhy = np.dot(dy.T, h_last)
            dby = np.sum(dy, axis=0, keepdims=True).T
            
            # Backprop through time (BPTT)
            dh = np.dot(dy, self.Why) # (batch, hidden)
            dc = np.zeros_like(dh)
            
            dW = np.zeros_like(self.W)
            dU = np.zeros_like(self.U)
            db = np.zeros_like(self.b)
            
            seq_len = X.shape[1]
            
            for t in reversed(range(seq_len)):
                i, f, o, g = self.cache['gates'][t] # shapes (hidden, batch)
                c = self.cache['c_states'][t+1].T 
                c_prev = self.cache['c_states'][t].T
                
                tanh_c = self.tanh(c)
                
                # Gradients through Gates
                # h = o * tanh(c)
                do = dh.T * tanh_c
                dcs = dh.T * o * self.d_tanh(c) + dc.T
                
                # c = f * c_prev + i * g
                df = dcs * c_prev
                di = dcs * g
                dg = dcs * i
                dc_prev = dcs * f # Propagate back to previous cell
                
                # Derivatives of activations
                d_gates = np.zeros((4 * self.hidden_size, batch_size))
                d_gates[0:self.hidden_size, :] = di * i * (1 - i)
                d_gates[self.hidden_size:2*self.hidden_size, :] = df * f * (1 - f)
                d_gates[2*self.hidden_size:3*self.hidden_size, :] = do * o * (1 - o)
                d_gates[3*self.hidden_size:4*self.hidden_size, :] = dg * (1 - g**2)
                
                # Accumulate gradients
                x_t = X[:, t, :].T
                h_prev = self.cache['h_states'][t].T
                
                dW += np.dot(d_gates, x_t.T)
                dU += np.dot(d_gates, h_prev.T)
                db += np.sum(d_gates, axis=1, keepdims=True)
                
                # Update dh for previous step
                dh = np.dot(self.U.T, d_gates).T
                dc = dc_prev.T
                
            # Gradient Clipping
            for grad in [dW, dU, db, dWhy, dby]:
                np.clip(grad, -clip_value, clip_value, out=grad)
                
            # Update Weights
            self.W -= learning_rate * dW
            self.U -= learning_rate * dU
            self.b -= learning_rate * db
            self.Why -= learning_rate * dWhy
            self.by -= learning_rate * dby
            
            # Stability check
            if np.isnan(self.W).any() or np.isnan(self.U).any() or np.isnan(self.b).any() or \
               np.isnan(self.Why).any() or np.isnan(self.by).any():
                logger.error(f"LSTM: NaN detected in weights at epoch {epoch}. Aborting training.")
                self.is_fitted = False
                return
            
        self.is_fitted = True
        logger.info(f"LSTM trained with {epochs} epochs")

    def to_json(self):
        return json.dumps({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'W': self.W.tolist(),
            'U': self.U.tolist(),
            'b': self.b.tolist(),
            'Why': self.Why.tolist(),
            'by': self.by.tolist(),
            'is_fitted': self.is_fitted
        })
        
    @classmethod
    def from_json(cls, json_str):
        d = json.loads(json_str)
        model = cls(d['input_size'], d['hidden_size'])
        model.W = np.array(d['W'])
        model.U = np.array(d['U'])
        model.b = np.array(d['b'])
        model.Why = np.array(d['Why'])
        model.by = np.array(d['by'])
        model.is_fitted = d['is_fitted']
        return model


class MLPredictor:
    """
    ML Predictor using Pure Python Gradient Boosting.
    Learns from historical data, runs entirely on Vercel.
    """
    
    MIN_TRAINING_SAMPLES = 30  # Reduced from 50 for faster start
    LSTM_SEQ_LEN = 10
    WALK_FORWARD_MAX_FOLDS = 4
    WALK_FORWARD_MIN_TRAIN_RATIO = 0.65
    CLASSIFIER_MIN_GAIN = 0.02
    REGRESSION_MIN_GAIN = 0.02
    LSTM_MIN_GAIN = 0.015
    MAX_CHAMPION_DEGRADATION = 0.01
    
    FEATURE_COLUMNS = [
        'rsi_14', 'rsi_slope', 
        'macd_hist', 'macd_signal_cross',
        'bb_position', 'bb_width',
        'volume_ratio', 'volume_trend',
        'price_sma20_ratio', 'price_sma50_ratio',
        'atr_percent', 'momentum_10d',
        'vix_level', 'vix_change',
        'market_regime_encoded',
        # Phase 1 Quick Win: Time-aware features for calendar effects
        'day_of_week',      # 0=Monday..6=Sunday (Monday effect)
        'month',            # 1-12 (January effect, Sell in May)
        'is_month_end',     # 1 if last 5 days of month (rebalancing flows)
        'is_opex_week',     # 1 if options expiry week (volatility spike)
        # Phase 2: Sentiment features for narrative alpha
        'fear_greed_score',     # 0-100 market-wide fear/greed index
        'social_hype_score',    # 0-10 social media hype level
        'whale_activity_score', # -100 to +100 (distribution to accumulation)
    ]
    
    def __init__(self):
        self.model: Optional[PureGradientBoosting] = None
        self.regression_model: Optional[PureGradientBoostingRegressor] = None
        self.lstm_model: Optional[PureLSTM] = None
        self.dry_run = _env_flag("DRY_RUN")
        
        self.model_version = "pure_gb_v1"
        self.regression_model_version = "pure_gbr_v1"
        self.lstm_model_version = "pure_lstm_v1"
        
        self.is_ml_ready = False
        self.is_regression_ready = False
        self.is_lstm_ready = False
        
        # Internal caches for the current session/training run
        self._feature_cache = {}
        self._prediction_cache = {}
        self._market_data_cache = {} # Central cache for raw OHLCV data
        self._session_cache = {}     # Global features (VIX, Regime, Hype, etc.)
        self._ml_registry_enabled = True
        self.last_training_report: Dict[str, Any] = {}
        
        self._load_model()
        self._load_regression_model()
        self._load_lstm_model()
        
        # Phase 6: ML Health Monitor
        from ml_health_monitor import MLHealthMonitor
        from db_handler import DBHandler
        self.db = DBHandler()
        self.health = MLHealthMonitor(db=self.db)
        if self.dry_run:
            logger.info("MLPredictor DRY_RUN enabled: model/prediction persistence and Telegram notifications are disabled.")
    
    def _load_model(self):
        """Load trained model from Supabase."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Check for stored classifier model
            result = db.supabase.table("ml_model_state") \
                .select("model_version, model_weights") \
                .eq("model_type", "classifier") \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            
            if result.data and result.data[0].get('model_weights'):
                state = result.data[0]
                self.model = PureGradientBoosting.from_json(state['model_weights'])
                self.model_version = state.get('model_version', 'pure_gb_v1')
                self.is_ml_ready = self.model.is_fitted
                logger.info(f"ML Predictor: Loaded model {self.model_version}")
            else:
                logger.info("ML Predictor: No trained model found, using rule-based")
                
        except Exception as e:
            logger.warning(f"ML Predictor: Model load failed: {e}")
    
    def _load_regression_model(self):
        """Load regression model from Supabase."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            result = db.supabase.table("ml_model_state") \
                .select("model_version, model_weights") \
                .eq("model_type", "regressor") \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            
            if result.data and result.data[0].get('model_weights'):
                state = result.data[0]
                self.regression_model = PureGradientBoostingRegressor.from_json(state['model_weights'])
                self.regression_model_version = state.get('model_version', 'pure_gbr_v1')
                self.is_regression_ready = self.regression_model.is_fitted
                logger.info(f"ML Predictor: Loaded regression model {self.regression_model_version}")
            else:
                logger.info("ML Predictor: No regression model found")
                
        except Exception as e:
            logger.warning(f"ML Predictor: Regression model load failed: {e}")

    def _load_lstm_model(self):
        """Load LSTM model from Supabase."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            result = db.supabase.table("ml_model_state") \
                .select("model_version, model_weights") \
                .eq("model_type", "lstm") \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            
            if result.data and result.data[0].get('model_weights'):
                state = result.data[0]
                self.lstm_model = PureLSTM.from_json(state['model_weights'])
                self.lstm_model_version = state.get('model_version', 'pure_lstm_v1')
                self.is_lstm_ready = self.lstm_model.is_fitted
                logger.info(f"ML Predictor: Loaded LSTM model {self.lstm_model_version}")
            else:
                logger.info("ML Predictor: No LSTM model found")
                
        except Exception as e:
            logger.warning(f"ML Predictor: LSTM model load failed: {e}")

    def _get_sequence_features(self, ticker: str) -> Optional[np.ndarray]:
        """Extract sequence of features for LSTM (Last N days)."""
        try:
            import ta
            from ticker_resolver import resolve_ticker
            ticker_u = (ticker or "").upper().strip()
            if not ticker_u:
                return None

            if ticker_u in self._feature_cache and self._feature_cache[ticker_u] is None:
                return None
            
            # Check cache first
            if ticker_u in self._market_data_cache:
                hist = self._market_data_cache[ticker_u]
            else:
                import yfinance as yf
                search_ticker = resolve_ticker(ticker_u)
                if not search_ticker:
                    self._record_ticker_data_failure(ticker_u, "resolve_ticker returned None for sequence extraction")
                    return None
                hist = yf.Ticker(search_ticker).history(period="60d")
                if hist.empty and search_ticker != ticker_u:
                    hist = yf.Ticker(ticker_u).history(period="60d")
                self._market_data_cache[ticker_u] = hist

            if hist.empty:
                self._record_ticker_data_failure(ticker_u, "empty yfinance history for LSTM sequence")
                return None
            if len(hist) < 30:
                return None
            
            df = hist.copy()
            # ... Rest of logic remains the same but uses 'hist' from cache ...
            
            # Normalized features for LSTM
            # 1. Log Returns
            df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
            # 2. Volume Change
            df['vol_chg'] = df['Volume'].pct_change()
            # 3. High-Low Range
            df['range'] = (df['High'] - df['Low']) / df['Close']
            
            # Select columns
            cols = ['log_ret', 'vol_chg', 'range']
            
            # Drop NaNs and Infs (Safety)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(df) < self.LSTM_SEQ_LEN:
                return None
                
            # Get last seq_len rows
            seq = df[cols].tail(self.LSTM_SEQ_LEN).values # (10, 3)
            
            # Normalize sequence (Z-score per column)
            std = np.std(seq, axis=0)
            seq = (seq - np.mean(seq, axis=0)) / (std + 1e-6)
            
            # Final safety check for nans in the normalized sequence
            if np.isnan(seq).any():
                return None
                
            # Add batch dim
            seq = seq.reshape(1, self.LSTM_SEQ_LEN, len(cols))
            
            return seq
            
        except Exception as e:
            logger.warning(f"LSTM Sequence extraction error: {e}")
            return None
    
    def _get_features(self, ticker: str) -> Optional[Dict]:
        """Extract features for prediction."""
        try:
            import ta
            from ticker_resolver import resolve_ticker, is_probable_ticker
            ticker_u = (ticker or "").upper().strip()
            if not ticker_u:
                return None

            if ticker_u in self._feature_cache:
                return self._feature_cache[ticker_u]

            if not is_probable_ticker(ticker_u):
                self._record_ticker_data_failure(ticker_u, "ticker failed basic sanity pattern")
                return None
            
            # --- 1. SESSION-LEVEL GLOBAL FEATURES (Populated once per run) ---
            if "_global" not in self._session_cache:
                global_f = {}
                # VIX
                try:
                    import yfinance as yf
                    vix_data = yf.Ticker("^VIX").history(period="5d")
                    if not vix_data.empty:
                        global_f['vix_level'] = vix_data['Close'].iloc[-1]
                        global_f['vix_change'] = (vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[0]) / vix_data['Close'].iloc[0] * 100 if len(vix_data) > 1 else 0
                    else:
                        global_f['vix_level'] = 20
                        global_f['vix_change'] = 0
                except:
                    global_f['vix_level'] = 20
                    global_f['vix_change'] = 0
                
                # Market Regime
                try:
                    from market_regime import MarketRegimeClassifier
                    regime = MarketRegimeClassifier().classify()
                    regime_map = {'BULL': 1, 'SIDEWAYS': 0, 'BEAR': -1}
                    global_f['market_regime_encoded'] = regime_map.get(regime.get('regime', 'SIDEWAYS'), 0)
                except:
                    global_f['market_regime_encoded'] = 0
                
                # Sentiment (Global indicators)
                try:
                    from sentiment_aggregator import SentimentAggregator
                    sa = SentimentAggregator()
                    # Pre-calculate scores (avoids multiple network calls per session)
                    global_f['_sent_stock'] = sa.get_numeric_scores(is_crypto=False)
                    global_f['_sent_crypto'] = sa.get_numeric_scores(is_crypto=True)
                except:
                    global_f['_sent_stock'] = {"fear_greed_score": 50, "whale_activity_score": 0, "vix_normalized": 50}
                    global_f['_sent_crypto'] = global_f['_sent_stock']
                
                # Social Trends (Scraped once per session)
                try:
                    from social_scraper import SocialScraper
                    ss = SocialScraper()
                    global_f['_trending'] = ss.get_reddit_trending()
                except:
                    global_f['_trending'] = {}

                self._session_cache["_global"] = global_f

            global_features = self._session_cache["_global"]
            
            # --- 2. TICKER-SPECIFIC MARKET DATA (Cached iteratively) ---
            if ticker_u in self._market_data_cache:
                hist = self._market_data_cache[ticker_u]
            else:
                import yfinance as yf
                search_ticker = resolve_ticker(ticker_u)
                if not search_ticker:
                    self._record_ticker_data_failure(ticker_u, "resolve_ticker returned None")
                    return None
                t = yf.Ticker(search_ticker)
                hist = t.history(period="3mo")
                if hist.empty or len(hist) < 30:
                    if search_ticker != ticker_u:
                        t = yf.Ticker(ticker_u)
                        hist = t.history(period="3mo")
                self._market_data_cache[ticker_u] = hist
                
            if hist.empty:
                self._record_ticker_data_failure(ticker_u, "empty yfinance history")
                return None
            if len(hist) < 30:
                self._feature_cache[ticker_u] = None
                return None
            
            df = hist.copy()
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            features = {}
            
            # RSI
            rsi = ta.momentum.RSIIndicator(close, window=14)
            features['rsi_14'] = rsi.rsi().iloc[-1]
            rsi_values = rsi.rsi().tail(5)
            features['rsi_slope'] = (rsi_values.iloc[-1] - rsi_values.iloc[0]) / 5 if len(rsi_values) >= 5 else 0
            
            # MACD
            macd = ta.trend.MACD(close)
            features['macd_hist'] = macd.macd_diff().iloc[-1]
            macd_prev = macd.macd_diff().iloc[-2] if len(macd.macd_diff()) > 1 else 0
            features['macd_signal_cross'] = 1 if features['macd_hist'] > 0 and macd_prev <= 0 else (-1 if features['macd_hist'] < 0 and macd_prev >= 0 else 0)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            bb_mid = bb.bollinger_mavg().iloc[-1]
            current_price = close.iloc[-1]
            
            if bb_high != bb_low:
                features['bb_position'] = (current_price - bb_low) / (bb_high - bb_low)
            else:
                features['bb_position'] = 0.5
            features['bb_width'] = (bb_high - bb_low) / bb_mid if bb_mid > 0 else 0
            
            # Volume
            vol_avg = volume.tail(20).mean()
            features['volume_ratio'] = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1.0
            vol_5d = volume.tail(5).mean()
            vol_20d = volume.tail(20).mean()
            features['volume_trend'] = vol_5d / vol_20d if vol_20d > 0 else 1.0
            
            # Moving Averages
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
            features['price_sma20_ratio'] = current_price / sma20 if sma20 > 0 else 1.0
            features['price_sma50_ratio'] = current_price / sma50 if sma50 > 0 else 1.0
            
            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
            features['atr_percent'] = (atr.average_true_range().iloc[-1] / current_price * 100) if current_price > 0 else 0
            
            # Momentum
            if len(close) >= 10:
                features['momentum_10d'] = (current_price - close.iloc[-10]) / close.iloc[-10] * 100
            else:
                features['momentum_10d'] = 0
            
            # Populate features from session cache
            features['vix_level'] = global_features.get('vix_level', 20)
            features['vix_change'] = global_features.get('vix_change', 0)
            features['market_regime_encoded'] = global_features.get('market_regime_encoded', 0)
            
            # Phase 1 Quick Win: Time-Aware Features
            from datetime import datetime
            import calendar
            now = datetime.now()
            
            # Day of week (0=Monday, 6=Sunday) - Monday effect
            features['day_of_week'] = now.weekday()
            
            # Month (1-12) - January effect, Sell in May
            features['month'] = now.month
            
            # Is month end (last 5 days) - Portfolio rebalancing flows
            days_in_month = calendar.monthrange(now.year, now.month)[1]
            features['is_month_end'] = 1 if now.day > (days_in_month - 5) else 0
            
            # Is OPEX week (3rd Friday of month) - Options expiry volatility
            def is_opex_week(dt):
                """Check if date falls in options expiry week (week containing 3rd Friday)."""
                # Find 3rd Friday
                first_day = dt.replace(day=1)
                first_friday = (4 - first_day.weekday()) % 7 + 1
                third_friday = first_friday + 14
                # OPEX week is Mon-Fri of that week
                opex_week_start = third_friday - 4 if third_friday > 4 else 1
                opex_week_end = third_friday
                return opex_week_start <= dt.day <= opex_week_end
            
            features['is_opex_week'] = 1 if is_opex_week(now) else 0
            
            # Phase 2: Sentiment & Hype (Using session cache)
            is_crypto = ticker_u in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'MATIC', 'DOT', 'ATOM', 'LINK', 'UNI', 'AAVE', 'RENDER']
            
            # Sentiment scores
            sent_scores = global_features.get('_sent_crypto' if is_crypto else '_sent_stock', {})
            features['fear_greed_score'] = sent_scores.get('fear_greed_score', 50)
            features['whale_activity_score'] = sent_scores.get('whale_activity_score', 0)
            
            # Hype score logic (Simplified for session cache)
            # count reflects mentions in trending which was scraped once
            trending = global_features.get('_trending', {})
            count = trending.get(ticker_u, 0)
            features['social_hype_score'] = min(8.0, count * 2.0) if count > 0 else 0.0
            
            # Add velocity placeholder or simplified check
            # (Velocity relies on DB, let's keep it simple during massive training runs to save time)
            if ticker_u in trending and count > 5:
                features['social_hype_score'] += 2.0 # Surge bonus
            
            # Handle NaN/Inf (Phase 6: Robust Cleaning)
            for key, val in features.items():
                try:
                    fval = float(val)
                    if np.isnan(fval) or np.isinf(fval):
                        features[key] = 0.0
                    else:
                        features[key] = fval
                except (ValueError, TypeError):
                    features[key] = 0.0
                    
            self._feature_cache[ticker_u] = features
            return features
            
        except Exception as e:
            logger.error(f"ML Feature extraction failed for {ticker}: {e}")
            ticker_u = (ticker or "").upper().strip()
            if ticker_u:
                self._feature_cache[ticker_u] = None
            return None
    
    def _rule_based_predict(self, features: Dict) -> Tuple[str, float]:
        """Rule-based fallback predictor."""
        score = 0.0
        max_score = 7.0
        
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            score += 1.5
        elif rsi > 70:
            score -= 1.5
        elif rsi < 45:
            score += 0.5
        elif rsi > 55:
            score -= 0.5
        
        macd_hist = features.get('macd_hist', 0)
        if macd_hist > 0:
            score += 1.0
        elif macd_hist < 0:
            score -= 1.0
        
        bb_pos = features.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 1.0
        elif bb_pos > 0.8:
            score -= 1.0
        
        sma20_ratio = features.get('price_sma20_ratio', 1.0)
        if sma20_ratio > 1.02:
            score += 1.0
        elif sma20_ratio < 0.98:
            score -= 1.0
        
        vix = features.get('vix_level', 20)
        if vix > 30:
            score -= 0.5
        elif vix < 15:
            score += 0.5
        
        momentum = features.get('momentum_10d', 0)
        if momentum > 5:
            score += 0.5
        elif momentum < -5:
            score -= 0.5
        
        normalized = score / max_score
        
        if normalized > 0.2:
            direction = "UP"
            confidence = min(0.5 + abs(normalized) * 0.4, 0.85)
        elif normalized < -0.2:
            direction = "DOWN"
            confidence = min(0.5 + abs(normalized) * 0.4, 0.85)
        else:
            direction = "HOLD"
            confidence = 0.5 + (0.2 - abs(normalized)) * 0.3
        
        return direction, confidence
    
    def _ml_predict(self, features: Dict) -> Tuple[str, float]:
        """ML model prediction."""
        try:
            # Convert features to array
            feature_array = np.array([[features.get(f, 0) for f in self.FEATURE_COLUMNS]])
            
            # Get predictions
            probs = self.model.predict_proba(feature_array)[0]
            prediction = self.model.predict(feature_array)[0]
            
            direction_map = {0: "DOWN", 1: "HOLD", 2: "UP"}
            direction = direction_map.get(int(prediction), "HOLD")
            confidence = float(max(probs))
            
            logger.info(f"ML Prediction: {direction} ({confidence:.0%})")
            return direction, confidence
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._rule_based_predict(features)
    
    def predict(self, ticker: str, sentiment_score: int = None, market_regime: str = None) -> MLPrediction:
        """Main prediction method. Now accepts external context (Quant Path)."""
        ticker_u = ticker.upper()
        
        # Check cache (exact match for ticker/regime/sentiment)
        cache_key = f"{ticker_u}_{sentiment_score}_{market_regime}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        if ticker_u in self._feature_cache:
            features = self._feature_cache[ticker_u]
        else:
            features = self._get_features(ticker)
            self._feature_cache[ticker_u] = features
        
        if not features:
            return MLPrediction(
                ticker=ticker,
                direction="HOLD",
                confidence=0.5,
                features_used={},
                model_version="fallback",
                is_ml=False
            )
        
        # Use Ensemble Prediction if ready, otherwise fallback to Classification or Rules
        try:
            if (self.is_regression_ready or self.is_lstm_ready):
                ret_pred = self.predict_return(ticker)
                direction = ret_pred.action
                if direction == "BUY": direction = "UP"
                elif direction == "SELL": direction = "DOWN"
                
                confidence = ret_pred.confidence
                is_ml = ret_pred.is_regression
                model_version = f"Ensemble({self.regression_model_version}/{self.lstm_model_version})"
            elif self.is_ml_ready and self.model is not None:
                direction, confidence = self._ml_predict(features)
                is_ml = True
                model_version = self.model_version
            else:
                direction, confidence = self._rule_based_predict(features)
                is_ml = False
                model_version = "rule_based"
            
            # Log success
            self.health.log_prediction('classifier', 'SUCCESS')
        except Exception as e:
            logger.error(f"Prediction logic error for {ticker}: {e}")
            direction, confidence = self._rule_based_predict(features)
            is_ml = False
            model_version = "rule_fallback"
            self.health.log_prediction('classifier', 'FAILURE', str(e))
        
        # Save prediction with context
        self._save_prediction(ticker, direction, confidence, features, sentiment_score, market_regime)
        
        res = MLPrediction(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            features_used=features,
            model_version=model_version,
            is_ml=is_ml
        )
        
        self._prediction_cache[cache_key] = res
        return res
    
    def _save_prediction(self, ticker: str, direction: str, confidence: float, features: Dict, sentiment_score: int = None, market_regime: str = None):
        """Save prediction to DB."""
        if self.dry_run:
            logger.info("DRY_RUN: skipping prediction persistence for %s", ticker.upper())
            return
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            db.supabase.table("ml_predictions").insert({
                "ticker": ticker.upper(),
                "predicted_direction": direction,
                "ml_confidence": confidence,
                "features": json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()}),
                "sentiment_score": sentiment_score,
                "market_regime": market_regime
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save prediction: {e}")
    
    def get_confidence_modifier(self, ticker: str, ai_sentiment: str, sentiment_score: int = None, market_regime: str = None) -> float:
        """Get confidence modifier based on ML agreement with AI."""
        prediction = self.predict(ticker, sentiment_score, market_regime)
        return self.get_confidence_modifier_from_pred(prediction, ai_sentiment)

    def get_confidence_modifier_from_pred(self, prediction: MLPrediction, ai_sentiment: str) -> float:
        """Calculate modifier from an existing prediction object."""
        bullish = {"BUY", "ACCUMULATE", "STRONG BUY"}
        bearish = {"SELL", "PANIC SELL", "STRONG SELL"}
        
        if ai_sentiment.upper() in bullish:
            ai_dir = "UP"
        elif ai_sentiment.upper() in bearish:
            ai_dir = "DOWN"
        else:
            ai_dir = "HOLD"
        
        if prediction.direction == ai_dir:
            return 1.15 if prediction.confidence > 0.7 else 1.08
        elif prediction.direction == "HOLD":
            return 0.95
        else:
            return 0.85 if prediction.confidence > 0.7 else 0.92
    
    def get_context_for_ai(self, ticker: str) -> str:
        """Generate context for AI prompt."""
        pred = self.predict(ticker)
        model_type = "Pure-ML" if pred.is_ml else "Rule-Based"
        return f"[{model_type} PREDICTOR: {ticker} → {pred.direction} ({pred.confidence:.0%})]"
    
    def predict_return(self, ticker: str) -> ReturnPrediction:
        """
        Ensemble Prediction (GB + LSTM).
        Returns regression prediction object.
        """
        # Feature Extraction
        features = self._get_features(ticker)
        seq_features = self._get_sequence_features(ticker)
        
        gb_pred = 0.0
        lstm_pred = 0.0
        
        used_regression = False
        used_lstm = False
        
        # 1. Gradient Boosting Prediction
        if self.is_regression_ready and self.regression_model and features:
            try:
                X = np.array([list(features.values())])
                gb_pred = float(self.regression_model.predict(X)[0])
                used_regression = True
                self.health.log_prediction('regressor', 'SUCCESS')
            except Exception as e:
                logger.error(f"GB Prediction failed: {e}")
                self.health.log_prediction('regressor', 'FAILURE', str(e))
                
        # 2. LSTM Prediction
        if self.is_lstm_ready and self.lstm_model and seq_features is not None:
            try:
                raw_pred = self.lstm_model.forward(seq_features)
                # Handle potential array output (e.g. [1, 1] or similar)
                if isinstance(raw_pred, (np.ndarray, list)):
                    lstm_pred = float(raw_pred.flatten()[0])
                else:
                    lstm_pred = float(raw_pred)
                    
                used_lstm = True
                self.health.log_prediction('lstm', 'SUCCESS')
            except Exception as e:
                logger.error(f"LSTM Prediction failed: {e}")
                self.health.log_prediction('lstm', 'FAILURE', str(e))
        
        # Ensemble Logic
        final_return = 0.0
        confidence = 0.0
        
        if used_regression and used_lstm:
            # Weighted Ensemble: GB (60%) + LSTM (40%)
            final_return = (gb_pred * 0.6) + (lstm_pred * 0.4)
            confidence = 0.85 # High confidence if both models agree on sign
            if np.sign(gb_pred) != np.sign(lstm_pred):
                confidence = 0.50 # Low confidence if disagreement
        elif used_regression:
            final_return = gb_pred
            confidence = 0.70
        elif used_lstm:
            final_return = lstm_pred
            confidence = 0.60
        else:
            # Rule-based fallback (using features if available)
            if features:
                 rsi = features.get('rsi_14', 50)
                 if rsi < 30: final_return = 5.0
                 elif rsi > 70: final_return = -5.0
            confidence = 0.40
            
        # Determine Action
        action = "HOLD"
        # Thresholds: > +2% BUY, < -2% SELL
        if final_return > 2.0:
            action = "BUY"
        elif final_return < -2.0:
            action = "SELL"
            
        # Refine confidence based on magnitude (stronger signal = higher confidence)
        if action != "HOLD":
             confidence = min(0.95, confidence + (abs(final_return) / 50.0))
            
        return ReturnPrediction(
            ticker=ticker,
            expected_return=round(final_return, 2),
            action=action,
            confidence=round(confidence, 2),
            is_regression=(used_regression or used_lstm)
        )
    
    def get_training_data_count(self) -> int:
        """Count available training samples."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            result = db.supabase.table("signal_tracking") \
                .select("id", count="exact") \
                .in_("status", ["WIN", "LOSS"]) \
                .execute()
            
            return result.count if result.count else 0
        except:
            return 0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert unknown payload values to float safely."""
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _normalize_metric_for_gate(self, model_type: str, metric: Optional[float], source: str) -> Optional[float]:
        """
        Normalize legacy metrics before comparing challenger vs champion.
        Current gate metrics are probabilities/accuracies in [0, 1].
        """
        if metric is None:
            return None
        metric_f = self._safe_float(metric, None)
        if metric_f is None:
            return None
        if 0.0 <= metric_f <= 1.0:
            return metric_f
        logger.warning(
            f"ML Gate [{model_type}] ignoring incompatible champion metric from {source}: {metric_f}. "
            "Expected range is [0,1]."
        )
        return None

    def _record_ticker_data_failure(self, ticker: str, reason: str = ""):
        """
        Cache invalid tickers for the current run and increment DB fail_count.
        Prevents repeated yfinance 404 loops during training.
        """
        ticker_u = (ticker or "").upper().strip()
        if not ticker_u:
            return
        self._feature_cache[ticker_u] = None
        self._market_data_cache[ticker_u] = pd.DataFrame()
        if reason:
            logger.info(f"ML data skip {ticker_u}: {reason}")

        if self.dry_run:
            return
        try:
            if self.db and hasattr(self.db, "register_ticker_failure"):
                self.db.register_ticker_failure(ticker_u)
        except Exception as e:
            logger.debug(f"Failed to register ticker failure for {ticker_u}: {e}")

    @staticmethod
    def _is_missing_table_error(error: Exception, table_name: str) -> bool:
        """Detect Supabase missing table/schema errors."""
        msg = str(error).lower()
        table = table_name.lower()
        if "pgrst204" in msg and ("schema cache" in msg or "could not find" in msg):
            return True
        return (
            table in msg
            and (
                "404" in msg
                or "pgrst205" in msg
                or "not found" in msg
                or "relation" in msg
                or "does not exist" in msg
                or "schema cache" in msg
            )
        )

    def _new_training_report(self):
        """Reset per-run training report."""
        self.last_training_report = {
            "started_at": datetime.now().isoformat(),
            "validation_method": "walk_forward",
            "components": {},
            "promotions": 0,
            "rollbacks": 0,
        }

    def _record_training_component(self, component: str, report: Dict[str, Any]):
        """Store one component report in the current run summary."""
        if not self.last_training_report:
            self._new_training_report()
        self.last_training_report.setdefault("components", {})[component] = report
        if report.get("promoted"):
            self.last_training_report["promotions"] = self.last_training_report.get("promotions", 0) + 1
        if report.get("rolled_back"):
            self.last_training_report["rollbacks"] = self.last_training_report.get("rollbacks", 0) + 1

    def _walk_forward_slices(self, n_samples: int, max_folds: int = None) -> List[Tuple[slice, slice]]:
        """
        Build chronological walk-forward splits.
        Train window grows over time; each fold validates on the next chunk.
        """
        if n_samples < 8:
            return []

        max_folds = max_folds or self.WALK_FORWARD_MAX_FOLDS
        min_train = max(self.MIN_TRAINING_SAMPLES // 2, int(n_samples * self.WALK_FORWARD_MIN_TRAIN_RATIO))
        min_train = min(min_train, n_samples - 2)
        min_train = max(4, min_train)
        if min_train >= n_samples - 1:
            return []

        remaining = n_samples - min_train
        n_folds = min(max_folds, remaining)
        if n_folds <= 0:
            return []

        val_size = max(1, remaining // n_folds)
        splits: List[Tuple[slice, slice]] = []
        cursor = min_train

        while cursor < n_samples and len(splits) < n_folds:
            val_end = min(n_samples, cursor + val_size)
            splits.append((slice(0, cursor), slice(cursor, val_end)))
            cursor = val_end

        if splits:
            tr_slice, val_slice = splits[-1]
            if val_slice.stop < n_samples:
                splits[-1] = (tr_slice, slice(val_slice.start, n_samples))

        return [s for s in splits if (s[1].stop - s[1].start) > 0]

    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional hit-rate: sign(pred) vs sign(realized return)."""
        if y_true is None or y_pred is None or len(y_true) == 0:
            return 0.0
        true_dir = np.sign(np.asarray(y_true).astype(float))
        pred_dir = np.sign(np.asarray(y_pred).astype(float))
        return float(np.mean(true_dir == pred_dir))

    def _walk_forward_validate_classifier(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Walk-forward validation for classifier, with majority-class baseline."""
        splits = self._walk_forward_slices(len(X))
        fold_scores: List[float] = []
        baseline_scores: List[float] = []

        for tr_slice, val_slice in splits:
            X_train, y_train = X[tr_slice], y[tr_slice]
            X_val, y_val = X[val_slice], y[val_slice]
            if len(y_train) < 4 or len(y_val) < 1:
                continue

            try:
                candidate = PureGradientBoosting(n_estimators=50, learning_rate=0.1)
                candidate.fit(X_train, y_train)
                pred = candidate.predict(X_val)
                fold_scores.append(float(np.mean(pred == y_val)))

                # Baseline: most frequent class in train split.
                classes, counts = np.unique(y_train, return_counts=True)
                baseline_label = classes[np.argmax(counts)] if len(classes) > 0 else 1
                baseline_pred = np.full(len(y_val), baseline_label)
                baseline_scores.append(float(np.mean(baseline_pred == y_val)))
            except Exception as e:
                logger.warning(f"Walk-forward classifier fold skipped: {e}")

        return {
            "folds": len(fold_scores),
            "candidate_metric": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "baseline_metric": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
            "metric_name": "accuracy",
        }

    def _walk_forward_validate_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Walk-forward validation for regression.
        Metric: directional accuracy (higher is better), plus MAE diagnostics.
        """
        splits = self._walk_forward_slices(len(X))
        fold_scores: List[float] = []
        baseline_scores: List[float] = []
        fold_mae: List[float] = []
        baseline_mae: List[float] = []

        for tr_slice, val_slice in splits:
            X_train, y_train = X[tr_slice], y[tr_slice]
            X_val, y_val = X[val_slice], y[val_slice]
            if len(y_train) < 4 or len(y_val) < 1:
                continue

            try:
                candidate = PureGradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
                candidate.fit(X_train, y_train)
                pred = candidate.predict(X_val)

                baseline_pred = np.full(len(y_val), float(np.mean(y_train)))

                fold_scores.append(self._directional_accuracy(y_val, pred))
                baseline_scores.append(self._directional_accuracy(y_val, baseline_pred))
                fold_mae.append(float(np.mean(np.abs(y_val - pred))))
                baseline_mae.append(float(np.mean(np.abs(y_val - baseline_pred))))
            except Exception as e:
                logger.warning(f"Walk-forward regression fold skipped: {e}")

        return {
            "folds": len(fold_scores),
            "candidate_metric": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "baseline_metric": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
            "metric_name": "directional_accuracy",
            "candidate_mae": float(np.mean(fold_mae)) if fold_mae else 0.0,
            "baseline_mae": float(np.mean(baseline_mae)) if baseline_mae else 0.0,
        }

    def _walk_forward_validate_lstm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Lightweight walk-forward validation for LSTM to keep cloud runtime bounded.
        """
        splits = self._walk_forward_slices(len(X), max_folds=3)
        fold_scores: List[float] = []
        baseline_scores: List[float] = []
        fold_mae: List[float] = []
        baseline_mae: List[float] = []

        if X.ndim != 3 or y.ndim != 2:
            return {
                "folds": 0,
                "candidate_metric": 0.0,
                "baseline_metric": 0.0,
                "metric_name": "directional_accuracy",
                "candidate_mae": 0.0,
                "baseline_mae": 0.0,
            }

        input_size = int(X.shape[2])
        hidden_size = 16
        np.random.seed(42)

        for tr_slice, val_slice in splits:
            X_train, y_train = X[tr_slice], y[tr_slice]
            X_val, y_val = X[val_slice], y[val_slice]
            if len(y_train) < 4 or len(y_val) < 1:
                continue

            try:
                candidate = PureLSTM(input_size=input_size, hidden_size=hidden_size)
                candidate.fit(X_train, y_train, epochs=8, learning_rate=0.01)
                if not candidate.is_fitted:
                    continue

                raw = candidate.forward(X_val)
                pred = np.asarray(raw).reshape(-1)
                y_val_flat = np.asarray(y_val).reshape(-1)
                baseline_pred = np.full(len(y_val_flat), float(np.mean(y_train)))

                fold_scores.append(self._directional_accuracy(y_val_flat, pred))
                baseline_scores.append(self._directional_accuracy(y_val_flat, baseline_pred))
                fold_mae.append(float(np.mean(np.abs(y_val_flat - pred))))
                baseline_mae.append(float(np.mean(np.abs(y_val_flat - baseline_pred))))
            except Exception as e:
                logger.warning(f"Walk-forward LSTM fold skipped: {e}")

        return {
            "folds": len(fold_scores),
            "candidate_metric": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "baseline_metric": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
            "metric_name": "directional_accuracy",
            "candidate_mae": float(np.mean(fold_mae)) if fold_mae else 0.0,
            "baseline_mae": float(np.mean(baseline_mae)) if baseline_mae else 0.0,
        }

    def _get_champion_snapshot(self, model_type: str) -> Dict[str, Any]:
        """
        Resolve current champion metric/version.
        Priority:
        1) ml_model_registry latest active CHAMPION
        2) ml_model_state latest row for model_type
        """
        champion = {"model_version": None, "metric": None}
        if not self.db or not getattr(self.db, "supabase", None):
            return champion

        if self._ml_registry_enabled:
            try:
                reg = self.db.supabase.table("ml_model_registry") \
                    .select("model_version, candidate_metric, metric_name, is_active") \
                    .eq("model_type", model_type) \
                    .eq("role", "CHAMPION") \
                    .eq("is_active", True) \
                    .order("trained_at", desc=True) \
                    .limit(1) \
                    .execute()
                if reg.data:
                    row = reg.data[0]
                    champion["model_version"] = row.get("model_version")
                    champion["metric"] = self._normalize_metric_for_gate(
                        model_type=model_type,
                        metric=self._safe_float(row.get("candidate_metric"), None),
                        source=f"ml_model_registry/{row.get('metric_name') or 'unknown'}",
                    )
                    # If registry has a champion row, trust it as source-of-truth.
                    return champion
            except Exception as e:
                if self._is_missing_table_error(e, "ml_model_registry"):
                    self._ml_registry_enabled = False
                else:
                    logger.warning(f"Champion lookup (registry) failed: {e}")

        try:
            state = self.db.supabase.table("ml_model_state") \
                .select("model_version, accuracy") \
                .eq("model_type", model_type) \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            if state.data:
                row = state.data[0]
                champion["model_version"] = row.get("model_version")
                champion["metric"] = self._normalize_metric_for_gate(
                    model_type=model_type,
                    metric=self._safe_float(row.get("accuracy"), None),
                    source="ml_model_state/accuracy",
                )
        except Exception as e:
            logger.warning(f"Champion lookup (model_state) failed: {e}")

        return champion

    def _log_model_registry_event(self, event: Dict[str, Any]):
        """Write champion/challenger event to registry table (best effort)."""
        if self.dry_run or not self._ml_registry_enabled:
            return
        if not self.db or not getattr(self.db, "supabase", None):
            return

        model_type = event.get("model_type")
        is_champion = event.get("role") == "CHAMPION"

        try:
            if is_champion and model_type:
                self.db.supabase.table("ml_model_registry") \
                    .update({"is_active": False}) \
                    .eq("model_type", model_type) \
                    .eq("role", "CHAMPION") \
                    .eq("is_active", True) \
                    .execute()
        except Exception as e:
            if self._is_missing_table_error(e, "ml_model_registry"):
                self._ml_registry_enabled = False
                return
            logger.warning(f"ML registry pre-update failed: {e}")

        try:
            self.db.supabase.table("ml_model_registry").insert(event).execute()
        except Exception as e:
            if self._is_missing_table_error(e, "ml_model_registry"):
                self._ml_registry_enabled = False
                return
            logger.warning(f"ML registry insert failed: {e}")

    def _gate_candidate(
        self,
        model_type: str,
        candidate_metric: float,
        baseline_metric: float,
        champion_metric: Optional[float],
        min_gain: float,
        metric_name: str,
    ) -> Dict[str, Any]:
        """
        Gate model promotion:
        - Must beat walk-forward baseline by min_gain.
        - Must not degrade current champion beyond tolerated drawdown.
        """
        gain_vs_baseline = candidate_metric - baseline_metric
        champion_gap = None if champion_metric is None else candidate_metric - champion_metric
        beat_baseline = gain_vs_baseline >= min_gain
        within_champion_guard = champion_metric is None or candidate_metric >= (champion_metric - self.MAX_CHAMPION_DEGRADATION)
        promoted = beat_baseline and within_champion_guard

        if promoted:
            gate_status = "PASS"
            reason = (
                f"Pass gate: {metric_name} {candidate_metric:.3f}, "
                f"baseline {baseline_metric:.3f}, gain {gain_vs_baseline:.3f}"
            )
        else:
            gate_status = "ROLLED_BACK" if champion_metric is not None else "FAIL"
            if not beat_baseline:
                reason = (
                    f"Rejected: insufficient gain vs baseline "
                    f"({gain_vs_baseline:.3f} < {min_gain:.3f})"
                )
            else:
                reason = (
                    f"Rejected: candidate below champion tolerance "
                    f"(gap {champion_gap:.3f}, max drawdown {self.MAX_CHAMPION_DEGRADATION:.3f})"
                )

        logger.info(
            f"ML Gate [{model_type}] {gate_status} | candidate={candidate_metric:.4f} "
            f"| baseline={baseline_metric:.4f} | champion={champion_metric if champion_metric is not None else 'n/a'}"
        )
        return {
            "promoted": promoted,
            "gate_status": gate_status,
            "reason": reason,
            "gain_vs_baseline": gain_vs_baseline,
            "gap_vs_champion": champion_gap,
        }
    
    def train(self) -> bool:
        """Master training method - triggers all models for the Ensemble."""
        logger.info("ML: Starting Ensemble Training Pipeline...")
        self._new_training_report()

        # 1. Train Classification Model (Decision Path)
        class_usable = self._train_classification()

        # 2. Train Regression & LSTM Models (Value Path)
        ensemble_usable = self.train_regression()

        components = self.last_training_report.get("components", {})
        promotions = self.last_training_report.get("promotions", 0)
        rollbacks = self.last_training_report.get("rollbacks", 0)
        self.last_training_report["completed_at"] = datetime.now().isoformat()
        self.last_training_report["usable_components"] = sum(1 for c in components.values() if c.get("usable"))
        self.last_training_report["status"] = "ok" if (class_usable or ensemble_usable) else "failed"
        self.last_training_report["summary"] = (
            f"TrainML done: promoted={promotions}, rollbacks={rollbacks}, "
            f"usable_components={self.last_training_report['usable_components']}"
        )
        logger.info(self.last_training_report["summary"])

        # Success means at least one component remains usable (either promoted or champion kept).
        return bool(class_usable or ensemble_usable)

    def _train_classification(self) -> bool:
        """
        Train classifier with walk-forward validation and promotion gate.
        Automatic rollback keeps current champion if challenger fails.
        """
        had_champion = bool(self.is_ml_ready and self.model is not None)
        base_report: Dict[str, Any] = {
            "model_type": "classifier",
            "validation_method": "walk_forward",
            "promoted": False,
            "rolled_back": False,
            "usable": had_champion,
        }

        try:
            from db_handler import DBHandler
            db = self.db if (self.db and getattr(self.db, "supabase", None)) else DBHandler()
            if not getattr(self, "db", None) or not getattr(self.db, "supabase", None):
                self.db = db

            result = db.supabase.table("signal_tracking") \
                .select("ticker, pnl_percent, status, created_at") \
                .in_("status", ["WIN", "LOSS"]) \
                .order("created_at", desc=False) \
                .limit(500) \
                .execute()

            signals = result.data or []
            if len(signals) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Classifier: only {len(signals)} samples, need {self.MIN_TRAINING_SAMPLES}")
                base_report.update({
                    "status": "insufficient_data",
                    "samples_count": len(signals),
                    "usable": had_champion,
                })
                self._record_training_component("classifier", base_report)
                return had_champion

            X_rows: List[List[float]] = []
            y_rows: List[int] = []

            for sig in signals:
                ticker = sig.get("ticker")
                if not ticker:
                    continue
                features = self._get_features(ticker)
                if not features:
                    continue

                pnl = self._safe_float(sig.get("pnl_percent"), 0.0)
                status = (sig.get("status") or "").upper()
                if status == "WIN" and pnl > 0:
                    label = 2  # UP
                elif status == "LOSS" or pnl < 0:
                    label = 0  # DOWN
                else:
                    label = 1  # HOLD

                X_rows.append([features.get(f, 0.0) for f in self.FEATURE_COLUMNS])
                y_rows.append(label)

            if len(X_rows) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Classifier: only {len(X_rows)} valid feature rows")
                base_report.update({
                    "status": "insufficient_valid_rows",
                    "samples_count": len(X_rows),
                    "usable": had_champion,
                })
                self._record_training_component("classifier", base_report)
                return had_champion

            X = np.array(X_rows, dtype=float)
            y = np.array(y_rows, dtype=int)

            wf = self._walk_forward_validate_classifier(X, y)
            candidate_metric = self._safe_float(wf.get("candidate_metric"), 0.0)
            baseline_metric = self._safe_float(wf.get("baseline_metric"), 0.0)
            folds = int(wf.get("folds", 0) or 0)
            metric_name = wf.get("metric_name", "accuracy")

            # Train challenger on full sample after validation.
            candidate_model = PureGradientBoosting(n_estimators=50, learning_rate=0.1)
            candidate_model.fit(X, y)
            candidate_version = f"pure_gb_v{datetime.now().strftime('%Y%m%d_%H%M')}"

            champion = self._get_champion_snapshot("classifier")
            champion_metric = champion.get("metric")
            champion_version = champion.get("model_version")

            gate = self._gate_candidate(
                model_type="classifier",
                candidate_metric=candidate_metric,
                baseline_metric=baseline_metric,
                champion_metric=champion_metric,
                min_gain=self.CLASSIFIER_MIN_GAIN,
                metric_name=metric_name,
            )

            promoted = bool(gate.get("promoted"))
            rolled_back = bool((not promoted) and had_champion)
            usable = bool(promoted or had_champion)

            if promoted:
                self.model = candidate_model
                self.model_version = candidate_version
                self.is_ml_ready = True

                if self.dry_run:
                    logger.info("DRY_RUN: classifier challenger passed gate but persistence skipped.")
                else:
                    db.supabase.table("ml_model_state").insert({
                        "model_version": candidate_version,
                        "model_type": "classifier",
                        "accuracy": candidate_metric,
                        "samples_count": len(X),
                        "model_weights": candidate_model.to_json()
                    }).execute()
                logger.info(
                    f"Classifier promoted: {candidate_version} | "
                    f"{metric_name}={candidate_metric:.3f} | baseline={baseline_metric:.3f}"
                )
            else:
                logger.warning(
                    f"Classifier challenger rejected ({candidate_version}). "
                    f"{gate.get('reason', 'gate failed')}"
                )

            report = {
                "status": "ok" if usable else "failed",
                "model_type": "classifier",
                "model_version": candidate_version if promoted else (champion_version or self.model_version),
                "candidate_version": candidate_version,
                "champion_version": champion_version,
                "metric_name": metric_name,
                "candidate_metric": candidate_metric,
                "baseline_metric": baseline_metric,
                "champion_metric": champion_metric,
                "samples_count": len(X),
                "folds": folds,
                "gate_status": gate.get("gate_status"),
                "gate_reason": gate.get("reason"),
                "promoted": promoted,
                "rolled_back": rolled_back,
                "usable": usable,
            }
            self._record_training_component("classifier", report)

            self._log_model_registry_event({
                "model_type": "classifier",
                "model_version": candidate_version,
                "role": "CHAMPION" if promoted else "CHALLENGER",
                "is_active": promoted,
                "gate_status": gate.get("gate_status"),
                "validation_method": "walk_forward",
                "metric_name": metric_name,
                "candidate_metric": candidate_metric,
                "baseline_metric": baseline_metric,
                "champion_metric": champion_metric,
                "improvement_abs": gate.get("gain_vs_baseline"),
                "degradation_vs_champion": gate.get("gap_vs_champion"),
                "notes": gate.get("reason"),
                "replaced_version": champion_version if promoted else None,
            })

            return usable

        except Exception as e:
            logger.error(f"Classification training failed: {e}")
            base_report.update({"status": "error", "error": str(e), "usable": had_champion})
            self._record_training_component("classifier", base_report)
            return had_champion
    
    def train_lstm(self) -> bool:
        """Train LSTM with walk-forward validation and promotion gate."""
        had_champion = bool(self.is_lstm_ready and self.lstm_model is not None)
        base_report: Dict[str, Any] = {
            "model_type": "lstm",
            "validation_method": "walk_forward",
            "promoted": False,
            "rolled_back": False,
            "usable": had_champion,
        }

        try:
            from db_handler import DBHandler
            db = self.db if (self.db and getattr(self.db, "supabase", None)) else DBHandler()
            if not getattr(self, "db", None) or not getattr(self.db, "supabase", None):
                self.db = db

            result = db.supabase.table("signal_tracking") \
                .select("ticker, pnl_percent, status, created_at") \
                .in_("status", ["WIN", "LOSS"]) \
                .order("created_at", desc=False) \
                .limit(250) \
                .execute()

            signals = result.data or []
            if len(signals) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"LSTM: only {len(signals)} samples")
                base_report.update({
                    "status": "insufficient_data",
                    "samples_count": len(signals),
                    "usable": had_champion,
                })
                self._record_training_component("lstm", base_report)
                return had_champion

            X_seq: List[np.ndarray] = []
            y_seq: List[List[float]] = []
            for sig in signals:
                ticker = sig.get("ticker")
                if not ticker:
                    continue

                seq = self._get_sequence_features(ticker)
                if seq is None:
                    continue

                pnl = self._safe_float(sig.get("pnl_percent"), 0.0)
                X_seq.append(seq[0])  # remove batch dim (10, 3)
                y_seq.append([pnl])

            if len(X_seq) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"LSTM: only {len(X_seq)} valid sequences")
                base_report.update({
                    "status": "insufficient_valid_rows",
                    "samples_count": len(X_seq),
                    "usable": had_champion,
                })
                self._record_training_component("lstm", base_report)
                return had_champion

            X = np.array(X_seq, dtype=float)
            y = np.array(y_seq, dtype=float)

            wf = self._walk_forward_validate_lstm(X, y)
            candidate_metric = self._safe_float(wf.get("candidate_metric"), 0.0)
            baseline_metric = self._safe_float(wf.get("baseline_metric"), 0.0)
            folds = int(wf.get("folds", 0) or 0)
            metric_name = wf.get("metric_name", "directional_accuracy")

            # Train challenger on full sample after validation.
            input_size = int(X.shape[2])
            hidden_size = 16
            np.random.seed(42)
            candidate_model = PureLSTM(input_size, hidden_size)
            candidate_model.fit(X, y, epochs=20, learning_rate=0.01)
            if not candidate_model.is_fitted:
                raise RuntimeError("LSTM challenger training produced unstable weights")

            split_idx = max(1, int(len(X) * 0.8))
            holdout_mse = float(candidate_model.compute_loss(X[split_idx:], y[split_idx:])) if split_idx < len(X) else 0.0

            candidate_version = f"pure_lstm_v{datetime.now().strftime('%Y%m%d_%H%M')}"
            champion = self._get_champion_snapshot("lstm")
            champion_metric = champion.get("metric")
            champion_version = champion.get("model_version")

            gate = self._gate_candidate(
                model_type="lstm",
                candidate_metric=candidate_metric,
                baseline_metric=baseline_metric,
                champion_metric=champion_metric,
                min_gain=self.LSTM_MIN_GAIN,
                metric_name=metric_name,
            )

            promoted = bool(gate.get("promoted"))
            rolled_back = bool((not promoted) and had_champion)
            usable = bool(promoted or had_champion)

            if promoted:
                self.lstm_model = candidate_model
                self.lstm_model_version = candidate_version
                self.is_lstm_ready = True

                if self.dry_run:
                    logger.info("DRY_RUN: LSTM challenger passed gate but persistence skipped.")
                else:
                    db.supabase.table("ml_model_state").insert({
                        "model_version": candidate_version,
                        "model_type": "lstm",
                        "accuracy": candidate_metric,
                        "samples_count": len(X),
                        "model_weights": candidate_model.to_json()
                    }).execute()
                logger.info(
                    f"LSTM promoted: {candidate_version} | {metric_name}={candidate_metric:.3f} "
                    f"| baseline={baseline_metric:.3f} | holdout_mse={holdout_mse:.4f}"
                )
            else:
                logger.warning(
                    f"LSTM challenger rejected ({candidate_version}). "
                    f"{gate.get('reason', 'gate failed')}"
                )

            report = {
                "status": "ok" if usable else "failed",
                "model_type": "lstm",
                "model_version": candidate_version if promoted else (champion_version or self.lstm_model_version),
                "candidate_version": candidate_version,
                "champion_version": champion_version,
                "metric_name": metric_name,
                "candidate_metric": candidate_metric,
                "baseline_metric": baseline_metric,
                "champion_metric": champion_metric,
                "candidate_mae": self._safe_float(wf.get("candidate_mae"), 0.0),
                "baseline_mae": self._safe_float(wf.get("baseline_mae"), 0.0),
                "holdout_mse": holdout_mse,
                "samples_count": len(X),
                "folds": folds,
                "gate_status": gate.get("gate_status"),
                "gate_reason": gate.get("reason"),
                "promoted": promoted,
                "rolled_back": rolled_back,
                "usable": usable,
            }
            self._record_training_component("lstm", report)

            self._log_model_registry_event({
                "model_type": "lstm",
                "model_version": candidate_version,
                "role": "CHAMPION" if promoted else "CHALLENGER",
                "is_active": promoted,
                "gate_status": gate.get("gate_status"),
                "validation_method": "walk_forward",
                "metric_name": metric_name,
                "candidate_metric": candidate_metric,
                "baseline_metric": baseline_metric,
                "champion_metric": champion_metric,
                "improvement_abs": gate.get("gain_vs_baseline"),
                "degradation_vs_champion": gate.get("gap_vs_champion"),
                "notes": gate.get("reason"),
                "replaced_version": champion_version if promoted else None,
            })

            return usable

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            base_report.update({"status": "error", "error": str(e), "usable": had_champion})
            self._record_training_component("lstm", base_report)
            return had_champion

    def train_regression(self) -> bool:
        """Train regression model and LSTM model."""
        try:
            reg_usable = self._train_gb_regression()
            lstm_usable = self.train_lstm()
            return bool(reg_usable or lstm_usable)
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return False

    def _train_gb_regression(self) -> bool:
        """Train GB regressor with walk-forward validation and promotion gate."""
        had_champion = bool(self.is_regression_ready and self.regression_model is not None)
        base_report: Dict[str, Any] = {
            "model_type": "regressor",
            "validation_method": "walk_forward",
            "promoted": False,
            "rolled_back": False,
            "usable": had_champion,
        }

        try:
            from db_handler import DBHandler
            db = self.db if (self.db and getattr(self.db, "supabase", None)) else DBHandler()
            if not getattr(self, "db", None) or not getattr(self.db, "supabase", None):
                self.db = db

            result = db.supabase.table("signal_tracking") \
                .select("ticker, pnl_percent, status, created_at") \
                .in_("status", ["WIN", "LOSS"]) \
                .order("created_at", desc=False) \
                .limit(500) \
                .execute()

            signals = result.data or []
            if len(signals) < self.MIN_TRAINING_SAMPLES:
                base_report.update({
                    "status": "insufficient_data",
                    "samples_count": len(signals),
                    "usable": had_champion,
                })
                self._record_training_component("regressor", base_report)
                return had_champion

            X_rows: List[List[float]] = []
            y_rows: List[float] = []
            for sig in signals:
                ticker = sig.get("ticker")
                if not ticker:
                    continue
                features = self._get_features(ticker)
                if not features:
                    continue
                pnl = self._safe_float(sig.get("pnl_percent"), 0.0)
                X_rows.append([features.get(f, 0.0) for f in self.FEATURE_COLUMNS])
                y_rows.append(pnl)

            if len(X_rows) < self.MIN_TRAINING_SAMPLES:
                base_report.update({
                    "status": "insufficient_valid_rows",
                    "samples_count": len(X_rows),
                    "usable": had_champion,
                })
                self._record_training_component("regressor", base_report)
                return had_champion

            X = np.array(X_rows, dtype=float)
            y = np.array(y_rows, dtype=float)

            wf = self._walk_forward_validate_regression(X, y)
            candidate_metric = self._safe_float(wf.get("candidate_metric"), 0.0)
            baseline_metric = self._safe_float(wf.get("baseline_metric"), 0.0)
            folds = int(wf.get("folds", 0) or 0)
            metric_name = wf.get("metric_name", "directional_accuracy")

            # Challenger trained on full sample after validation.
            candidate_model = PureGradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
            candidate_model.fit(X, y)
            holdout_r2 = float(candidate_model.score(X[int(len(X) * 0.8):], y[int(len(X) * 0.8):])) if len(X) > 5 else 0.0
            candidate_version = f"pure_gbr_v{datetime.now().strftime('%Y%m%d_%H%M')}"

            champion = self._get_champion_snapshot("regressor")
            champion_metric = champion.get("metric")
            champion_version = champion.get("model_version")

            gate = self._gate_candidate(
                model_type="regressor",
                candidate_metric=candidate_metric,
                baseline_metric=baseline_metric,
                champion_metric=champion_metric,
                min_gain=self.REGRESSION_MIN_GAIN,
                metric_name=metric_name,
            )

            promoted = bool(gate.get("promoted"))
            rolled_back = bool((not promoted) and had_champion)
            usable = bool(promoted or had_champion)

            if promoted:
                self.regression_model = candidate_model
                self.regression_model_version = candidate_version
                self.is_regression_ready = True

                if self.dry_run:
                    logger.info("DRY_RUN: regressor challenger passed gate but persistence skipped.")
                else:
                    db.supabase.table("ml_model_state").insert({
                        "model_version": candidate_version,
                        "model_type": "regressor",
                        "accuracy": candidate_metric,
                        "samples_count": len(X),
                        "model_weights": candidate_model.to_json()
                    }).execute()
                logger.info(
                    f"Regressor promoted: {candidate_version} | {metric_name}={candidate_metric:.3f} "
                    f"| baseline={baseline_metric:.3f} | holdout_r2={holdout_r2:.3f}"
                )
            else:
                logger.warning(
                    f"Regressor challenger rejected ({candidate_version}). "
                    f"{gate.get('reason', 'gate failed')}"
                )

            report = {
                "status": "ok" if usable else "failed",
                "model_type": "regressor",
                "model_version": candidate_version if promoted else (champion_version or self.regression_model_version),
                "candidate_version": candidate_version,
                "champion_version": champion_version,
                "metric_name": metric_name,
                "candidate_metric": candidate_metric,
                "baseline_metric": baseline_metric,
                "champion_metric": champion_metric,
                "candidate_mae": self._safe_float(wf.get("candidate_mae"), 0.0),
                "baseline_mae": self._safe_float(wf.get("baseline_mae"), 0.0),
                "holdout_r2": holdout_r2,
                "samples_count": len(X),
                "folds": folds,
                "gate_status": gate.get("gate_status"),
                "gate_reason": gate.get("reason"),
                "promoted": promoted,
                "rolled_back": rolled_back,
                "usable": usable,
            }
            self._record_training_component("regressor", report)

            self._log_model_registry_event({
                "model_type": "regressor",
                "model_version": candidate_version,
                "role": "CHAMPION" if promoted else "CHALLENGER",
                "is_active": promoted,
                "gate_status": gate.get("gate_status"),
                "validation_method": "walk_forward",
                "metric_name": metric_name,
                "candidate_metric": candidate_metric,
                "baseline_metric": baseline_metric,
                "champion_metric": champion_metric,
                "improvement_abs": gate.get("gain_vs_baseline"),
                "degradation_vs_champion": gate.get("gap_vs_champion"),
                "notes": gate.get("reason"),
                "replaced_version": champion_version if promoted else None,
            })

            return usable
        except Exception as e:
            logger.error(f"GB Regression training failed: {e}")
            base_report.update({"status": "error", "error": str(e), "usable": had_champion})
            self._record_training_component("regressor", base_report)
            return had_champion
    
    def get_last_training_report(self) -> Dict[str, Any]:
        """Return the latest in-memory training summary for this process."""
        return self.last_training_report or {}

    def get_dashboard_stats(self) -> Dict:
        """Get ML stats for ensemble components."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Fetch latest for all types
            results = db.supabase.table("ml_model_state") \
                .select("*") \
                .order("trained_at", desc=True) \
                .limit(10) \
                .execute().data
            
            stats = {
                "classifier": None,
                "regressor": None,
                "lstm": None
            }
            
            for res in results:
                m_type = res.get('model_type', 'classifier') # default to classifier if null
                if not stats.get(m_type):
                    stats[m_type] = res

            pred_result = db.supabase.table("ml_predictions") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()
            
            predictions = pred_result.data if pred_result.data else []
            training_count = self.get_training_data_count()
            
            # Extract metrics
            clf = stats.get('classifier') or {}
            reg = stats.get('regressor') or {}
            lst = stats.get('lstm') or {}

            gate_events = []
            try:
                gate_events = db.supabase.table("ml_model_registry") \
                    .select("model_type, model_version, role, gate_status, candidate_metric, baseline_metric, trained_at") \
                    .order("trained_at", desc=True) \
                    .limit(6) \
                    .execute().data or []
            except Exception as gate_err:
                if not self._is_missing_table_error(gate_err, "ml_model_registry"):
                    logger.debug(f"Gate events fetch skipped: {gate_err}")
            
            return {
                "model_version": clf.get('model_version', self.model_version),
                "is_ml_ready": self.is_ml_ready,
                "accuracy": clf.get('accuracy'),
                "reg_r2": reg.get('accuracy'),
                "lstm_mse": lst.get('accuracy'),
                "last_trained": clf.get('trained_at'),
                "training_samples": clf.get('samples_count', 0),
                "available_samples": training_count,
                "recent_predictions": predictions,
                "gate_events": gate_events,
                "last_training_report": self.last_training_report,
            }
        except Exception as e:
            logger.error(f"Dashboard stats error: {e}")
            return {"model_version": self.model_version, "is_ml_ready": False}


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    is_dry_run = _env_flag("DRY_RUN")
    
    # Check if running in GitHub Actions (Remote Training Mode)
    target_chat_id = os.environ.get("TARGET_CHAT_ID")
    
    if target_chat_id:
        logger.info("Starting Remote ML Training (GitHub Actions)...")
        if not is_dry_run:
            from telegram_bot import TelegramNotifier
        
        async def run_remote_training():
            observer = RunObservability(
                "trainml",
                dry_run=is_dry_run,
                context={"target_chat_id": str(target_chat_id)},
            )
            notifier = TelegramNotifier() if not is_dry_run else None
            ml = MLPredictor()
            
            # Send start message
            if notifier:
                await notifier.send_message(target_chat_id, "⚙️ **GitHub Action:** Training ML in corso...")
            elif is_dry_run:
                logger.info("DRY_RUN: start notification skipped.")
            
            try:
                # Execute Training
                success = ml.train()
                
                if success:
                    stats = ml.get_dashboard_stats()
                    acc = stats.get('accuracy', 0) or 0
                    r2 = stats.get('reg_r2', 0) or 0
                    mse = stats.get('lstm_mse', 0) or 0
                    training_report = stats.get("last_training_report") or {}
                    promotions = training_report.get("promotions", 0)
                    rollbacks = training_report.get("rollbacks", 0)
                    components = training_report.get("components", {})

                    def _gate_summary(component_name: str) -> str:
                        comp = components.get(component_name, {}) if isinstance(components, dict) else {}
                        status = comp.get("gate_status", "N/A")
                        candidate = comp.get("candidate_metric")
                        baseline = comp.get("baseline_metric")
                        if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
                            return f"{status} ({candidate:.2f} vs base {baseline:.2f})"
                        return status
                    
                    msg = (
                        f"✅ **Training Remoto Completato!**\n\n"
                        f"🧪 **Walk-forward + Model Gating:**\n"
                        f"├ Promotions: {promotions}\n"
                        f"└ Rollback automatici: {rollbacks}\n\n"
                        f"📦 **Ensemble components:**\n"
                        f"├ Classifier: `{stats.get('model_version')}` (Acc: {acc:.1%})\n"
                        f"├ Regressor: `{ml.regression_model_version}` (Score: {r2:.1%})\n"
                        f"└ LSTM: `{ml.lstm_model_version}` (Score: {mse:.1%})\n\n"
                        f"🛡️ **Gate Status:**\n"
                        f"├ Classifier: {_gate_summary('classifier')}\n"
                        f"├ Regressor: {_gate_summary('regressor')}\n"
                        f"└ LSTM: {_gate_summary('lstm')}\n\n"
                        f"📈 Samples: {stats.get('training_samples', 0)}\n\n"
                        f"{'🧪 DRY RUN: nessun salvataggio su Supabase.' if is_dry_run else '💡 Champion aggiornato solo se gate PASS.'}"
                    )
                    observer.finalize(
                        status="success",
                        summary="TrainML completed.",
                        kpis={
                            "training_success": True,
                            "promotions": promotions,
                            "rollbacks": rollbacks,
                            "usable_components": training_report.get("usable_components", 0),
                            "classifier_accuracy": round(float(acc), 6),
                            "regressor_score": round(float(r2), 6),
                            "lstm_score": round(float(mse), 6),
                            "training_samples": stats.get("training_samples", 0),
                        },
                        context={
                            "classifier_gate": _gate_summary("classifier"),
                            "regressor_gate": _gate_summary("regressor"),
                            "lstm_gate": _gate_summary("lstm"),
                        },
                    )
                else:
                    msg = "❌ Training non utilizzabile: nessun champion disponibile dopo i controlli di gate."
                    observer.finalize(
                        status="error",
                        summary="TrainML ended with no usable champion.",
                        kpis={"training_success": False},
                    )
                
                if notifier:
                    await notifier.send_message(target_chat_id, msg)
                elif is_dry_run:
                    logger.info("DRY_RUN completion summary: %s", msg.replace("\n", " "))
                
            except Exception as e:
                logger.error(f"Remote training script error: {e}")
                observer.add_error("run_remote_training", e)
                observer.finalize(
                    status="error",
                    summary="TrainML failed with an exception.",
                    kpis={"training_success": False},
                )
                if notifier:
                    await notifier.send_message(target_chat_id, f"❌ Errore critico script: {e}")
                elif is_dry_run:
                    logger.error("DRY_RUN: error notification skipped.")
        
        asyncio.run(run_remote_training())
        
    else:
        # Local Diagnostic / Test Mode
        ml = MLPredictor()
        print(f"\n=== Pure Python ML Test ===")
        print(f"Model: {ml.model_version}")
        print(f"ML Ready: {ml.is_ml_ready}")
        print(f"Training Samples: {ml.get_training_data_count()}")
        
        result = ml.predict("BTC-USD")
        print(f"\nBTC-USD: {result.direction} ({result.confidence:.0%})")
        print(f"Is ML: {result.is_ml}")
