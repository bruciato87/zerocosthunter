"""
ML Predictor - Level 4 Machine Learning
========================================
Pure Python/NumPy Gradient Boosting for price direction prediction.
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("MLPredictor")


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


class MLPredictor:
    """
    ML Predictor using Pure Python Gradient Boosting.
    Learns from historical data, runs entirely on Vercel.
    """
    
    MIN_TRAINING_SAMPLES = 30  # Reduced from 50 for faster start
    
    FEATURE_COLUMNS = [
        'rsi_14', 'rsi_slope', 
        'macd_hist', 'macd_signal_cross',
        'bb_position', 'bb_width',
        'volume_ratio', 'volume_trend',
        'price_sma20_ratio', 'price_sma50_ratio',
        'atr_percent', 'momentum_10d',
        'vix_level', 'vix_change',
        'market_regime_encoded'
    ]
    
    def __init__(self):
        self.model: Optional[PureGradientBoosting] = None
        self.model_version = "pure_gb_v1"
        self.is_ml_ready = False
        self._load_model()
    
    def _load_model(self):
        """Load trained model from Supabase."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Check for stored model
            result = db.supabase.table("ml_model_state") \
                .select("model_version, model_weights") \
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
    
    def _get_features(self, ticker: str) -> Optional[Dict]:
        """Extract features for prediction."""
        try:
            import yfinance as yf
            import ta
            
            search_ticker = ticker
            if not any(c in ticker for c in ['-', '.']):
                search_ticker = f"{ticker}-USD"
            
            t = yf.Ticker(search_ticker)
            hist = t.history(period="3mo")
            
            if hist.empty or len(hist) < 30:
                if '-USD' in search_ticker:
                    search_ticker = ticker
                    t = yf.Ticker(search_ticker)
                    hist = t.history(period="3mo")
                    
            if hist.empty or len(hist) < 30:
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
            
            # VIX
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="5d")
                if not vix_hist.empty:
                    features['vix_level'] = vix_hist['Close'].iloc[-1]
                    features['vix_change'] = (vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[0]) / vix_hist['Close'].iloc[0] * 100 if len(vix_hist) > 1 else 0
                else:
                    features['vix_level'] = 20
                    features['vix_change'] = 0
            except:
                features['vix_level'] = 20
                features['vix_change'] = 0
            
            # Market Regime
            try:
                from market_regime import MarketRegimeClassifier
                regime = MarketRegimeClassifier().classify()
                regime_map = {'BULL': 1, 'SIDEWAYS': 0, 'BEAR': -1}
                features['market_regime_encoded'] = regime_map.get(regime.get('regime', 'SIDEWAYS'), 0)
            except:
                features['market_regime_encoded'] = 0
            
            # Handle NaN/Inf
            for key, val in features.items():
                if pd.isna(val) or np.isinf(val):
                    features[key] = 0.0
                    
            return features
            
        except Exception as e:
            logger.error(f"ML Feature extraction failed for {ticker}: {e}")
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
    
    def predict(self, ticker: str) -> MLPrediction:
        """Main prediction method."""
        features = self._get_features(ticker)
        
        if not features:
            return MLPrediction(
                ticker=ticker,
                direction="HOLD",
                confidence=0.5,
                features_used={},
                model_version="fallback",
                is_ml=False
            )
        
        # Use ML model if available, otherwise rule-based
        if self.is_ml_ready and self.model is not None:
            direction, confidence = self._ml_predict(features)
            is_ml = True
        else:
            direction, confidence = self._rule_based_predict(features)
            is_ml = False
        
        # Save prediction
        self._save_prediction(ticker, direction, confidence, features)
        
        return MLPrediction(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            features_used=features,
            model_version=self.model_version,
            is_ml=is_ml
        )
    
    def _save_prediction(self, ticker: str, direction: str, confidence: float, features: Dict):
        """Save prediction to DB."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            db.supabase.table("ml_predictions").insert({
                "ticker": ticker.upper(),
                "predicted_direction": direction,
                "ml_confidence": confidence,
                "features": json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()})
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to save prediction: {e}")
    
    def get_confidence_modifier(self, ticker: str, ai_sentiment: str) -> float:
        """Get confidence modifier based on ML agreement with AI."""
        prediction = self.predict(ticker)
        
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
    
    def train(self) -> bool:
        """Train the ML model on historical data."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Fetch closed signals
            result = db.supabase.table("signal_tracking") \
                .select("ticker, entry_price, current_price, pnl_percent, status, created_at") \
                .in_("status", ["WIN", "LOSS"]) \
                .order("created_at", desc=True) \
                .limit(500) \
                .execute()
            
            signals = result.data
            
            if len(signals) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Only {len(signals)} samples, need {self.MIN_TRAINING_SAMPLES}")
                return False
            
            # Build training data
            X = []
            y = []
            
            for sig in signals:
                ticker = sig['ticker']
                features = self._get_features(ticker)
                
                if not features:
                    continue
                
                pnl = float(sig.get('pnl_percent', 0))
                status = sig['status']
                
                if status == "WIN" and pnl > 0:
                    label = 2  # UP
                elif status == "LOSS" or pnl < 0:
                    label = 0  # DOWN
                else:
                    label = 1  # HOLD
                
                feature_row = [features.get(f, 0) for f in self.FEATURE_COLUMNS]
                X.append(feature_row)
                y.append(label)
            
            if len(X) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Only {len(X)} valid samples")
                return False
            
            # Convert to numpy
            X = np.array(X)
            y = np.array(y)
            
            # Split (simple 80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            self.model = PureGradientBoosting(n_estimators=50, learning_rate=0.1)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            
            # Save model to Supabase
            self.model_version = f"pure_gb_v{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.is_ml_ready = True
            
            db.supabase.table("ml_model_state").insert({
                "model_version": self.model_version,
                "accuracy": accuracy,
                "samples_count": len(X),
                "model_weights": self.model.to_json()
            }).execute()
            
            logger.info(f"Model trained! Version={self.model_version}, Accuracy={accuracy:.2%}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_dashboard_stats(self) -> Dict:
        """Get ML stats for dashboard."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            model_result = db.supabase.table("ml_model_state") \
                .select("*") \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            
            model_state = model_result.data[0] if model_result.data else None
            
            pred_result = db.supabase.table("ml_predictions") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()
            
            predictions = pred_result.data if pred_result.data else []
            training_count = self.get_training_data_count()
            
            return {
                "model_version": self.model_version,
                "is_ml_ready": self.is_ml_ready,
                "accuracy": model_state.get('accuracy') if model_state else None,
                "last_trained": model_state.get('trained_at') if model_state else None,
                "training_samples": model_state.get('samples_count') if model_state else 0,
                "available_samples": training_count,
                "recent_predictions": predictions
            }
        except Exception as e:
            logger.error(f"Dashboard stats error: {e}")
            return {"model_version": self.model_version, "is_ml_ready": False}


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Check if running in GitHub Actions (Remote Training Mode)
    target_chat_id = os.environ.get("TARGET_CHAT_ID")
    
    if target_chat_id:
        logger.info("Starting Remote ML Training (GitHub Actions)...")
        from telegram_bot import TelegramNotifier
        
        async def run_remote_training():
            notifier = TelegramNotifier()
            ml = MLPredictor()
            
            # Send start message
            await notifier.send_message(target_chat_id, "⚙️ **GitHub Action:** Training ML in corso...")
            
            try:
                # Execute Training
                success = ml.train()
                
                if success:
                    stats = ml.get_dashboard_stats()
                    msg = (
                        f"✅ **Training Remoto Completato!**\n\n"
                        f"📦 Modello: `{stats.get('model_version')}`\n"
                        f"📊 Accuracy: {stats.get('accuracy', 0):.1%}\n"
                        f"📈 Samples: {stats.get('training_samples', 0)}\n\n"
                        f"💡 Il modello è stato salvato su Supabase."
                    )
                else:
                    msg = "❌ Training Fallito. Controlla i log di GitHub Actions."
                
                await notifier.send_message(target_chat_id, msg)
                
            except Exception as e:
                logger.error(f"Remote training script error: {e}")
                await notifier.send_message(target_chat_id, f"❌ Errore critico script: {e}")
        
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
