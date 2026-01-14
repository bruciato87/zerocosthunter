"""
ML Predictor - Level 4 Machine Learning
========================================
Price direction prediction using XGBoost with technical indicators.
Integrates with Brain for confidence adjustment.
"""

import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger("MLPredictor")


@dataclass
class MLPrediction:
    """Structured ML prediction result."""
    ticker: str
    direction: str  # UP, DOWN, HOLD
    confidence: float  # 0.0 - 1.0
    features_used: Dict
    model_version: str
    is_ml: bool  # True if ML, False if rule-based fallback


class MLPredictor:
    """
    Machine Learning predictor for price direction.
    
    Uses XGBoost when sufficient training data is available,
    falls back to rule-based heuristics otherwise.
    """
    
    # Minimum samples required for ML training
    MIN_TRAINING_SAMPLES = 50
    
    # Feature definitions
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
        self.model = None
        self.model_version = "rule_v1"  # Start with rule-based
        self.is_ml_ready = False
        self._load_model()
    
    def _load_model(self):
        """Load trained model from DB or fallback to rules."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Check if we have a trained model in DB
            result = db.supabase.table("ml_model_state") \
                .select("*") \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            
            if result.data:
                state = result.data[0]
                self.model_version = state.get('model_version', 'rule_v1')
                self.is_ml_ready = 'xgb' in self.model_version.lower()
                logger.info(f"ML Predictor: Loaded model {self.model_version}")
            else:
                logger.info("ML Predictor: No trained model found, using rule-based")
                
        except Exception as e:
            logger.warning(f"ML Predictor: Model load failed, using rules: {e}")
    
    def _get_features(self, ticker: str) -> Optional[Dict]:
        """
        Extract features for prediction.
        Returns dict of feature values or None if insufficient data.
        """
        try:
            import yfinance as yf
            import ta
            
            # Normalize ticker
            search_ticker = ticker
            if not any(c in ticker for c in ['-', '.']):
                # Try crypto first
                search_ticker = f"{ticker}-USD"
            
            # Fetch price data
            t = yf.Ticker(search_ticker)
            hist = t.history(period="3mo")
            
            if hist.empty or len(hist) < 30:
                # Fallback to stock format
                if '-USD' in search_ticker:
                    search_ticker = ticker
                    t = yf.Ticker(search_ticker)
                    hist = t.history(period="3mo")
                    
            if hist.empty or len(hist) < 30:
                logger.warning(f"ML: Insufficient data for {ticker}")
                return None
            
            df = hist.copy()
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # Calculate indicators
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
            
            # ATR (Volatility)
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
            features['atr_percent'] = (atr.average_true_range().iloc[-1] / current_price * 100) if current_price > 0 else 0
            
            # Momentum
            if len(close) >= 10:
                features['momentum_10d'] = (current_price - close.iloc[-10]) / close.iloc[-10] * 100
            else:
                features['momentum_10d'] = 0
            
            # VIX (Market Fear)
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="5d")
                if not vix_hist.empty:
                    features['vix_level'] = vix_hist['Close'].iloc[-1]
                    features['vix_change'] = (vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[0]) / vix_hist['Close'].iloc[0] * 100 if len(vix_hist) > 1 else 0
                else:
                    features['vix_level'] = 20  # Default neutral
                    features['vix_change'] = 0
            except:
                features['vix_level'] = 20
                features['vix_change'] = 0
            
            # Market Regime (encoded: 1=BULL, 0=NEUTRAL, -1=BEAR)
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
        """
        Rule-based fallback predictor.
        Returns (direction, confidence).
        """
        score = 0.0
        max_score = 7.0  # Maximum possible score
        
        # RSI Rules
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            score += 1.5  # Oversold = bullish
        elif rsi > 70:
            score -= 1.5  # Overbought = bearish
        elif rsi < 45:
            score += 0.5
        elif rsi > 55:
            score -= 0.5
        
        # MACD Rules
        macd_hist = features.get('macd_hist', 0)
        macd_cross = features.get('macd_signal_cross', 0)
        if macd_hist > 0:
            score += 1.0
        elif macd_hist < 0:
            score -= 1.0
        score += macd_cross * 0.5  # Bonus for fresh cross
        
        # Bollinger Position
        bb_pos = features.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 1.0  # Near lower band = bullish
        elif bb_pos > 0.8:
            score -= 1.0  # Near upper band = bearish
        
        # Price vs SMA
        sma20_ratio = features.get('price_sma20_ratio', 1.0)
        sma50_ratio = features.get('price_sma50_ratio', 1.0)
        if sma20_ratio > 1.0 and sma50_ratio > 1.0:
            score += 1.0  # Above both = bullish
        elif sma20_ratio < 1.0 and sma50_ratio < 1.0:
            score -= 1.0  # Below both = bearish
        
        # VIX (inverse relationship)
        vix = features.get('vix_level', 20)
        if vix > 30:
            score -= 0.5  # High fear = cautious
        elif vix < 15:
            score += 0.5  # Low fear = bullish
        
        # Market Regime
        regime = features.get('market_regime_encoded', 0)
        score += regime * 0.5
        
        # Momentum
        momentum = features.get('momentum_10d', 0)
        if momentum > 5:
            score += 0.5
        elif momentum < -5:
            score -= 0.5
        
        # Normalize score to direction and confidence
        normalized = score / max_score  # -1 to 1 range
        
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
    
    def _xgb_predict(self, features: Dict) -> Tuple[str, float]:
        """
        XGBoost ML prediction.
        Requires trained model.
        """
        try:
            # Convert features to array in correct order
            feature_array = np.array([[features.get(f, 0) for f in self.FEATURE_COLUMNS]])
            
            # Get prediction probabilities
            probas = self.model.predict_proba(feature_array)[0]
            prediction = self.model.predict(feature_array)[0]
            
            direction_map = {0: "DOWN", 1: "HOLD", 2: "UP"}
            direction = direction_map.get(prediction, "HOLD")
            confidence = float(max(probas))
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._rule_based_predict(features)
    
    def predict(self, ticker: str) -> MLPrediction:
        """
        Main prediction method.
        Returns MLPrediction dataclass with direction, confidence, and metadata.
        """
        # Extract features
        features = self._get_features(ticker)
        
        if not features:
            # Return neutral prediction if features unavailable
            return MLPrediction(
                ticker=ticker,
                direction="HOLD",
                confidence=0.5,
                features_used={},
                model_version="fallback",
                is_ml=False
            )
        
        # Use ML or rule-based
        if self.is_ml_ready and self.model is not None:
            direction, confidence = self._xgb_predict(features)
            is_ml = True
        else:
            direction, confidence = self._rule_based_predict(features)
            is_ml = False
        
        # Save prediction to DB for tracking
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
        """Save prediction to DB for tracking accuracy."""
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
            logger.warning(f"ML: Failed to save prediction: {e}")
    
    def get_confidence_modifier(self, ticker: str, ai_sentiment: str) -> float:
        """
        Get confidence modifier based on ML agreement with AI.
        
        Args:
            ticker: Asset ticker
            ai_sentiment: Brain's sentiment (BUY, SELL, HOLD, etc.)
        
        Returns:
            Modifier between 0.85 and 1.15
        """
        prediction = self.predict(ticker)
        
        # Map AI sentiment to direction
        bullish_sentiments = {"BUY", "ACCUMULATE", "STRONG BUY"}
        bearish_sentiments = {"SELL", "PANIC SELL", "STRONG SELL"}
        
        if ai_sentiment.upper() in bullish_sentiments:
            ai_direction = "UP"
        elif ai_sentiment.upper() in bearish_sentiments:
            ai_direction = "DOWN"
        else:
            ai_direction = "HOLD"
        
        # Calculate modifier based on agreement
        if prediction.direction == ai_direction:
            # Agreement: boost confidence
            if prediction.confidence > 0.7:
                return 1.15
            elif prediction.confidence > 0.5:
                return 1.08
            else:
                return 1.0
        elif prediction.direction == "HOLD":
            # ML uncertain: slight reduction
            return 0.95
        else:
            # Disagreement: reduce confidence
            if prediction.confidence > 0.7:
                return 0.85
            else:
                return 0.92
    
    def get_context_for_ai(self, ticker: str) -> str:
        """
        Generate context string for Brain AI prompt.
        """
        prediction = self.predict(ticker)
        
        model_type = "ML" if prediction.is_ml else "Rule-Based"
        
        context = f"[{model_type} PREDICTOR: {ticker} → {prediction.direction} ({prediction.confidence:.0%} confidence)]"
        
        # Add key feature insights
        features = prediction.features_used
        if features:
            insights = []
            
            rsi = features.get('rsi_14', 50)
            if rsi < 30:
                insights.append("RSI oversold")
            elif rsi > 70:
                insights.append("RSI overbought")
            
            bb_pos = features.get('bb_position', 0.5)
            if bb_pos < 0.2:
                insights.append("near lower Bollinger")
            elif bb_pos > 0.8:
                insights.append("near upper Bollinger")
            
            if insights:
                context += f" Key: {', '.join(insights)}"
        
        return context
    
    def get_training_data_count(self) -> int:
        """Check how many labeled samples we have for training."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            result = db.supabase.table("signal_tracking") \
                .select("id", count="exact") \
                .in_("status", ["WIN", "LOSS"]) \
                .execute()
            
            return result.count if result.count else 0
            
        except Exception as e:
            logger.error(f"ML: Failed to count training data: {e}")
            return 0
    
    def train(self) -> bool:
        """
        Train XGBoost model on historical signal data.
        Requires MIN_TRAINING_SAMPLES closed signals.
        """
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Fetch closed signals with outcomes
            result = db.supabase.table("signal_tracking") \
                .select("ticker, entry_price, current_price, pnl_percent, status, created_at") \
                .in_("status", ["WIN", "LOSS"]) \
                .order("created_at", desc=True) \
                .limit(500) \
                .execute()
            
            signals = result.data
            
            if len(signals) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"ML: Only {len(signals)} samples, need {self.MIN_TRAINING_SAMPLES}")
                return False
            
            # Extract features for each signal (historical)
            X = []
            y = []
            
            for sig in signals:
                ticker = sig['ticker']
                features = self._get_features(ticker)
                
                if not features:
                    continue
                
                # Label: 2=UP (WIN with positive PnL), 0=DOWN (WIN with negative or LOSS)
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
                logger.warning(f"ML: Only {len(X)} valid samples after feature extraction")
                return False
            
            # Train XGBoost
            from xgboost import XGBClassifier
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                np.array(X), np.array(y), test_size=0.2, random_state=42
            )
            
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=3,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.model.score(X_test, y_test)
            
            # Save model state to DB
            self.model_version = f"xgb_v{datetime.now().strftime('%Y%m%d')}"
            self.is_ml_ready = True
            
            db.supabase.table("ml_model_state").insert({
                "model_version": self.model_version,
                "accuracy": accuracy,
                "samples_count": len(X)
            }).execute()
            
            logger.info(f"ML: Model trained! Version={self.model_version}, Accuracy={accuracy:.2%}, Samples={len(X)}")
            return True
            
        except ImportError:
            logger.error("ML: xgboost/sklearn not installed. Run: pip install xgboost scikit-learn")
            return False
        except Exception as e:
            logger.error(f"ML: Training failed: {e}")
            return False
    
    def get_dashboard_stats(self) -> Dict:
        """Get ML stats for dashboard display."""
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            # Model state
            model_result = db.supabase.table("ml_model_state") \
                .select("*") \
                .order("trained_at", desc=True) \
                .limit(1) \
                .execute()
            
            model_state = model_result.data[0] if model_result.data else None
            
            # Recent predictions
            pred_result = db.supabase.table("ml_predictions") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()
            
            predictions = pred_result.data if pred_result.data else []
            
            # Training data count
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
            logger.error(f"ML Dashboard stats error: {e}")
            return {
                "model_version": self.model_version,
                "is_ml_ready": False,
                "error": str(e)
            }


# CLI Test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    ml = MLPredictor()
    
    print("\n=== ML Predictor Test ===")
    print(f"Model Version: {ml.model_version}")
    print(f"ML Ready: {ml.is_ml_ready}")
    print(f"Training Data Count: {ml.get_training_data_count()}")
    
    # Test prediction
    print("\n--- Testing BTC-USD ---")
    result = ml.predict("BTC-USD")
    print(f"Direction: {result.direction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Is ML: {result.is_ml}")
    
    # Test modifier
    print(f"\nModifier for BUY signal: {ml.get_confidence_modifier('BTC-USD', 'BUY'):.2f}")
    print(f"Modifier for SELL signal: {ml.get_confidence_modifier('BTC-USD', 'SELL'):.2f}")
    
    # Test AI context
    print(f"\nAI Context: {ml.get_context_for_ai('BTC-USD')}")
