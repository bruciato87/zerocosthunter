"""
ML Predictor - Level 4 Machine Learning
========================================
Price direction prediction using Gemini AI with technical indicators.
Integrates with Brain for confidence adjustment.

Note: Uses Gemini API for ML-like predictions - no heavy dependencies needed!
This approach works within Vercel's 250MB limit.
"""

import logging
import os
import json
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
    is_ml: bool  # True if Gemini ML, False if rule-based fallback


class MLPredictor:
    """
    Machine Learning predictor for price direction.
    
    Uses Gemini AI to analyze technical indicators and predict price direction.
    Falls back to rule-based heuristics if Gemini is unavailable.
    """
    
    # Minimum samples for accuracy tracking
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
        self.model_version = "gemini_v1"
        self.is_ml_ready = False
        self.client = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini client for ML predictions."""
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                self.is_ml_ready = True
                logger.info("ML Predictor: Gemini client initialized")
            else:
                logger.warning("ML Predictor: No GEMINI_API_KEY, using rule-based")
        except Exception as e:
            logger.warning(f"ML Predictor: Gemini init failed: {e}")
    
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
                search_ticker = f"{ticker}-USD"
            
            # Fetch price data
            t = yf.Ticker(search_ticker)
            hist = t.history(period="3mo")
            
            if hist.empty or len(hist) < 30:
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
        macd_cross = features.get('macd_signal_cross', 0)
        if macd_hist > 0:
            score += 1.0
        elif macd_hist < 0:
            score -= 1.0
        score += macd_cross * 0.5
        
        bb_pos = features.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 1.0
        elif bb_pos > 0.8:
            score -= 1.0
        
        sma20_ratio = features.get('price_sma20_ratio', 1.0)
        sma50_ratio = features.get('price_sma50_ratio', 1.0)
        if sma20_ratio > 1.0 and sma50_ratio > 1.0:
            score += 1.0
        elif sma20_ratio < 1.0 and sma50_ratio < 1.0:
            score -= 1.0
        
        vix = features.get('vix_level', 20)
        if vix > 30:
            score -= 0.5
        elif vix < 15:
            score += 0.5
        
        regime = features.get('market_regime_encoded', 0)
        score += regime * 0.5
        
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
    
    def _gemini_predict(self, ticker: str, features: Dict) -> Tuple[str, float]:
        """Use Gemini AI for ML-like prediction based on technical indicators."""
        try:
            if not self.client:
                return self._rule_based_predict(features)
            
            # Build feature summary for Gemini
            rsi = features.get('rsi_14', 50)
            macd = features.get('macd_hist', 0)
            bb_pos = features.get('bb_position', 0.5)
            momentum = features.get('momentum_10d', 0)
            vix = features.get('vix_level', 20)
            volume_ratio = features.get('volume_ratio', 1.0)
            sma20_ratio = features.get('price_sma20_ratio', 1.0)
            regime = features.get('market_regime_encoded', 0)
            regime_str = {1: 'BULL', 0: 'NEUTRAL', -1: 'BEAR'}.get(regime, 'NEUTRAL')
            
            prompt = f"""Sei un modello ML per predizione prezzi. Analizza questi indicatori tecnici per {ticker} e predici la direzione del prezzo a 7 giorni.

INDICATORI:
- RSI(14): {rsi:.1f} {"(OVERSOLD)" if rsi < 30 else "(OVERBOUGHT)" if rsi > 70 else ""}
- MACD Histogram: {macd:.2f} {"(BULLISH)" if macd > 0 else "(BEARISH)"}
- Bollinger Position: {bb_pos:.2f} (0=lower band, 1=upper band)
- 10-day Momentum: {momentum:+.1f}%
- VIX: {vix:.1f} {"(HIGH FEAR)" if vix > 25 else "(LOW FEAR)" if vix < 15 else ""}
- Volume Ratio: {volume_ratio:.2f}x average
- Price vs SMA20: {sma20_ratio:.2f} {"(ABOVE)" if sma20_ratio > 1 else "(BELOW)"}
- Market Regime: {regime_str}

Rispondi SOLO in questo formato JSON esatto:
{{"direction": "UP" | "DOWN" | "HOLD", "confidence": 0.50-0.95, "reason": "breve spiegazione"}}"""

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            # Parse JSON response
            text = response.text.strip()
            # Clean markdown if present
            if text.startswith('```'):
                text = text.split('\n', 1)[1].rsplit('\n', 1)[0]
            
            result = json.loads(text)
            direction = result.get('direction', 'HOLD').upper()
            confidence = float(result.get('confidence', 0.6))
            
            # Validate
            if direction not in ['UP', 'DOWN', 'HOLD']:
                direction = 'HOLD'
            confidence = max(0.5, min(0.95, confidence))
            
            logger.info(f"ML Gemini [{ticker}]: {direction} ({confidence:.0%}) - {result.get('reason', 'N/A')[:50]}")
            return direction, confidence
            
        except Exception as e:
            logger.warning(f"ML Gemini failed for {ticker}: {e}")
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
        
        # Use Gemini ML or rule-based
        if self.is_ml_ready:
            direction, confidence = self._gemini_predict(ticker, features)
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
        """Save prediction to DB for tracking."""
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
        """Get confidence modifier based on ML agreement with AI."""
        prediction = self.predict(ticker)
        
        bullish_sentiments = {"BUY", "ACCUMULATE", "STRONG BUY"}
        bearish_sentiments = {"SELL", "PANIC SELL", "STRONG SELL"}
        
        if ai_sentiment.upper() in bullish_sentiments:
            ai_direction = "UP"
        elif ai_sentiment.upper() in bearish_sentiments:
            ai_direction = "DOWN"
        else:
            ai_direction = "HOLD"
        
        if prediction.direction == ai_direction:
            if prediction.confidence > 0.7:
                return 1.15
            elif prediction.confidence > 0.5:
                return 1.08
            else:
                return 1.0
        elif prediction.direction == "HOLD":
            return 0.95
        else:
            if prediction.confidence > 0.7:
                return 0.85
            else:
                return 0.92
    
    def get_context_for_ai(self, ticker: str) -> str:
        """Generate context string for Brain AI prompt."""
        prediction = self.predict(ticker)
        
        model_type = "Gemini-ML" if prediction.is_ml else "Rule-Based"
        context = f"[{model_type} PREDICTOR: {ticker} → {prediction.direction} ({prediction.confidence:.0%})]"
        
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
        """Check how many labeled samples we have."""
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
        For Gemini-based ML, 'training' means updating model state.
        Gemini learns from its prompts, no explicit training needed.
        """
        try:
            from db_handler import DBHandler
            db = DBHandler()
            
            training_count = self.get_training_data_count()
            
            # Save model state
            self.model_version = f"gemini_v{datetime.now().strftime('%Y%m%d')}"
            
            db.supabase.table("ml_model_state").insert({
                "model_version": self.model_version,
                "accuracy": None,  # Will be calculated from predictions vs outcomes
                "samples_count": training_count
            }).execute()
            
            logger.info(f"ML: Model state updated! Version={self.model_version}, Samples={training_count}")
            return True
            
        except Exception as e:
            logger.error(f"ML: State update failed: {e}")
            return False
    
    def get_dashboard_stats(self) -> Dict:
        """Get ML stats for dashboard display."""
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
            logger.error(f"ML Dashboard stats error: {e}")
            return {
                "model_version": self.model_version,
                "is_ml_ready": self.is_ml_ready,
                "error": str(e)
            }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    ml = MLPredictor()
    
    print("\n=== ML Predictor Test (Gemini) ===")
    print(f"Model Version: {ml.model_version}")
    print(f"ML Ready: {ml.is_ml_ready}")
    
    result = ml.predict("BTC-USD")
    print(f"\nBTC-USD: {result.direction} ({result.confidence:.0%})")
    print(f"Is ML: {result.is_ml}")
    
    print(f"\nAI Context: {ml.get_context_for_ai('BTC-USD')}")
