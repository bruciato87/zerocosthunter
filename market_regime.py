"""
Market Regime Classifier - L2 Predictive System
================================================
Classifies overall market state to adjust strategy aggressiveness.

Regimes:
- BULL: Strong uptrend (buy aggressively)
- BEAR: Strong downtrend (defensive, trim)
- SIDEWAYS: Range-bound (accumulate on dips)
- ACCUMULATION: Post-crash, smart money buying
- DISTRIBUTION: Pre-crash, smart money selling
"""

import logging
import yfinance as yf
from typing import Dict
from datetime import datetime

logger = logging.getLogger("MarketRegime")


class MarketRegimeClassifier:
    """
    Classifies the current market regime based on multiple indicators.
    
    Uses:
    - SPY trend vs SMA200 (stock market health)
    - VIX level (fear gauge)
    - BTC trend vs SMA50 (crypto/risk appetite)
    - RSI of SPY (momentum)
    """
    
    # Thresholds
    VIX_LOW = 15      # Complacency
    VIX_NORMAL = 20   # Normal
    VIX_HIGH = 25     # Fear
    VIX_EXTREME = 30  # Panic
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self.CACHE_TTL_MINUTES = 30
        logger.info("MarketRegimeClassifier: Predictive Layer Active.")
    
    def _fetch_indicator(self, ticker: str, period: str = "6mo") -> Dict:
        """Fetch price data and calculate indicators for a ticker."""
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if data.empty or len(data) < 50:
                return {"error": f"Insufficient data for {ticker}"}
            
            # Handle MultiIndex
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                close = data['Close']
            
            current_price = float(close.iloc[-1])
            
            # SMA200 (or SMA50 if not enough data)
            if len(close) >= 200:
                sma200 = float(close.rolling(200).mean().iloc[-1])
                sma_label = "SMA200"
            else:
                sma200 = float(close.rolling(50).mean().iloc[-1])
                sma_label = "SMA50"
            
            # SMA50
            sma50 = float(close.rolling(50).mean().iloc[-1])
            
            # RSI (14-day)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            # Trend direction
            price_vs_sma200 = "above" if current_price > sma200 else "below"
            price_vs_sma50 = "above" if current_price > sma50 else "below"
            
            return {
                "ticker": ticker,
                "price": current_price,
                "sma200": sma200,
                "sma50": sma50,
                "sma_label": sma_label,
                "rsi": current_rsi,
                "price_vs_sma200": price_vs_sma200,
                "price_vs_sma50": price_vs_sma50
            }
        except Exception as e:
            logger.warning(f"Failed to fetch indicator for {ticker}: {e}")
            return {"error": str(e)}
    
    def _get_vix(self) -> float:
        """Get current VIX level."""
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="5d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get VIX: {e}")
        return 20.0  # Default normal
    
    def classify(self) -> Dict:
        """
        Classify current market regime.
        
        Returns:
            {
                "regime": "BULL" | "BEAR" | "SIDEWAYS" | "ACCUMULATION" | "DISTRIBUTION",
                "confidence": 0.0-1.0,
                "signals": ["SPY > SMA200", "VIX low", ...],
                "recommendation": "aggressive" | "normal" | "defensive",
                "confidence_multiplier": 0.8-1.2
            }
        """
        # Check cache
        if self._cache and self._cache_time:
            age_minutes = (datetime.now() - self._cache_time).seconds / 60
            if age_minutes < self.CACHE_TTL_MINUTES:
                return self._cache
        
        signals = []
        bull_score = 0
        bear_score = 0
        
        # 1. SPY Analysis
        spy = self._fetch_indicator("SPY")
        if "error" not in spy:
            if spy["price_vs_sma200"] == "above":
                signals.append("SPY > SMA200 (bullish)")
                bull_score += 2
            else:
                signals.append("SPY < SMA200 (bearish)")
                bear_score += 2
            
            if spy["rsi"] > 70:
                signals.append(f"SPY RSI {spy['rsi']:.0f} (overbought)")
                bear_score += 1
            elif spy["rsi"] < 30:
                signals.append(f"SPY RSI {spy['rsi']:.0f} (oversold)")
                bull_score += 1  # Accumulation opportunity
        
        # 2. VIX Analysis
        vix = self._get_vix()
        if vix < self.VIX_LOW:
            signals.append(f"VIX {vix:.1f} (complacent)")
            bull_score += 1
        elif vix < self.VIX_NORMAL:
            signals.append(f"VIX {vix:.1f} (normal)")
        elif vix < self.VIX_HIGH:
            signals.append(f"VIX {vix:.1f} (elevated)")
            bear_score += 1
        elif vix < self.VIX_EXTREME:
            signals.append(f"VIX {vix:.1f} (fear)")
            bear_score += 2
        else:
            signals.append(f"VIX {vix:.1f} (extreme fear)")
            bear_score += 3
            bull_score += 1  # Panic = opportunity
        
        # 3. BTC Analysis (crypto/risk sentiment)
        btc = self._fetch_indicator("BTC-USD")
        if "error" not in btc:
            if btc["price_vs_sma50"] == "above":
                signals.append("BTC > SMA50 (risk-on)")
                bull_score += 1
            else:
                signals.append("BTC < SMA50 (risk-off)")
                bear_score += 1
        
        # Determine regime
        total_score = bull_score + bear_score
        if total_score == 0:
            regime = "SIDEWAYS"
            confidence = 0.5
        elif bull_score > bear_score * 2:
            regime = "BULL"
            confidence = min(0.95, 0.6 + (bull_score / 10))
        elif bear_score > bull_score * 2:
            regime = "BEAR"
            confidence = min(0.95, 0.6 + (bear_score / 10))
        elif vix > self.VIX_EXTREME and spy.get("rsi", 50) < 35:
            regime = "ACCUMULATION"
            confidence = 0.7
        elif vix < self.VIX_LOW and spy.get("rsi", 50) > 70:
            regime = "DISTRIBUTION"
            confidence = 0.65
        else:
            regime = "SIDEWAYS"
            confidence = 0.5 + abs(bull_score - bear_score) / 10
        
        # Determine recommendation and multiplier
        if regime == "BULL":
            recommendation = "aggressive"
            confidence_multiplier = 1.15
        elif regime == "BEAR":
            recommendation = "defensive"
            confidence_multiplier = 0.85
        elif regime == "ACCUMULATION":
            recommendation = "aggressive"  # Buy the dip
            confidence_multiplier = 1.10
        elif regime == "DISTRIBUTION":
            recommendation = "defensive"
            confidence_multiplier = 0.90
        else:  # SIDEWAYS
            recommendation = "normal"
            confidence_multiplier = 1.0
        
        # Determine volatility state
        if vix < self.VIX_LOW:
            vol_state = "LOW (Complacent)"
        elif vix < self.VIX_HIGH:
            vol_state = "NORMAL"
        elif vix < self.VIX_EXTREME:
            vol_state = "HIGH (Fear)"
        else:
            vol_state = "EXTREME (Panic)"

        result = {
            "regime": regime,
            "confidence": round(confidence, 2),
            "signals": signals,
            "recommendation": recommendation,
            "strategy_suggestion": self._get_educational_suggestion(regime),
            "volatility_state": vol_state,
            "confidence_multiplier": confidence_multiplier,
            "recommended_min_confidence": max(0.6, min(0.9, 0.75 / confidence_multiplier)), # Adjusted threshold
            "indicators": {
                "spy": spy if "error" not in spy else None,
                "btc": btc if "error" not in btc else None,
                "vix": vix
            },
            "bull_score": bull_score,
            "bear_score": bear_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        self._cache = result
        self._cache_time = datetime.now()
        
        logger.info(f"Market Regime: {regime} ({confidence:.0%}) - {recommendation}")
        return result
    
    def _get_educational_suggestion(self, regime: str) -> str:
        """Get an educational tip for the dashboard."""
        if regime == "BULL":
            return "Momentum is strong. Look for breakouts and buy dips in leaders."
        elif regime == "BEAR":
            return "Trend is down. Preserve capital. Cash is a position."
        elif regime == "ACCUMULATION":
            return "Smart money is buying. Look for divergence and volume bottoms."
        elif regime == "DISTRIBUTION":
            return "Smart money is selling. Tighten stops and take profits."
        else:
            return "Market is chopping. Trade ranges or wait for a clear trend."

    def get_regime(self) -> str:
        """Shortcut to get just the regime name."""
        return self.classify()["regime"]
    
    def get_multiplier(self) -> float:
        """Shortcut to get confidence multiplier for current regime."""
        return self.classify()["confidence_multiplier"]


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = MarketRegimeClassifier()
    result = classifier.classify()
    
    print("\n=== Market Regime Classification ===")
    print(f"Regime: {result['regime']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence Multiplier: {result['confidence_multiplier']}")
    print(f"\nSignals:")
    for sig in result['signals']:
        print(f"  - {sig}")
