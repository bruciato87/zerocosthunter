"""
Sentiment Aggregator - Unified Market Sentiment Score
======================================================
Combines multiple sentiment sources into a single actionable score.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger("SentimentAggregator")


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources into a unified market score.
    
    Score ranges 0-100:
    - 0-25: Extreme Fear (buying opportunity)
    - 25-45: Fear (cautious accumulation)
    - 45-55: Neutral
    - 55-75: Greed (cautious profit-taking)
    - 75-100: Extreme Greed (take profits)
    """
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        logger.info("SentimentAggregator: Analyzing cross-source sentiment...")
    
    def get_fear_greed_stock(self) -> Optional[int]:
        """Get CNN Fear & Greed Index for stocks."""
        try:
            from insider import Insider
            insider = Insider()
            result = insider.get_stock_fear_greed()
            # Insider returns dict {"value": 50, "classification": "Neutral"}
            if result and isinstance(result, dict):
                return result.get("value")
            return result if isinstance(result, int) else None
        except Exception as e:
            logger.warning(f"Failed to get stock Fear & Greed: {e}")
            return None
    
    def get_fear_greed_crypto(self) -> Optional[int]:
        """Get Crypto Fear & Greed Index."""
        try:
            from insider import Insider
            insider = Insider()
            result = insider.get_crypto_fear_greed()
            # Insider returns dict {"value": 50, "classification": "Neutral"}
            if result and isinstance(result, dict):
                return result.get("value")
            return result if isinstance(result, int) else None
        except Exception as e:
            logger.warning(f"Failed to get crypto Fear & Greed: {e}")
            return None
    
    def get_whale_sentiment(self) -> Optional[str]:
        """Get whale watcher status."""
        try:
            from whale_watcher import WhaleWatcher
            watcher = WhaleWatcher()
            stats = watcher.get_dashboard_stats()
            return stats.get("status", "NEUTRAL")
        except Exception as e:
            logger.warning(f"Failed to get whale sentiment: {e}")
            return None
    
    def get_vix_level(self) -> Optional[float]:
        """Get VIX (fear index) level."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
        except Exception as e:
            logger.warning(f"Failed to get VIX: {e}")
        return None
    
    def _whale_to_score(self, status: str) -> int:
        """Convert whale status to score contribution."""
        status = status.upper() if status else "NEUTRAL"
        mapping = {
            "BULLISH": 65,
            "ACCUMULATION": 60,
            "NEUTRAL": 50,
            "DISTRIBUTION": 40,
            "BEARISH": 35,
        }
        return mapping.get(status, 50)
    
    def _vix_to_score(self, vix: float) -> int:
        """Convert VIX level to score contribution (inverted - high VIX = fear)."""
        if vix is None:
            return 50
        if vix > 30:
            return 20  # Extreme fear
        elif vix > 25:
            return 35  # Fear
        elif vix > 20:
            return 45  # Mild concern
        elif vix > 15:
            return 55  # Normal
        else:
            return 70  # Complacency/Greed
    
    def get_aggregated_score(self) -> Dict:
        """
        Calculate aggregated sentiment score from all sources.
        
        Returns:
            {
                "score": 0-100,
                "label": "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed",
                "recommendation": "BUY" / "ACCUMULATE" / "HOLD" / "TRIM" / "SELL",
                "components": {
                    "stock_fear_greed": 45,
                    "crypto_fear_greed": 30,
                    "whale_status": "NEUTRAL",
                    "vix": 22.5
                }
            }
        """
        # Gather components
        stock_fg = self.get_fear_greed_stock()
        crypto_fg = self.get_fear_greed_crypto()
        whale_status = self.get_whale_sentiment()
        vix = self.get_vix_level()
        
        # Convert to scores
        scores = []
        weights = []
        
        if stock_fg is not None:
            scores.append(stock_fg)
            weights.append(0.25)
        
        if crypto_fg is not None:
            scores.append(crypto_fg)
            weights.append(0.25)  # Reduced from 0.30 for balanced portfolio
        
        if whale_status:
            scores.append(self._whale_to_score(whale_status))
            weights.append(0.25)
        
        if vix is not None:
            scores.append(self._vix_to_score(vix))
            weights.append(0.25)  # Increased from 0.20 for mixed portfolio
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            final_score = 50  # Default neutral
        
        final_score = int(round(final_score))
        
        # Determine label and recommendation
        if final_score <= 25:
            label = "Extreme Fear"
            recommendation = "BUY"
        elif final_score <= 45:
            label = "Fear"
            recommendation = "ACCUMULATE"
        elif final_score <= 55:
            label = "Neutral"
            recommendation = "HOLD"
        elif final_score <= 75:
            label = "Greed"
            recommendation = "TRIM"
        else:
            label = "Extreme Greed"
            recommendation = "SELL"
        
        result = {
            "score": final_score,
            "label": label,
            "recommendation": recommendation,
            "components": {
                "stock_fear_greed": stock_fg,
                "crypto_fear_greed": crypto_fg,
                "whale_status": whale_status,
                "vix": round(vix, 2) if vix else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Aggregated Sentiment: {final_score} ({label}) → {recommendation}")
        return result
    
    def get_score(self) -> int:
        """Shortcut to get just the score."""
        return self.get_aggregated_score()["score"]
    
    def should_boost_buy(self) -> bool:
        """Return True if market sentiment suggests good buying opportunity."""
        score = self.get_score()
        return score <= 35
    
    def should_take_profit(self) -> bool:
        """Return True if market sentiment suggests taking profits."""
        score = self.get_score()
        return score >= 70
    
    def adjust_confidence(self, original_confidence: float, sentiment: str) -> float:
        """
        Adjust signal confidence based on market sentiment.
        
        - In extreme fear, boost BUY confidence
        - In extreme greed, reduce BUY confidence
        """
        score = self.get_score()
        
        if sentiment in ["BUY", "ACCUMULATE"]:
            if score <= 25:
                # Extreme fear = great buy opportunity
                return min(1.0, original_confidence * 1.15)
            elif score >= 75:
                # Extreme greed = reduce buy confidence
                return original_confidence * 0.85
        elif sentiment in ["SELL", "TRIM"]:
            if score >= 75:
                # Extreme greed = boost sell confidence
                return min(1.0, original_confidence * 1.10)
            elif score <= 25:
                # Extreme fear = reduce sell confidence
                return original_confidence * 0.80
        
        return original_confidence
    
    def get_numeric_scores(self, is_crypto: bool = False) -> Dict:
        """
        [Phase 2] Get individual sentiment scores for ML feature extraction.
        
        Returns:
            {
                "fear_greed_score": 0-100 (market-wide fear/greed),
                "whale_activity_score": -100 to +100 (accumulation vs distribution),
                "vix_normalized": 0-100 (inverted VIX level)
            }
        """
        # Get raw components
        stock_fg = self.get_fear_greed_stock()
        crypto_fg = self.get_fear_greed_crypto()
        whale_status = self.get_whale_sentiment()
        vix = self.get_vix_level()
        
        # Fear/Greed score - use crypto or stock based on asset type
        if is_crypto:
            fear_greed = crypto_fg if crypto_fg is not None else (stock_fg if stock_fg is not None else 50)
        else:
            fear_greed = stock_fg if stock_fg is not None else (crypto_fg if crypto_fg is not None else 50)
        
        # Whale activity score: -100 (extreme selling) to +100 (extreme buying)
        whale_mapping = {
            "BULLISH": 80,
            "ACCUMULATION": 50,
            "NEUTRAL": 0,
            "DISTRIBUTION": -50,
            "BEARISH": -80,
        }
        whale_score = whale_mapping.get((whale_status or "NEUTRAL").upper(), 0)
        
        # VIX normalized (inverted): low VIX = high score (greed), high VIX = low score (fear)
        vix_normalized = self._vix_to_score(vix) if vix else 50
        
        return {
            "fear_greed_score": fear_greed,
            "whale_activity_score": whale_score,
            "vix_normalized": vix_normalized
        }


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    aggregator = SentimentAggregator()
    
    print("\n=== Aggregated Sentiment Score ===")
    result = aggregator.get_aggregated_score()
    print(f"Score: {result['score']}")
    print(f"Label: {result['label']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Components: {result['components']}")
    
    print("\n=== Confidence Adjustments ===")
    print(f"BUY 80% adjusted: {aggregator.adjust_confidence(0.80, 'BUY'):.2%}")
    print(f"SELL 75% adjusted: {aggregator.adjust_confidence(0.75, 'SELL'):.2%}")
