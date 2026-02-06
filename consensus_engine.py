import logging
from typing import Dict, List, Optional

logger = logging.getLogger("ConsensusEngine")

class ConsensusEngine:
    """
    Aggregates signals from multiple agents and calculates a weighted final action.
    
    Default Weights (adjusted dynamically based on track record):
    - Council Consensus: 40%
    - ML Predictor: 30%
    - Expert Critic: 20%
    - AI Analyst (Hunter): 10%
    """

    SENTIMENT_MAP = {
        "STRONG BUY": 100,
        "BUY": 70,
        "ACCUMULATE": 50,
        "WAIT": 0,
        "HOLD": 0,
        "SKIP": 0,
        "SELL": -70,
        "PANIC SELL": -100,
        "UP": 80,
        "DOWN": -80
    }

    def __init__(self):
        self.default_weights = {
            "council": 0.40,
            "ml": 0.30,
            "critic": 0.20,
            "analyst": 0.10
        }
        self._dynamic_weights_cache = None
        self._cache_time = None

    def get_dynamic_weights(self, force_refresh: bool = False) -> Dict[str, float]:
        """
        [Phase 2] Calculate dynamic weights based on system performance.
        
        Strategy:
        - High win rate (>65%): Boost ML weight (predictions working well)
        - Low win rate (<45%): Boost Council weight (need more debate)
        - Medium: Use default weights
        
        Returns adjusted weight dict.
        """
        from datetime import datetime, timedelta
        
        # Cache for 5 minutes to avoid DB spam
        if self._dynamic_weights_cache and self._cache_time and not force_refresh:
            if datetime.now() - self._cache_time < timedelta(minutes=5):
                return self._dynamic_weights_cache
        
        try:
            from db_handler import DBHandler
            db = DBHandler()
            stats = db.get_audit_stats()
            
            win_rate = stats.get("win_rate", 0)
            total_closed = stats.get("closed", 0)
            
            # Need minimum sample size for adjustments
            if total_closed < 10:
                logger.info(f"Dynamic Weights: Insufficient data ({total_closed} trades). Using defaults.")
                return self.default_weights.copy()
            
            weights = self.default_weights.copy()
            
            if win_rate > 65:
                # System is performing well - trust ML more
                weights["ml"] = 0.40
                weights["council"] = 0.35
                weights["analyst"] = 0.05
                weights["critic"] = 0.20
                logger.info(f"Dynamic Weights: HIGH performance ({win_rate}% WR). Boosting ML to 40%.")
                
            elif win_rate < 45:
                # System underperforming - reduce ML, boost Council (more debate)
                weights["ml"] = 0.15
                weights["council"] = 0.50
                weights["analyst"] = 0.10
                weights["critic"] = 0.25
                logger.info(f"Dynamic Weights: LOW performance ({win_rate}% WR). Boosting Council to 50%.")
                
            else:
                # Medium performance - slight adjustments
                if win_rate > 55:
                    weights["ml"] = 0.35
                    weights["council"] = 0.35
                logger.info(f"Dynamic Weights: NORMAL performance ({win_rate}% WR). Using balanced weights.")
            
            # Normalize to ensure sum = 1.0
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            
            # Cache result
            self._dynamic_weights_cache = weights
            self._cache_time = datetime.now()
            
            return weights
            
        except Exception as e:
            logger.warning(f"Dynamic weights calculation failed: {e}. Using defaults.")
            return self.default_weights.copy()

    def calculate_weighted_action(self, prediction: Dict, is_owned: bool = False) -> Dict:
        """
        Calculates the final consensus score and action.
        Uses dynamic weights based on system performance.
        """
        ticker = prediction.get("ticker", "UNKNOWN")
        
        # [Phase 2] Get dynamic weights instead of static
        weights = self.get_dynamic_weights()
        
        # 1. Analyst (Hunter) Score
        analyst_sent = prediction.get("sentiment", "HOLD").upper()
        analyst_score = self.SENTIMENT_MAP.get(analyst_sent, 0)
        
        # 2. Critic Score
        critic_score_raw = prediction.get("critic_score", 50) # 0-100
        # Convert 0-100 to -100 to 100 range
        critic_score = (critic_score_raw - 50) * 2
        
        # 3. Council Score
        council_summary = prediction.get("council_summary", "")
        council_sent = "HOLD"
        if "BUY" in council_summary: council_sent = "BUY"
        elif "SELL" in council_summary: council_sent = "SELL"
        council_score = self.SENTIMENT_MAP.get(council_sent, 0)
        
        # 4. ML Score
        # Look for ML data in council_full_debate or direct keys
        ml_dir = "HOLD"
        full_debate = prediction.get("council_full_debate", "")
        if "ML agrees (UP" in full_debate or "ML Prediction: UP" in full_debate:
            ml_dir = "UP"
        elif "ML Prediction: DOWN" in full_debate:
            ml_dir = "DOWN"
            
        ml_score = self.SENTIMENT_MAP.get(ml_dir, 0)

        # Weighted Calculation with dynamic weights
        final_score = (
            (analyst_score * weights["analyst"]) +
            (critic_score * weights["critic"]) +
            (council_score * weights["council"]) +
            (ml_score * weights["ml"])
        )

        # Map Final Score to Action
        if final_score >= 60:
            final_action = "STRONG ACCUMULATE" if is_owned else "STRONG BUY"
        elif final_score >= 25:
            final_action = "ACCUMULATE" if is_owned else "BUY"
        elif final_score >= 10:
            final_action = "ACCUMULATE" if is_owned else "WATCH"
        elif final_score <= -60:
            final_action = "STRONG SELL" if is_owned else "AVOID (High Risk)"
        elif final_score <= -25:
            final_action = "SELL" if is_owned else "AVOID"
        else:
            final_action = "HOLD" if is_owned else "WATCH"

        # Check for extreme disagreement (Disputed flag)
        is_disputed = False
        if (analyst_score > 50 and ml_score < -50) or (analyst_score < -50 and ml_score > 50):
            is_disputed = True
        
        if is_disputed:
            final_action += " (Disputed)"

        logger.info(f"Consensus [{ticker}]: Final Score {final_score:.1f} -> {final_action} (Weights: ML={weights['ml']:.0%}, Council={weights['council']:.0%})")
        
        return {
            "final_action": final_action,
            "final_score": final_score,
            "is_disputed": is_disputed,
            "weights_used": weights,  # [Phase 2] Include weights in response
            "components": {
                "analyst": analyst_score,
                "critic": critic_score,
                "council": council_score,
                "ml": ml_score
            }
        }

