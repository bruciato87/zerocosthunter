import logging
from typing import Dict, List, Optional

logger = logging.getLogger("ConsensusEngine")

class ConsensusEngine:
    """
    Aggregates signals from multiple agents and calculates a weighted final action.
    Weights:
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
        self.weights = {
            "council": 0.40,
            "ml": 0.30,
            "critic": 0.20,
            "analyst": 0.10
        }

    def calculate_weighted_action(self, prediction: Dict) -> Dict:
        """
        Calculates the final consensus score and action.
        """
        ticker = prediction.get("ticker", "UNKNOWN")
        
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

        # Weighted Calculation
        final_score = (
            (analyst_score * self.weights["analyst"]) +
            (critic_score * self.weights["critic"]) +
            (council_score * self.weights["council"]) +
            (ml_score * self.weights["ml"])
        )

        # Map Final Score to Action
        if final_score >= 60:
            final_action = "STRONG BUY"
        elif final_score >= 25:
            final_action = "BUY"
        elif final_score >= 10:
            final_action = "ACCUMULATE"
        elif final_score <= -60:
            final_action = "STRONG SELL"
        elif final_score <= -25:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # Check for extreme disagreement (Disputed flag)
        is_disputed = False
        if (analyst_score > 50 and ml_score < -50) or (analyst_score < -50 and ml_score > 50):
            is_disputed = True
        
        if is_disputed:
            final_action += " (Disputed)"

        logger.info(f"Consensus [{ticker}]: Final Score {final_score:.1f} -> {final_action}")
        
        return {
            "final_action": final_action,
            "final_score": final_score,
            "is_disputed": is_disputed,
            "components": {
                "analyst": analyst_score,
                "critic": critic_score,
                "council": council_score,
                "ml": ml_score
            }
        }
