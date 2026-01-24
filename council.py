import logging
import json
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Council")

class Council:
    """
    Phase C: The Council (Multi-Agent Consensus Engine).
    Orchestrates specialized personas to debate and validate trading signals.
    """
    
    PERSONAS = {
        "THE_BULL": {
            "role": "Optimistic Growth Analyst",
            "focus": "Growth catalysts, bullish momentum, high-level potential, and fundamental 'why buy' reasons.",
            "bias": "BULLISH"
        },
        "THE_BEAR": {
            "role": "Risk Management Skeptic",
            "focus": "Macro risks, valuation bubbles, technical breakdowns, and 'why this will fail'.",
            "bias": "BEARISH"
        },
        "THE_QUANT": {
            "role": "Technical Data Scientist",
            "focus": "RSI/MACD confluence, liquidity depth, statistical probability, and volume profile.",
            "bias": "NEUTRAL/DATA-DRIVEN"
        }
    }

    def __init__(self, brain_instance=None):
        self.brain = brain_instance

    async def get_consensus(self, ticker: str, initial_signal: Dict) -> Dict:
        """
        Runs the council debate for a specific signal.
        ticker: Asset ticker
        initial_signal: Data from the initial Brain analysis
        """
        if not self.brain:
            logger.warning("Council: Brain instance missing, simplified consensus used.")
            return initial_signal

        results = {}
        votes = []

        logger.info(f"🏛️ THE COUNCIL: Debating {ticker}...")

        # 1. Run Personas (Parallelized prompt augmentation)
        for name, profile in self.PERSONAS.items():
            try:
                prompt = self._build_persona_prompt(ticker, profile, initial_signal)
                # Call brain with specific persona instruction
                response = self.brain._generate_with_fallback(prompt, json_mode=True, task_type="analyze", prefer_direct=True)
                
                # Simple parsing of persona output
                persona_data = json.loads(response)
                results[name] = persona_data
                votes.append(persona_data.get("sentiment", "HOLD"))
                
                logger.debug(f"Council [{name}]: {persona_data.get('sentiment')} ({persona_data.get('confidence', 0):.0%})")
            except Exception as e:
                logger.error(f"Council Agent {name} failed: {e}")

        # 2. Consensus Logic (2/3 Majority)
        verdict = self._calculate_verdict(votes, results, initial_signal)
        
        logger.info(f"🏛️ COUNCIL VERDICT [{ticker}]: {verdict['sentiment']} ({verdict['consensus_score']}/3 Agreement)")
        return verdict

    def _build_persona_prompt(self, ticker: str, profile: Dict, signal: Dict) -> str:
        return f"""
        YOU ARE: {profile['role']}
        YOUR FOCUS: {profile['focus']}
        
        ASSET: {ticker}
        INITIAL ANALYSIS: {signal.get('sentiment')} with {signal.get('confidence', 0)*100:.0f}% confidence.
        REASONING: {signal.get('reasoning')}
        
        TASK:
        Debate this signal based ONLY on your focus area. 
        Provide a JSON response:
        {{
            "sentiment": "BUY|SELL|HOLD",
            "confidence": 0.0-1.0,
            "argument": "Your specific reasons based on your persona"
        }}
        """

    def _calculate_verdict(self, votes: List[str], results: Dict, original: Dict) -> Dict:
        """Determines the final consensus and aggregates reasoning."""
        from collections import Counter
        counts = Counter(votes)
        
        # Find the most common sentiment
        common_sentiment, count = counts.most_common(1)[0]
        
        # If consensus is high or aligns with original, we're good
        # Otherwise, if total disagreement, default to HOLD/Original with low confidence
        
        final_reasoning = "CONUNCIL DEBATE:\n"
        for name, data in results.items():
            final_reasoning += f"- {name}: {data.get('sentiment')} | {data.get('argument')}\n"
            
        return {
            "ticker": original.get("ticker"),
            "sentiment": common_sentiment,
            "confidence": (original.get("confidence", 0) + (count / 3)) / 2, # Blended confidence
            "reasoning": f"{original.get('reasoning')}\n\n{final_reasoning}",
            "consensus_score": count,
            "is_council_verified": True
        }
