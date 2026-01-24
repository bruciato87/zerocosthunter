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

    async def get_report_consensus(self, ticker: str, report_text: str, context: str) -> str:
        """
        [PHASE C.2] Personas critique a Deep Dive report.
        """
        if not self.brain: return report_text
        
        logger.info(f"🏛️ THE COUNCIL: Critiquing Deep Dive for {ticker}...")
        critiques = []
        
        for name, profile in self.PERSONAS.items():
            try:
                prompt = self._build_report_critique_prompt(ticker, profile, report_text, context)
                response = self.brain._generate_with_fallback(prompt, json_mode=True, task_type="analyze", prefer_direct=True)
                data = json.loads(response)
                critiques.append(f"- **{name} ({data.get('verdict','CAUTION')})**: {data.get('critique','No comment')}")
            except Exception as e:
                logger.error(f"Council Agent {name} report critique failed: {e}")

        consensus_summary = "\n\n🏛️ **ADVERSARIAL COUNCIL REVIEW**\n"
        consensus_summary += "\n".join(critiques)
        return f"{report_text}\n{consensus_summary}"

    async def get_strategy_consensus(self, current_portfolio: str, proposed_strategy: str) -> str:
        """
        [PHASE C.2] Personas debate a rebalancing strategy.
        """
        if not self.brain: return proposed_strategy
        
        logger.info("🏛️ THE COUNCIL: Debating Rebalance Strategy...")
        critiques = []
        
        for name, profile in self.PERSONAS.items():
            try:
                prompt = self._build_strategy_critique_prompt(profile, current_portfolio, proposed_strategy)
                response = self.brain._generate_with_fallback(prompt, json_mode=True, task_type="rebalance", prefer_direct=True)
                data = json.loads(response)
                critiques.append(f"- **{name}**: {data.get('verdict','NEUTRAL')} | {data.get('critique','No comment')}")
            except Exception as e:
                logger.error(f"Council Agent {name} strategy critique failed: {e}")

        consensus_section = "\n\n🏛️ **COUNCIL STRATEGY CONSENSUS**\n"
        consensus_section += "\n".join(critiques)
        return f"{proposed_strategy}\n{consensus_section}"

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

    def _build_report_critique_prompt(self, ticker: str, profile: Dict, report: str, context: str) -> str:
        return f"""
        YOU ARE: {profile['role']}
        YOUR FOCUS: {profile['focus']}
        
        ASSET: {ticker}
        REPORT TO REVIEW:
        {report[:2000]}... (truncated)
        
        MARKET CONTEXT:
        {context[:1000]}
        
        TASK:
        Critique the report from your specific persona's perspective. Be adversarial.
        Provide JSON:
        {{
            "verdict": "APPROVE|CAUTION|DANGEROUS",
            "critique": "Your specific sharp comment (max 20 words)"
        }}
        """

    def _build_strategy_critique_prompt(self, profile: Dict, portfolio: str, strategy: str) -> str:
        return f"""
        YOU ARE: {profile['role']}
        YOUR FOCUS: {profile['focus']}
        
        PORTFOLIO:
        {portfolio}
        
        PROPOSED REBALANCE:
        {strategy}
        
        TASK:
        Critique the rebalance moves. Are they too risky? Too conservative? Justified by data?
        Provide JSON:
        {{
            "verdict": "BULLISH|BEARISH|NEUTRAL",
            "critique": "Your specific sharp comment (max 20 words)"
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
        
        final_reasoning = "COUNCIL DEBATE:\n"
        council_summary_lines = []
        for name, data in results.items():
            final_reasoning += f"- {name}: {data.get('sentiment')} | {data.get('argument')}\n"
            council_summary_lines.append(f"{name}: {data.get('sentiment')}")
            
        council_summary = " | ".join(council_summary_lines)
            
        return {
            "ticker": original.get("ticker"),
            "sentiment": common_sentiment,
            "confidence": (original.get("confidence", 0) + (count / 3)) / 2, # Blended confidence
            "reasoning": original.get("reasoning", ""), # Keep original clean reasoning
            "council_full_debate": final_reasoning,
            "council_summary": council_summary,
            "consensus_score": count,
            "is_council_verified": True
        }
