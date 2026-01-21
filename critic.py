import logging
import json
import os
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("Critic")

@dataclass
class CriticVerdict:
    verdict: str  # "APPROVE" or "REJECT"
    score: int    # 0-100 (Flow quality score, higher is better quality/less risk)
    concerns: List[str]
    reasoning: str

class Critic:
    """
    The Critic Agent (Adversarial Validator).
    Acts as a conservative Risk Manager, actively looking for reasons NOT to trade.
    """
    
    def __init__(self):
        self.model = "gemini-2.5-flash" # Verified available model
        # Fallback to Llama 3 if Gemini fails
        self.fallback_model = "meta-llama/llama-3.3-70b-instruct:free"

    def critique_signal(self, signal: Dict, context: str) -> CriticVerdict:
        """
        Critiques a proposed trading signal.
        
        Args:
            signal: Dict containing {ticker, direction, confidence, reasoning}
            context: The original context provided to the Hunter (News, Technicals, etc.)
            
        Returns:
            CriticVerdict object
        """
        ticker = signal.get('ticker')
        direction = signal.get('direction', 'HOLD')
        hunter_reasoning = signal.get('reasoning', '')
        
        # Skip critique for HOLD signals (already safe)
        if direction == "HOLD":
            return CriticVerdict("APPROVE", 100, [], "Signal is HOLD, no risk to capital.")

        logger.info(f"🧐 Critic evaluating {direction} signal for {ticker}...")

        prompt = f"""
        You are the CHIEF RISK OFFICER of a Hedge Fund. 
        Your job is to protecting capital by REJECTING bad trade ideas.
        
        A Junior Analyst (The Hunter) has proposed the following trade:
        TYPE: {direction}
        ASSET: {ticker}
        CONFIDENCE: {signal.get('confidence', 0):.2f}
        RATIONALE: {hunter_reasoning}
        
        MARKET CONTEXT:
        {context}
        
        YOUR TASK:
        Analyze this proposal with EXTREME SKEPTICISM. Look for:
        1. Macro Headwinds (Is the general market against this?)
        2. Technical conflicting signals (e.g. Buying into resistance?)
        3. Weak Fundamentals/News (Is the "good news" actually priced in?)
        4. Over-optimism in the Hunter's reasoning.
        
        OUTPUT JSON ONLY:
        {{
            "verdict": "APPROVE" (if solid) or "REJECT" (if risky),
            "score": 0-100 (0=Terrible idea, 100=Flawless),
            "concerns": ["List", "of", "specific", "risks"],
            "reasoning": "Short explanation of your decision (max 2 sentences)"
        }}
        """
        
        try:
            from brain import Brain
            brain = Brain()
            # We use a lower temperature for the Critic to be consistent and analytical
            # We use a lower temperature for the Critic to be consistent and analytical
            # Note: _generate_with_fallback handles model selection. We pass prefer_free=True.
            response = brain._generate_with_fallback(prompt, model=self.model, prefer_free=True, json_mode=True)
            
            # Parse JSON
            try:
                logger.info(f"Raw Critic Response: {response}")
                # Clean markdown code blocks if present
                # Clean markdown code blocks if present
                clean_response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(clean_response)
                
                return CriticVerdict(
                    verdict=data.get('verdict', 'REJECT').upper(),
                    score=int(data.get('score', 50)),
                    concerns=data.get('concerns', []),
                    reasoning=data.get('reasoning', 'No reasoning provided')
                )
            except json.JSONDecodeError:
                logger.error(f"Critic returned invalid JSON: {response}")
                return CriticVerdict("REJECT", 0, ["AI Parse Error"], "Failed to parse Critic response.")
                
        except Exception as e:
            logger.error(f"Critic execution failed: {e}")
            return CriticVerdict("APPROVE", 50, ["Critic Offline"], f"Critic check failed ({e}). Defaulting to cautious approval.")

    def critique_rebalance_strategy(self, strategy_text: str, regime: str, portfolio_value: float) -> str:
        """
        Critiques and potentially rewrites a rebalancing strategy to ensure regime consistency.
        Example: If Regime is BEARISH, blocks buying meme coins or increasing high-risk exposure.
        """
        logger.info(f"🧐 Critic evaluating rebalance strategy (Regime: {regime})...")
        
        prompt = f"""
        You are the SENIOR RISK MANAGER (The Critic).
        Current Market Regime: **{regime}**
        Portfolio Value: €{portfolio_value:.0f}
        
        PROPOSED STRATEGY (by Junior Algo):
        {strategy_text}
        
        YOUR TASK:
        Review the proposed actions for consistency with the regime.
        
        RULES:
        1. If Regime is BEARISH/DEFENSIVE:
           - VETO any "BUY" suggestion for High Risk assets (Meme Coins like DOGE, SHIB, PEPE, or Small Caps).
           - VETO increasing exposure to volatile Crypto (unless it's just rebalancing major coins BTC/ETH).
           - APPROVE "TRIM" or "HOLD" or "Accumulate Gold/Cash/Stablecoins".
           - IF you VETO an action, REPLACE it with "🟡 HOLD [Ticker] - Risk too high for Bearish Regime" or "🟢 BUY [Defensive Asset]".
           
        2. If Regime is BULLISH:
           - Allow aggressive moves if they make sense.
           
        3. GENERAL:
           - Ensure fees don't eat profits (Policy: Net Gain must justify €1 fee + 26% tax).
           
        OUTPUT:
        Return the FINAL STRATEGY text (Markdown).
        - If the original is safe, return it exactly as is.
        - If changes are needed, return the EDITED version.
        - Maintain the exact formatting (🟢, 🔴, 🟡 bullet points).
        - Do NOT add conversational filler like "Here is the revised strategy". Just the list.
        """
        
        try:
            from brain import Brain
            brain = Brain()
            # Use a smart model for this logic check
            response = brain._generate_with_fallback(prompt, model=self.model, prefer_free=True, json_mode=False)
            
            if not response:
                return strategy_text # Fallback to original if AI fails
                
            # Log if changes were made
            if response.strip() != strategy_text.strip():
                logger.warning("⛔ CRITIC MODIFIED THE STRATEGY! (Risk Override Applied)")
                logger.info(f"Original: {strategy_text[:50]}...")
                logger.info(f"Revised: {response[:50]}...")
                return f"\n\n👮‍♂️ **Risk Manager Update (Critic)**:\n{response.strip()}" # Return modified strategy with header
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Critic rebalance check failed: {e}")
            return strategy_text # Fail open (allow original) if Critic breaks

    def critique_deep_dive(self, ticker: str, analysis_text: str, market_context: str) -> str:
        """
        Critiques a Deep Dive Analysis report.
        Ensures the tone matches the market context (e.g., checks for over-optimism in a Bear Market).
        """
        prompt = f"""
        You are the CHIEF RISK OFFICER of a Hedge Fund.
        A Senior Analyst has submitted the following Deep Dive Report for {ticker}:
        
        ---
        {analysis_text}
        ---
        
        CURRENT MARKET REGIME: {market_context}
        
        YOUR TASK:
        Review the report for "Irrational Exuberance" or dangerous omissions.
        
        RULES:
        1. If the Market is BEARISH and the report is "Strong Buy" without mentioning risks -> FLAGGED.
        2. If the Market is BULLISH and the report is "Sell" without strong catalyst -> FLAGGED.
        3. If the report is balanced and accurate -> APPROVED.
        
        OUTPUT JSON ONLY:
        {{
            "verdict": "APPROVED" | "CAUTION" | "DANGEROUS",
            "risk_score": 0-10 (0=Safe, 10=Extremely Risky),
            "comment": "Short comment (max 15 words) visible to the user."
        }}
        """
        
        try:
            from brain import Brain
            brain = Brain()
            response = brain._generate_with_fallback(prompt, model=self.model, prefer_free=True, json_mode=True)
            
            import json
            import re
            
            # Clean JSON
            clean_json = re.sub(r"```json|```", "", response).strip()
            data = json.loads(clean_json)
            
            verdict = data.get("verdict", "APPROVED")
            comment = data.get("comment", "")
            
            icon = "✅"
            if verdict == "CAUTION": icon = "⚠️"
            if verdict == "DANGEROUS": icon = "🛑"
            
            return f"\n\n{icon} **Risk Officer (Critic)**: {verdict}\n> {comment}"
            
        except Exception as e:
            logger.error(f"Critic deep dive failed: {e}")
            return "\n\n⚠️ **Critic Unavailable** (System Error)"

if __name__ == "__main__":
    # Test
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    c = Critic()
    
    mock_signal = {
        "ticker": "BTC-USD",
        "direction": "UP",
        "confidence": 0.85,
        "reasoning": "RSI is low and ETF inflows are positive."
    }
    
    mock_context = "Bitcoin price is $95,000. RSI 35. Federal Reserve just hinted at rate HIKES. S&P500 is down 2% today."
    
    print("\n--- Testing Critic ---")
    result = c.critique_signal(mock_signal, mock_context)
    print(f"Verdict: {result.verdict}")
    print(f"Score: {result.score}")
    print(f"Concerns: {result.concerns}")
    print(f"Reasoning: {result.reasoning}")
