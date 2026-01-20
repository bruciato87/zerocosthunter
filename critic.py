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
