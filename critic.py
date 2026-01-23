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
        # Model IDs
        self.gemini_model = "gemini-2.0-flash" 
        self.openrouter_model = "google/gemini-2.0-flash-exp:free"
        
        # Initialize direct Gemini client if key available (Primary)
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.gemini_api_key:
            from google import genai
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.gemini_client = None

    def _generate_response(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        """Tries Gemini first, then OpenRouter fallback."""
        # 1. Try Direct Gemini (Primary)
        if self.gemini_client:
            try:
                from google.genai import types
                logger.info(f"🧐 Critic using Primary: Gemini ({self.gemini_model})")
                config = {}
                if json_mode:
                    config = types.GenerateContentConfig(response_mime_type="application/json")
                
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=config
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"Critic Primary (Gemini) failed: {e}. Falling back to OpenRouter...")

        # 2. Try OpenRouter (Fallback)
        try:
            from brain import Brain
            brain = Brain()
            logger.info(f"🧐 Critic using Fallback: OpenRouter ({self.openrouter_model})")
            return brain._generate_with_fallback(prompt, model=self.openrouter_model, prefer_free=True, json_mode=json_mode)
        except Exception as e:
            logger.error(f"Critic Fallback (OpenRouter) failed: {e}")
            return None

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
        You are a SENIOR FINANCIAL BROKER & MARKET STRATEGIST with 20+ years of experience in TradFi (Wall Street) and Crypto (DeFi/On-Chain).
        Your mission is to provide an ELITE level second opinion on proposed trades to ensure capital is deployed in high-probability setups.
        
        A Junior Analyst has proposed the following:
        - TICKER: {ticker}
        - ACTION: {direction}
        - INTENT: {hunter_reasoning}
        - CONFIDENCE: {signal.get('confidence', 0):.2f}
        
        MARKET CONTEXT:
        {context}
        
        ANALYSIS CRITERIA:
        1. **Liquidity & Spread**: Is there enough volume or is this a "low-float" trap?
        2. **Narrative Strength**: Is this a "buy the rumor" noise or a structural trend change? 
        3. **Risk/Reward skew**: If the news is already trending, what is the 'exhaustion risk'?
        4. **Regime Alignment**: Does this trade match the current Market Regime (Bull/Bear/Neutral)?
        
        YOUR ROLE:
        - Don't just REJECT. Be DISCERNING.
        - If the setup is good but risky, provide a high score (>70) but note the specific risks.
        - If the setup is weak, biased, or lacks clear volume confirmation, be firm and REJECT.
        - A score between 40-60 indicates a "Gray Area" trade - caution required.
        
        OUTPUT JSON ONLY:
        {{
            "verdict": "APPROVE" | "REJECT",
            "score": 0-100 (40-60 is neutral/gray),
            "concerns": ["ListSPECIFICMarketRisks"],
            "reasoning": "Nuanced strategist explanation (max 2 sentences)"
        }}
        """
        try:
            # Use our prioritized generator
            response = self._generate_response(prompt, json_mode=True)
            if not response:
                raise Exception("All Critic AI models failed.")
            
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

    def critique_rebalance_strategy(self, strategy_text: str, regime: str, portfolio_value: float, held_assets: List[str] = []) -> str:
        """
        Critiques and potentially rewrites a rebalancing strategy to ensure regime consistency.
        Example: If Regime is BEARISH, blocks buying meme coins or increasing high-risk exposure.
        Enforces that 'HOLD' is ONLY used for assets actually owned.
        """
        logger.info(f"🧐 Critic evaluating rebalance strategy (Regime: {regime})...")
        
        held_assets_str = ", ".join(held_assets) if held_assets else "NONE"
        
        prompt = f"""
        You are the SENIOR RISK MANAGER (The Critic).
        Current Market Regime: **{regime}**
        Portfolio Value: €{portfolio_value:.0f}
        
        CURRENT PORTFOLIO ASSETS: {held_assets_str}
        
        PROPOSED STRATEGY (by Junior Algo):
        {strategy_text}
        
        YOUR TASK:
        Review the proposed actions for consistency with the regime and portfolio reality.
        
        RULES:
        1. If Regime is BEARISH/DEFENSIVE:
           - VETO any "BUY" suggestion for High Risk assets (Meme Coins like DOGE, SHIB, PEPE, or Small Caps).
           - VETO increasing exposure to volatile Crypto (unless it's just rebalancing major coins BTC/ETH).
           - APPROVE "TRIM" or "HOLD" or "Accumulate Gold/Cash/Stablecoins".
           
        2. OWNERSHIP RULES (CRITICAL):
           - You can ONLY approve "HOLD", "ACCUMULATE" or "TRIM" for assets in CURRENT PORTFOLIO ASSETS.
           - If the Junior suggests "HOLD" for an asset NOT in portfolio -> DELETE THE LINE.
           - If you VETO a "BUY" for a NEW asset (not in portfolio) -> CHANGE it to "🚫 AVOID [Ticker] - Risk too high".
           - DO NOT change a VETOED "BUY" for a new asset into "HOLD". It must be "AVOID".
           
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
            # Use our prioritized generator
            response = self._generate_response(prompt, json_mode=False)
            
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
            # Use our prioritized generator
            response = self._generate_response(prompt, json_mode=True)
            
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
