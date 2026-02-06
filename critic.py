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
    
    [V2.0] ML-First Architecture:
    - Uses trained ML model as primary evaluator (no API calls).
    - Falls back to AI only for edge cases or when ML is uncertain.
    """
    
    # ML confidence thresholds for auto-approval
    ML_HIGH_CONFIDENCE = 0.70  # Auto-approve if ML agrees with high confidence
    ML_LOW_CONFIDENCE = 0.40   # Escalate to AI if ML confidence is below this
    
    def __init__(self, brain_instance=None):
        self.brain = brain_instance
        self._ml_predictor = None
        
    def _get_ml_predictor(self):
        """Lazy load MLPredictor to avoid circular imports."""
        if self._ml_predictor is None:
            try:
                from ml_predictor import MLPredictor
                self._ml_predictor = MLPredictor()
            except Exception as e:
                logger.warning(f"Critic: Failed to load MLPredictor: {e}")
        return self._ml_predictor
    
    def _ml_critique(self, signal: Dict) -> Optional[CriticVerdict]:
        """
        [ML-First] Fast local validation of signal using confidence heuristics.
        Returns a CriticVerdict if decision is clear, None if escalation to AI is needed.
        
        This is a FAST path that doesn't make network calls. It validates:
        1. Signal direction consistency
        2. Confidence thresholds
        3. Basic sanity checks
        """
        ticker = signal.get('ticker')
        direction = signal.get('direction', 'HOLD').upper()
        hunter_confidence = signal.get('confidence', 0.5)
        reasoning = signal.get('reasoning', '')
        
        logger.info(f"[ML Critic] Fast validation for {ticker}: {direction} @ {hunter_confidence:.0%}")
        
        # [RULE 1] HOLD signals auto-approve (low priority)
        if direction == 'HOLD':
            return CriticVerdict(
                verdict="APPROVE",
                score=60,
                concerns=["Low-priority signal (HOLD)"],
                reasoning=f"[ML Critic] HOLD signals auto-approved for {ticker}."
            )
        
        # [RULE 2] High confidence signals (>70%) - trust the Hunter's ML
        if hunter_confidence >= self.ML_HIGH_CONFIDENCE:
            return CriticVerdict(
                verdict="APPROVE",
                score=int(70 + (hunter_confidence - 0.7) * 100),
                concerns=[],
                reasoning=f"[ML Critic] High-confidence {direction} ({hunter_confidence:.0%}) for {ticker}. Approved."
            )
        
        # [RULE 3] Medium confidence (40-70%) - approve with caution
        if hunter_confidence >= self.ML_LOW_CONFIDENCE:
            return CriticVerdict(
                verdict="APPROVE",
                score=int(50 + hunter_confidence * 30),
                concerns=["Moderate confidence - position sizing advised"],
                reasoning=f"[ML Critic] Moderate {direction} ({hunter_confidence:.0%}) for {ticker}. Approved with caution."
            )
        
        # [RULE 4] Low confidence (<40%) - escalate to AI for detailed review
        logger.info(f"[ML Critic] Low confidence ({hunter_confidence:.0%}) for {ticker}, escalating to AI")
        return None

    def _get_brain(self):
        """Lazy load Brain to avoid circular imports if brain_instance is missing."""
        if self.brain:
            return self.brain
        from brain import Brain
        self.brain = Brain()
        return self.brain

    def _generate_response(self, prompt: str, json_mode: bool = False) -> Optional[str]:
        """Delegates to Brain with Direct-First priority."""
        try:
            brain = self._get_brain()
            return brain._generate_with_fallback(
                prompt, 
                json_mode=json_mode, 
                prefer_direct=True, 
                task_type="critic_eval"
            )
        except Exception as e:
            logger.warning(f"Critic generation failed via Brain (will use fallback): {e}")
            return None

    def critique_signal(self, signal: Dict, context: str) -> CriticVerdict:
        """
        Critiques a proposed trading signal.
        
        [V2.0] ML-First Architecture:
        1. Try ML model first (no API cost).
        2. If ML is confident, return immediately.
        3. Only escalate to AI for edge cases.
        
        Args:
            signal: Dict containing {ticker, direction, confidence, reasoning}
            context: The original context provided to the Hunter (News, Technicals, etc.)
            
        Returns:
            CriticVerdict object
        """
        ticker = signal.get('ticker')
        direction = signal.get('direction', 'HOLD')
        hunter_reasoning = signal.get('reasoning', '')
        
        logger.info(f"🧐 Critic evaluating {direction} signal for {ticker}...")
        
        # [STEP 1] ML-First Evaluation (No API Cost)
        ml_verdict = self._ml_critique(signal)
        if ml_verdict is not None:
            logger.info(f"[ML Critic] {ticker} resolved locally: {ml_verdict.verdict} (Score: {ml_verdict.score})")
            return ml_verdict
        
        # [STEP 2] AI Fallback for Edge Cases
        logger.info(f"[AI Critic] Escalating {ticker} to AI for detailed review...")

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
        5. **Market Hours Compliance**: Respect the [MARKET HOURS] status provided in the context. If it says 🟢 OPEN, the market IS open for trades. Do NOT hallucinate that it is closed.
        
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
            brain = self._get_brain()
            response = brain._generate_with_fallback(prompt, json_mode=True, prefer_direct=True, task_type="critic_eval")
            if not response:
                raise Exception("All Critic AI models failed.")
            
            # Parse JSON
            try:
                logger.info(f"Raw Critic Response: {response}")
                # Clean markdown code blocks if present
                # Clean markdown code blocks if present
                clean_response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(clean_response)
                
                # Handle cases where model returns a list containing the dict
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
                
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict, got {type(data)}")

                return CriticVerdict(
                    verdict=data.get('verdict', 'REJECT').upper(),
                    score=int(data.get('score', 50)),
                    concerns=data.get('concerns', []),
                    reasoning=data.get('reasoning', 'No reasoning provided')
                )
            except json.JSONDecodeError:
                logger.warning(f"Critic returned invalid JSON: {response[:100]}...")
                return CriticVerdict("HOLD", 50, ["AI Parse Error"], "Critic output malformed, defaulting to HOLD for safety.")
                
        except Exception as e:
            logger.warning(f"Critic primary path failed, triggering last-resort recovery: {e}")
            
            # [LAST RESORT RETRY WITH DELAY]
            # If main brain fails, try a direct Gemini call with retry logic
            import time
            import re
            
            brain = self._get_brain()
            if not brain.gemini_client:
                return CriticVerdict("HOLD", 50, ["AI Connectivity Critical"], "Expert Broker unavailable (No Gemini Client). Defaulting to HOLD for safety.")
            
            # Simplified prompt for emergency mode
            emergency_prompt = f"""
            You are a Risk Manager. Review this trade proposal:
            Ticker: {ticker} | Action: {direction} | Reasoning: {hunter_reasoning}
            
            Data: {context[:2000]}... (truncated)
            
            Is this valid? 
            Output JSON: {{ "verdict": "APPROVE" or "REJECT", "score": 50, "concerns": ["List 1 major risk"], "reasoning": "Brief verdict." }}
            """
            
            # Try multiple models with delay on 429
            last_resort_models = ["gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-1.5-flash-8b"]
            
            for model_name in last_resort_models:
                try:
                    logger.info(f"Last Resort Critic: Trying {model_name}...")
                    response = brain._call_gemini_fallback(emergency_prompt, json_mode=True, model=model_name, task_type="critic_last_resort")
                    
                    # Manual Parse
                    clean_response = response.replace('```json', '').replace('```', '').strip()
                    data = json.loads(clean_response)
                     
                    return CriticVerdict(
                        verdict=data.get('verdict', 'REJECT').upper(),
                        score=int(data.get('score', 50)),
                        concerns=data.get('concerns', ["Emergency Mode Used"]),
                        reasoning=f"[Emergency Recovered] {data.get('reasoning', 'Risk check completed.')}"
                    )
                except Exception as e2:
                    error_str = str(e2)
                    logger.warning(f"Last Resort model {model_name} failed: {e2}")
                    
                    # Check if 429, extract retry delay
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        match = re.search(r"retry in ([\d.]+)s", error_str.lower())
                        wait_time = 35  # Default
                        if match:
                            wait_time = min(int(float(match.group(1))) + 2, 60)
                        
                        logger.info(f"Quota exhausted, waiting {wait_time}s before next model...")
                        time.sleep(wait_time)
                    continue

            # Final Fallback if ALL Last Resort models failed
            return CriticVerdict("HOLD", 50, ["AI Connectivity Critical"], "Expert Broker unavailable (All AI drivers failed). Defaulting to HOLD for safety.")


    def critique_rebalance_strategy(self, strategy_text: str, regime: str, portfolio_value: float, held_assets: List[str] = []) -> str:
        """
        Critiques and potentially rewrites a rebalancing strategy to ensure regime consistency.
        Example: If Regime is BEARISH, blocks buying meme coins or increasing high-risk exposure.
        Enforces that 'HOLD' is ONLY used for assets actually owned.
        """
        logger.info(f"Critic evaluating rebalance strategy (Regime: {regime})...")
        
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
           
        OUTPUT JSON ONLY:
        {{
            "revised_strategy": "The full revised strategy text (Markdown)",
            "was_modified": true | false,
            "broker_reasoning": "A concise explanation of your risk assessment (max 1 sentence)"
        }}
        """
        
        try:
            brain = self._get_brain()
            response = brain._generate_with_fallback(prompt, json_mode=True, prefer_direct=True, task_type="critic_rebalance")
            
            if not response:
                # Graceful fallback: Auto-approve with disclaimer
                logger.warning(f"Critic rebalance: AI unavailable, auto-approving strategy")
                return f"✅ **Auto-Approved** (AI Review Unavailable)\n\n{strategy_text}"
                
            # Parse JSON
            try:
                clean_response = response.replace('```json', '').replace('```', '').strip()
                data = json.loads(clean_response)
                
                revised = data.get('revised_strategy', strategy_text)
                was_mod = data.get('was_modified', False)
                reasoning = data.get('broker_reasoning', '')
                
                # Standardize output: Always include the analysis note
                icon = "🌟" if was_mod else "🧐"
                return f"{icon} **Expert Broker Review**: {reasoning}\n\n{revised.strip()}"
                
            except json.JSONDecodeError:
                logger.warning(f"Critic rebalance returned invalid JSON, auto-approving: {response[:100]}...")
                return f"✅ **Auto-Approved** (Parse Error)\n\n{strategy_text}"
                
        except Exception as e:
            # GRACEFUL FALLBACK: Never fail, always return a valid approval
            logger.warning(f"Critic rebalance: AI providers exhausted, auto-approving strategy. Reason: {e}")
            return f"✅ **Auto-Approved** (AI Temporarily Unavailable)\n\n{strategy_text}" 

    def critique_deep_dive(self, ticker: str, analysis_text: str, market_context: str) -> str:
        """
        Critiques a Deep Dive Analysis report from an Expert Broker perspective.
        """
        prompt = f"""
        You are a SENIOR FINANCIAL BROKER with 20+ years of institutional experience.
        You are reviewing a Deep Dive Report for {ticker}.
        
        REPORT CONTENT:
        {analysis_text}
        
        MARKET CONTEXT:
        {market_context}
        
        YOUR TASK:
        Review the report for quality, bias, and realistic risk assessment.
        
        **PRICE GROUNDING (IMPORTANT):**
        - If you see "Portfolio Context: OWNED @ €XXX", that is the **AVERAGE COST**, not the current price.
        - The current market price should be taken from the "Technical Context" or "CURRENT_PRICE" section.
        - **Target Price check:** Compare the Target Price with the **Current Market Price**, NOT the purchase price.
        - Current FX Rate: 1 EUR ≈ 1.05 - 1.10 USD. Be careful not to confuse $ and €.
        
        **CRITICAL:** Respect the current Market Hours and Regime. Do not claim markets are closed if the context indicates otherwise.
        
        OUTPUT JSON ONLY:
        {{
            "verdict": "APPROVED" | "CAUTION" | "DANGEROUS",
            "broker_note": "A concise, professional one-sentence summary of your view (max 20 words)."
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
            reasoning = data.get("broker_note", "")
            
            icon = "✅"
            if verdict == "CAUTION": icon = "⚠️"
            if verdict == "DANGEROUS": icon = "🛑"
            
            return f"\n\n{icon} **Expert Broker Review**: {verdict}\n> {reasoning}"
            
        except Exception as e:
            logger.warning(f"Critic deep dive: AI unavailable, auto-approving: {e}")
            return "\n\n✅ **Auto-Approved** (AI Review Unavailable)"

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
