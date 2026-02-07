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

    @staticmethod
    def _base_ticker(value: str) -> str:
        t = str(value or "").upper().strip()
        if not t:
            return ""
        return t.replace("-USD", "").replace("-EUR", "").replace("-GBP", "")

    def _resolve_portfolio_holding(self, ticker: str, portfolio_context: Optional[List[Dict]]) -> Tuple[bool, Optional[Dict]]:
        if not isinstance(portfolio_context, list):
            return False, None

        target = self._base_ticker(ticker)
        if not target:
            return False, None

        for row in portfolio_context:
            if not isinstance(row, dict):
                continue
            row_ticker = row.get("ticker")
            if not row_ticker:
                continue
            if self._base_ticker(row_ticker) == target:
                return True, row
        return False, None

    def _portfolio_snapshot(self, portfolio_context: Optional[List[Dict]], max_items: int = 8) -> str:
        if not isinstance(portfolio_context, list) or not portfolio_context:
            return "Portafoglio non disponibile."

        items = []
        for row in portfolio_context:
            if not isinstance(row, dict):
                continue
            ticker = row.get("ticker")
            qty = row.get("quantity")
            avg = row.get("avg_price")
            if not ticker:
                continue
            piece = f"{ticker}"
            if qty is not None and avg is not None:
                piece += f" (qty {qty}, avg €{avg})"
            items.append(piece)
            if len(items) >= max_items:
                break

        if not items:
            return "Portafoglio non disponibile."
        suffix = "…" if isinstance(portfolio_context, list) and len(portfolio_context) > len(items) else ""
        return ", ".join(items) + suffix

    async def get_consensus(self, ticker: str, initial_signal: Dict, portfolio_context: Optional[List[Dict]] = None) -> Dict:
        """
        Runs the council debate for a specific signal.
        ticker: Asset ticker
        initial_signal: Data from the initial Brain analysis
        """
        if not self.brain:
            logger.warning("Council: Brain instance missing, simplified consensus used.")
            return initial_signal

        logger.info(f"THE COUNCIL: Debating {ticker}...")

        try:
            is_owned, holding = self._resolve_portfolio_holding(ticker, portfolio_context)
            enriched_signal = dict(initial_signal or {})
            enriched_signal["is_owned_asset"] = bool(is_owned)
            enriched_signal["holding_summary"] = holding or {}
            enriched_signal["portfolio_snapshot"] = self._portfolio_snapshot(portfolio_context)

            # 1. Single Multi-Persona Prompt
            prompt = self._build_unified_council_prompt(ticker, enriched_signal)
            
            # Call brain once for all personas (forced to background to save Gemini quota)
            response = self.brain._generate_with_fallback(prompt, json_mode=True, task_type="council_debate", prefer_direct=True)
            
            council_data = json.loads(response)
            
            # 2. Extract results and votes
            results = {
                "THE_BULL": council_data.get("THE_BULL", {}),
                "THE_BEAR": council_data.get("THE_BEAR", {}),
                "THE_QUANT": council_data.get("THE_QUANT", {})
            }
            votes = []
            for persona in ("THE_BULL", "THE_BEAR", "THE_QUANT"):
                raw_vote = str(results[persona].get("sentiment", "HOLD")).upper().strip()
                if raw_vote not in {"BUY", "SELL", "HOLD", "ACCUMULATE"}:
                    raw_vote = "HOLD"
                votes.append(raw_vote)
            
            # 3. Consensus Logic
            verdict = self._calculate_verdict(votes, results, enriched_signal)
            
            logger.info(f"🏛️ COUNCIL VERDICT [{ticker}]: {verdict['sentiment']} ({verdict['consensus_score']}/3 Agreement)")
            return verdict

        except Exception as e:
            logger.error(f"Council unified debate failed for {ticker}: {e}")
            return initial_signal

    def _build_unified_council_prompt(self, ticker: str, signal: Dict) -> str:
        is_owned = bool(signal.get("is_owned_asset"))
        holding = signal.get("holding_summary") or {}
        holding_ticker = holding.get("ticker", ticker)
        holding_qty = holding.get("quantity", "N/A")
        holding_avg = holding.get("avg_price", "N/A")
        portfolio_snapshot = signal.get("portfolio_snapshot", "Portafoglio non disponibile.")

        return f"""
        ROLES: You must simultaneously act as three specialized investment personas:
        1. THE_BULL: Optimistic Growth Analyst (Focus: Growth catalysts, bullish momentum).
        2. THE_BEAR: Risk Management Skeptic (Focus: Macro risks, valuation, breakdowns).
        3. THE_QUANT: Technical Data Scientist (Focus: RSI/MACD, liquidity, probability).

        ASSET: {ticker}
        INITIAL ANALYSIS: {signal.get('sentiment')} with {signal.get('confidence', 0)*100:.0f}% confidence.
        REASONING: {signal.get('reasoning')}
        OWNERSHIP: {"OWNED" if is_owned else "NOT_OWNED"}
        HOLDING DETAILS: ticker={holding_ticker}, quantity={holding_qty}, avg_price={holding_avg}
        PORTFOLIO SNAPSHOT: {portfolio_snapshot}
        
        TASK:
        Provide a consolidated debate. Each persona must evaluate the signal from its perspective.

        CRITICAL RULES:
        - Write arguments in ITALIANO.
        - If OWNERSHIP = OWNED and sentiment is bullish, prefer ACCUMULATE over BUY.
        - Use BUY for owned assets only if the setup is exceptionally strong.
        - If OWNERSHIP = NOT_OWNED, do not use ACCUMULATE.
        - Keep coherence with portfolio risk (do not ignore existing concentration).
        
        Return exactly this JSON structure:
        {{
            "THE_BULL": {{ "sentiment": "BUY|ACCUMULATE|SELL|HOLD", "confidence": 0.0-1.0, "argument": "short brief" }},
            "THE_BEAR": {{ "sentiment": "BUY|ACCUMULATE|SELL|HOLD", "confidence": 0.0-1.0, "argument": "short brief" }},
            "THE_QUANT": {{ "sentiment": "BUY|ACCUMULATE|SELL|HOLD", "confidence": 0.0-1.0, "argument": "short brief" }}
        }}
        """

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
                response = self.brain._generate_with_fallback(prompt, json_mode=True, task_type="council_critique", prefer_direct=True)
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
                response = self.brain._generate_with_fallback(prompt, json_mode=True, task_type="council_rebalance", prefer_direct=True)
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
        is_owned = bool(original.get("is_owned_asset"))

        normalized_votes = []
        for vote in votes:
            v = str(vote or "HOLD").upper().strip()
            if v not in {"BUY", "SELL", "HOLD", "ACCUMULATE"}:
                v = "HOLD"
            if is_owned and v == "BUY":
                v = "ACCUMULATE"
            normalized_votes.append(v)

        counts = Counter(normalized_votes)
        
        # Find the most common sentiment
        common_sentiment, count = counts.most_common(1)[0]
        
        # Determine descriptive label
        label = "UNANIMOUS" if count == 3 else "MAJORITY" if count == 2 else "DISPUTED"
        
        # Build nuanced summary
        # If Unanimous, emphasize the collective focus
        # If Majority, highlight what the dissenter said
        
        council_summary = f"{label} VERDICT: {common_sentiment} ({count}/3)"
        if is_owned and common_sentiment in {"ACCUMULATE", "BUY"}:
            council_summary += " | OWNED_ASSET"
        
        dissent_note = ""
        if count < 3:
            dissenter = None
            for name, data in results.items():
                diss_sent = str(data.get("sentiment", "HOLD")).upper().strip()
                if is_owned and diss_sent == "BUY":
                    diss_sent = "ACCUMULATE"
                if diss_sent != common_sentiment:
                    dissenter = name
                    break
            if dissenter:
                dissent_sentiment = str(results[dissenter].get("sentiment", "HOLD")).upper().strip()
                if is_owned and dissent_sentiment == "BUY":
                    dissent_sentiment = "ACCUMULATE"
                dissent_argument = results[dissenter].get("argument", "")
                dissent_note = f"\n⚠️ **Dissent ({dissenter})**: Argued for {dissent_sentiment} because '{dissent_argument}'"

        final_reasoning = f"🏛️ **COUNCIL DEBATE ({label})**\n"
        for name, data in results.items():
            persona_sent = str(data.get("sentiment", "HOLD")).upper().strip()
            if is_owned and persona_sent == "BUY":
                persona_sent = "ACCUMULATE"
            final_reasoning += f"- **{name}**: {persona_sent} | {data.get('argument')}\n"
        if is_owned:
            final_reasoning += "- **Ownership Context**: Asset già in portafoglio, privilegiata logica ACCUMULATE.\n"
            
        # [FIX] Preserve all original fields (Risk management, Expert Critic, etc.)
        verdict = original.copy()
        verdict.update({
            "ticker": original.get("ticker"),
            "sentiment": common_sentiment,
            "confidence": (original.get("confidence", 0) + (count / 3)) / 2, # Blended confidence
            "council_full_debate": final_reasoning + dissent_note,
            "council_summary": council_summary,
            "consensus_score": count,
            "is_council_verified": True
        })
        return verdict
