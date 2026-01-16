"""
strategy_manager.py
Level 8: Strategy Governance - Rules of Engagement

This module enforces user-defined trading rules to prevent over-trading
and maximize net profits by filtering AI suggestions through strategic constraints.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy types for assets."""
    ACCUMULATE = "ACCUMULATE"  # Build position over time (PAC)
    SWING = "SWING"            # Trade actively with take-profit
    LONG_TERM = "LONG_TERM"    # Never sell (hold forever)


@dataclass
class StrategyRule:
    """Represents a strategy rule for a single asset."""
    ticker: str
    strategy_type: StrategyType
    target_allocation_pct: float
    max_allocation_cap: float
    take_profit_threshold: Optional[float]  # e.g. 20.0 for +20%
    stop_loss_threshold: Optional[float]    # e.g. -15.0 for -15%
    min_net_profit_eur: float               # Minimum net profit to trigger sell
    notes: Optional[str] = None


class StrategyManager:
    """
    Central governance engine for trading decisions.
    Filters AI suggestions through user-defined rules.
    """
    
    # Default costs for Trade Republic Italy
    FEE_PER_TRADE = 1.0  # €1 per trade
    TAX_RATE = 0.26      # 26% capital gains tax
    
    def __init__(self):
        """Initialize the Strategy Manager."""
        self.db = None
        self._rules_cache: Dict[str, StrategyRule] = {}
        self._load_rules()
    
    def _load_rules(self) -> None:
        """Load strategy rules from the database."""
        try:
            from db_handler import DBHandler
            self.db = DBHandler()
            
            result = self.db.supabase.table("strategy_rules").select("*").execute()
            
            if result.data:
                for row in result.data:
                    rule = StrategyRule(
                        ticker=row['ticker'],
                        strategy_type=StrategyType(row['strategy_type']),
                        target_allocation_pct=float(row.get('target_allocation_pct', 10)),
                        max_allocation_cap=float(row.get('max_allocation_cap', 20)),
                        take_profit_threshold=float(row['take_profit_threshold']) if row.get('take_profit_threshold') else None,
                        stop_loss_threshold=float(row['stop_loss_threshold']) if row.get('stop_loss_threshold') else None,
                        min_net_profit_eur=float(row.get('min_net_profit_eur', 50)),
                        notes=row.get('notes')
                    )
                    self._rules_cache[row['ticker']] = rule
                
                logger.info(f"StrategyManager: Loaded {len(self._rules_cache)} rules from DB")
            else:
                logger.warning("StrategyManager: No strategy rules found in DB")
                
        except Exception as e:
            logger.error(f"StrategyManager: Failed to load rules: {e}")
    
    def get_rule(self, ticker: str) -> Optional[StrategyRule]:
        """Get the strategy rule for a specific ticker."""
        return self._rules_cache.get(ticker)
    
    def get_all_rules(self) -> Dict[str, StrategyRule]:
        """Get all strategy rules."""
        return self._rules_cache.copy()
    
    def set_rule(self, ticker: str, strategy_type: str, target_pct: float = 10, 
                 max_cap: float = 20, take_profit: float = None, stop_loss: float = None,
                 min_profit: float = 50, notes: str = None) -> bool:
        """
        Create or update a strategy rule for a ticker.
        Returns True on success.
        """
        try:
            data = {
                'ticker': ticker.upper(),
                'strategy_type': strategy_type.upper(),
                'target_allocation_pct': target_pct,
                'max_allocation_cap': max_cap,
                'take_profit_threshold': take_profit,
                'stop_loss_threshold': stop_loss,
                'min_net_profit_eur': min_profit,
                'notes': notes
            }
            
            # Upsert (insert if not exists, update if exists)
            self.db.supabase.table("strategy_rules").upsert(data, on_conflict='ticker').execute()
            
            # Update cache
            self._rules_cache[ticker.upper()] = StrategyRule(
                ticker=ticker.upper(),
                strategy_type=StrategyType(strategy_type.upper()),
                target_allocation_pct=target_pct,
                max_allocation_cap=max_cap,
                take_profit_threshold=take_profit,
                stop_loss_threshold=stop_loss,
                min_net_profit_eur=min_profit,
                notes=notes
            )
            
            logger.info(f"StrategyManager: Set rule for {ticker}: {strategy_type}")
            return True
            
        except Exception as e:
            logger.error(f"StrategyManager: Failed to set rule for {ticker}: {e}")
            return False
    
    def validate_signal(self, ticker: str, ai_signal: str, current_pnl_pct: float,
                        current_allocation_pct: float, potential_profit_eur: float) -> Tuple[str, str]:
        """
        Validate an AI signal against strategy rules.
        
        Args:
            ticker: Asset ticker
            ai_signal: AI suggested action (BUY, SELL, ACCUMULATE, TRIM, HOLD, WAIT)
            current_pnl_pct: Current profit/loss percentage (e.g. 15.5 for +15.5%)
            current_allocation_pct: Current allocation in portfolio (e.g. 12.0 for 12%)
            potential_profit_eur: Potential profit in EUR if sold
        
        Returns:
            Tuple of (final_signal, reason)
            - final_signal: The validated/modified signal
            - reason: Explanation for the decision
        """
        rule = self.get_rule(ticker)
        
        # If no rule exists, pass through the AI signal
        if not rule:
            return (ai_signal, "No strategy rule defined - following AI suggestion")
        
        # =========================================================================
        # RULE 1: LONG_TERM assets cannot be sold (except stop-loss)
        # =========================================================================
        if rule.strategy_type == StrategyType.LONG_TERM:
            if ai_signal in ["SELL", "TRIM", "PANIC SELL"]:
                # Check for stop-loss exception
                if rule.stop_loss_threshold and current_pnl_pct <= rule.stop_loss_threshold:
                    return ("TRIM", f"🛡️ LONG_TERM Override: Stop-loss hit ({current_pnl_pct:.1f}% <= {rule.stop_loss_threshold}%)")
                else:
                    return ("HOLD", f"🛡️ LONG_TERM Strategy: AI suggested {ai_signal}, but this asset is marked for long-term holding")
        
        # =========================================================================
        # RULE 2: Allocation Cap - Don't buy if already at max
        # =========================================================================
        if ai_signal in ["BUY", "ACCUMULATE"]:
            if current_allocation_pct >= rule.max_allocation_cap:
                return ("HOLD", f"🚫 Allocation Cap: Already at {current_allocation_pct:.1f}% (max: {rule.max_allocation_cap}%)")
        
        # =========================================================================
        # RULE 3: Take Profit - Force trim if above threshold
        # =========================================================================
        if rule.take_profit_threshold and current_pnl_pct >= rule.take_profit_threshold:
            if ai_signal not in ["SELL", "TRIM"]:
                # Calculate net profit after tax and fees
                net_profit = self._calculate_net_profit(potential_profit_eur)
                if net_profit >= rule.min_net_profit_eur:
                    return ("TRIM", f"💰 Take Profit: +{current_pnl_pct:.1f}% reached target of +{rule.take_profit_threshold}% (Net: €{net_profit:.0f})")
        
        # =========================================================================
        # RULE 4: Tax Efficiency - Don't sell if net profit is too low
        # =========================================================================
        if ai_signal in ["SELL", "TRIM"] and current_pnl_pct > 0:
            net_profit = self._calculate_net_profit(potential_profit_eur)
            if net_profit < rule.min_net_profit_eur:
                return ("HOLD", f"📊 Tax Efficiency: Net profit €{net_profit:.0f} < min €{rule.min_net_profit_eur:.0f} (not worth the tax)")
        
        # =========================================================================
        # RULE 5: Stop-Loss - Force sell if below threshold
        # =========================================================================
        if rule.stop_loss_threshold and current_pnl_pct <= rule.stop_loss_threshold:
            if ai_signal not in ["SELL", "PANIC SELL"]:
                return ("SELL", f"⚠️ Stop-Loss: {current_pnl_pct:.1f}% hit threshold of {rule.stop_loss_threshold}%")
        
        # All checks passed, use AI signal
        return (ai_signal, "✅ Strategy aligned with AI suggestion")
    
    def _calculate_net_profit(self, gross_profit_eur: float) -> float:
        """Calculate net profit after tax and fees."""
        if gross_profit_eur <= 0:
            return gross_profit_eur  # No tax on losses
        
        tax = gross_profit_eur * self.TAX_RATE
        net = gross_profit_eur - tax - self.FEE_PER_TRADE
        return max(0, net)
    
    def get_strategy_context(self, ticker: str, current_pnl_pct: float, current_allocation_pct: float) -> str:
        """
        Generate a strategy context string to inject into AI prompts.
        This helps the AI understand the user's rules before making suggestions.
        """
        rule = self.get_rule(ticker)
        
        if not rule:
            return f"[Strategy: No specific rule for {ticker}]"
        
        parts = [f"[Strategy for {ticker}]"]
        parts.append(f"- Type: {rule.strategy_type.value}")
        parts.append(f"- Target Allocation: {rule.target_allocation_pct}% (Current: {current_allocation_pct:.1f}%)")
        parts.append(f"- Max Cap: {rule.max_allocation_cap}%")
        
        if rule.take_profit_threshold:
            parts.append(f"- Take Profit: +{rule.take_profit_threshold}% (Current PnL: {current_pnl_pct:+.1f}%)")
        if rule.stop_loss_threshold:
            parts.append(f"- Stop Loss: {rule.stop_loss_threshold}%")
        
        parts.append(f"- Min Net Profit: €{rule.min_net_profit_eur}")
        
        if rule.strategy_type == StrategyType.LONG_TERM:
            parts.append("⚠️ RULE: DO NOT suggest SELL for this asset unless stop-loss is hit!")
        
        return "\n".join(parts)
    
    def format_rules_report(self) -> str:
        """Generate a formatted report of all strategy rules for Telegram."""
        if not self._rules_cache:
            return "📋 Strategy Rules\n\n❌ Nessuna regola definita.\nUsa /strategy set TICKER type=SWING per aggiungerne una."
        
        report = "📋 Strategy Rules\n"
        report += "━" * 24 + "\n\n"
        
        for ticker, rule in sorted(self._rules_cache.items()):
            emoji = "🔵" if rule.strategy_type == StrategyType.LONG_TERM else "🟢" if rule.strategy_type == StrategyType.ACCUMULATE else "🟡"
            report += f"{emoji} {ticker}: {rule.strategy_type.value}\n"
            report += f"   📎 Target: {rule.target_allocation_pct}% | Cap: {rule.max_allocation_cap}%\n"
            tp_sl = ""
            if rule.take_profit_threshold:
                tp_sl += f"TP: +{rule.take_profit_threshold}%"
            if rule.stop_loss_threshold:
                if tp_sl:
                    tp_sl += " | "
                tp_sl += f"SL: {rule.stop_loss_threshold}%"
            if tp_sl:
                report += f"   💰 {tp_sl}\n"
            report += "\n"
        
        report += "━" * 24 + "\n"
        report += "Modifica: /strategy set TICKER ..."
        
        return report
    
    # =========================================================================
    # Position Sizing (Kelly Criterion) - Level 10
    # =========================================================================
    
    def calculate_position_size(self, ticker: str, portfolio_value: float, 
                                 win_rate: float = 0.55, avg_win: float = 0.15, 
                                 avg_loss: float = 0.10, confidence_score: float = 0.8) -> Dict:
        """
        Calculate optimal position size using Modified Kelly Criterion.
        
        Args:
            ticker: Asset ticker
            portfolio_value: Total portfolio value in EUR
            win_rate: Historical win probability (0.0 - 1.0)
            avg_win: Average winning trade return (e.g. 0.15 = 15%)
            avg_loss: Average losing trade return (e.g. 0.10 = 10%)
            confidence_score: AI confidence in the signal (0.0 - 1.0)
        
        Returns:
            Dict with position sizing recommendations
        """
        rule = self.get_rule(ticker)
        max_allocation = rule.max_allocation_cap if rule else 20.0
        
        # Kelly Formula: f* = (bp - q) / b
        # Where: b = odds (avg_win/avg_loss), p = win prob, q = 1 - p
        if avg_loss == 0:
            avg_loss = 0.05  # Minimum 5% assumed loss
        
        b = avg_win / avg_loss  # Win/Loss ratio
        p = win_rate
        q = 1 - p
        
        kelly_pct = ((b * p) - q) / b
        
        # Clamp Kelly (can be negative if expected value is negative)
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
        
        # Half-Kelly (safer, commonly used in practice)
        half_kelly_pct = kelly_pct / 2
        
        # Apply confidence modifier
        adjusted_pct = half_kelly_pct * confidence_score
        
        # Respect max allocation cap
        final_pct = min(adjusted_pct * 100, max_allocation)
        
        # Calculate EUR amount
        position_eur = portfolio_value * (final_pct / 100)
        
        # Minimum trade size check (avoid trades smaller than €50)
        if position_eur < 50:
            final_pct = 0
            position_eur = 0
            reason = "Position too small (<€50)"
        else:
            reason = f"Kelly {kelly_pct*100:.1f}% → Half-Kelly {half_kelly_pct*100:.1f}% × Conf {confidence_score:.0%}"
        
        return {
            "ticker": ticker,
            "kelly_raw": round(kelly_pct * 100, 2),
            "half_kelly": round(half_kelly_pct * 100, 2),
            "final_allocation_pct": round(final_pct, 2),
            "position_eur": round(position_eur, 0),
            "max_cap": max_allocation,
            "reason": reason
        }
    
    def get_position_size_for_signal(self, ticker: str, portfolio_value: float, 
                                      ml_confidence: float = 0.6, 
                                      historical_win_rate: float = None) -> str:
        """
        Generate a human-readable position sizing recommendation for AI prompts.
        
        Args:
            ticker: Asset ticker
            portfolio_value: Total portfolio value in EUR
            ml_confidence: ML model confidence (0.0 - 1.0)
            historical_win_rate: Optional win rate from historical signals
        
        Returns:
            String recommendation for injection into AI prompt
        """
        # Default win rate if not provided
        win_rate = historical_win_rate if historical_win_rate else 0.55
        
        # Adjust based on strategy type
        rule = self.get_rule(ticker)
        if rule:
            if rule.strategy_type == StrategyType.SWING:
                avg_win = 0.20  # Swing traders aim for 20%
                avg_loss = 0.10
            elif rule.strategy_type == StrategyType.LONG_TERM:
                avg_win = 0.30  # Long term holders expect larger gains
                avg_loss = 0.15
            else:  # ACCUMULATE
                avg_win = 0.15
                avg_loss = 0.08
        else:
            avg_win = 0.15
            avg_loss = 0.10
        
        result = self.calculate_position_size(
            ticker, portfolio_value, win_rate, avg_win, avg_loss, ml_confidence
        )
        
        if result["position_eur"] == 0:
            return f"[Position Size {ticker}]: Skip trade ({result['reason']})"
        
        return (
            f"[Position Size {ticker}]: €{result['position_eur']:.0f} "
            f"({result['final_allocation_pct']:.1f}% of portfolio) - {result['reason']}"
        )


# =============================================================================
# Standalone Test
# =============================================================================
    
    def get_market_regime(self, economist) -> Dict:
        """
        Analyze Macro data to determine Market Regime (Risk ON/OFF).
        Arg: economist instance
        """
        # Get risk level from Economist (LOW/MEDIUM/HIGH)
        risk_level = economist.check_risk_level()
        
        # Determine Regime
        regime = "NEUTRAL"
        adjustments = {}
        
        if risk_level == "HIGH":
            regime = "RISK_OFF"
            description = "🐻 BEARISH / DEFENSIVE"
            adjustments = {
                "Crypto": -0.5,  # Reduce Crypto target by 50%
                "Technology": -0.3, # Reduce Tech by 30%
                "ETF": 0.2,      # Increase Safe Haven by 20%
                "Cash": 0.5      # Increase Cash
            }
        elif risk_level == "LOW":
            regime = "RISK_ON"
            description = "🐂 BULLISH / AGGRESSIVE"
            adjustments = {
                "Crypto": 0.2,   # Increase Crypto target by 20%
                "Technology": 0.1, # Increase Tech by 10%
                "ETF": -0.1      # Reduce Safe Haven slightly
            }
        else:
            description = "⚖️ NEUTRAL / CAUTIOUS"
            adjustments = {}
            
        return {
            "type": regime,
            "description": description,
            "adjustments": adjustments,
            "risk_level": risk_level
        }

    def get_dynamic_target(self, ticker: str, base_target: float, regime: Dict) -> float:
        """
        Calculate dynamic target allocation based on Market Regime.
        """
        # Map ticker to sector (Simplified logic, could be in DB)
        sector = "Other"
        if ticker in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            sector = "Crypto"
        elif ticker in ["NVDA", "AAPL", "MSFT", "GOOGL", "META"]:
            sector = "Technology"
        elif "EUNL" in ticker or "ETF" in ticker:
            sector = "ETF"
            
        # Apply adjustment factor
        adjustment = regime.get("adjustments", {}).get(sector, 0.0)
        
        # Calculate new target
        dynamic_target = base_target * (1 + adjustment)
        
        return round(dynamic_target, 1)

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    sm = StrategyManager()
    
    # Mock Economist for test
    class MockEconomist:
        def check_risk_level(self): return "HIGH"
        
    regime = sm.get_market_regime(MockEconomist())
    print(f"Regime: {regime['description']}")
    
    # Test Dynamic Target (Risk OFF -> Crypto should go down)
    base_btc = 10.0
    dyn_btc = sm.get_dynamic_target("BTC-USD", base_btc, regime)
    print(f"BTC Target: Base {base_btc}% -> Dynamic {dyn_btc}% (Risk OFF)")
