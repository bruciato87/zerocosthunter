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
            ai_signal: AI suggested action (BUY, SELL, ACCUMULATE, TRIM, HOLD)
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
        
        # =========================================================================
        # RULE 6: ATR-Based Dynamic Stop (Level 11)
        # =========================================================================
        try:
            from market_data import MarketData
            md = MarketData()
            atr_data = md.calculate_atr(ticker)
            
            if atr_data['atr_pct'] > 0 and atr_data['volatility'] != 'unknown':
                dynamic_stop = -atr_data['suggested_stop']  # Negative value
                
                # Check if current loss exceeds dynamic ATR-based stop
                if current_pnl_pct <= dynamic_stop:
                    if ai_signal not in ["SELL", "PANIC SELL"]:
                        return ("SELL", f"⚠️ ATR Stop: {current_pnl_pct:.1f}% hit dynamic stop of {dynamic_stop:.1f}% (Volatility: {atr_data['volatility']})")
        except Exception as e:
            logger.warning(f"ATR stop check failed for {ticker}: {e}")
        
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
# Level 9: Active AI Portfolio Management - Dynamic Regime Targets
# =============================================================================

    # Target allocations per market regime (%)
    REGIME_TARGETS = {
        "BULL": {
            "Crypto": 35.0,
            "Technology": 30.0,
            "ETF": 20.0,
            "Cash": 10.0,
            "Other": 5.0
        },
        "BEAR": {
            "Crypto": 15.0,
            "Technology": 20.0,
            "ETF": 30.0,
            "Cash": 30.0,
            "Other": 5.0
        },
        "SIDEWAYS": {
            "Crypto": 25.0,
            "Technology": 25.0,
            "ETF": 25.0,
            "Cash": 20.0,
            "Other": 5.0
        },
        "ACCUMULATION": {
            "Crypto": 40.0,
            "Technology": 25.0,
            "ETF": 20.0,
            "Cash": 10.0,
            "Other": 5.0
        },
        "DISTRIBUTION": {
            "Crypto": 10.0,
            "Technology": 15.0,
            "ETF": 35.0,
            "Cash": 35.0,
            "Other": 5.0
        }
    }
    
    # Safety Caps (min%, max%)
    SAFETY_CAPS = {
        "Crypto": (5.0, 50.0),
        "Technology": (10.0, 40.0),
        "ETF": (10.0, 50.0),
        "Cash": (5.0, 50.0),
        "Other": (0.0, 20.0)
    }
    
    def get_market_regime(self, economist=None) -> Dict:
        """
        Level 9: Analyze market to determine regime using sophisticated classifier.
        Falls back to economist-based detection if classifier fails.
        
        Returns:
            {
                "regime": "BULL" | "BEAR" | "SIDEWAYS" | "ACCUMULATION" | "DISTRIBUTION",
                "confidence": 0.0-1.0,
                "description": "🐂 BULLISH / AGGRESSIVE",
                "targets": {"Crypto": 35.0, "Technology": 30.0, ...},
                "signals": ["SPY > SMA200", "VIX low", ...],
                "recommendation": "aggressive" | "normal" | "defensive"
            }
        """
        try:
            from market_regime import MarketRegimeClassifier
            classifier = MarketRegimeClassifier()
            regime_data = classifier.classify()
            
            # Get regime name
            regime = regime_data.get("regime", "SIDEWAYS")
            confidence = regime_data.get("confidence", 0.5)
            recommendation = regime_data.get("recommendation", "normal")
            signals = regime_data.get("signals", [])
            
            # Map regime to description emoji
            descriptions = {
                "BULL": "🐂 BULLISH / AGGRESSIVE",
                "BEAR": "🐻 BEARISH / DEFENSIVE",
                "SIDEWAYS": "⚖️ SIDEWAYS / NEUTRAL",
                "ACCUMULATION": "💎 ACCUMULATION / BUY DIPS",
                "DISTRIBUTION": "⚠️ DISTRIBUTION / TAKE PROFITS"
            }
            description = descriptions.get(regime, "⚖️ NEUTRAL")
            
            # Get dynamic targets for this regime
            targets = self.get_regime_targets(regime)
            
            logger.info(f"L9 Regime: {regime} ({confidence:.0%}) -> Targets: Crypto={targets.get('Crypto')}%, Tech={targets.get('Technology')}%")
            
            return {
                "regime": regime,
                "confidence": confidence,
                "description": description,
                "targets": targets,
                "signals": signals,
                "recommendation": recommendation,
                "risk_level": "HIGH" if regime in ["BEAR", "DISTRIBUTION"] else ("LOW" if regime in ["BULL", "ACCUMULATION"] else "MEDIUM")
            }
            
        except Exception as e:
            logger.warning(f"MarketRegimeClassifier failed, using fallback: {e}")
            
            # Fallback to economist-based detection
            if economist:
                risk_level = economist.check_risk_level()
                if risk_level == "HIGH":
                    regime = "BEAR"
                elif risk_level == "LOW":
                    regime = "BULL"
                else:
                    regime = "SIDEWAYS"
            else:
                regime = "SIDEWAYS"
            
            return {
                "regime": regime,
                "confidence": 0.5,
                "description": f"⚖️ {regime} (Fallback)",
                "targets": self.get_regime_targets(regime),
                "signals": ["Fallback mode"],
                "recommendation": "normal",
                "risk_level": "MEDIUM"
            }
    
    def get_regime_targets(self, regime: str) -> Dict[str, float]:
        """
        Get target allocations for a specific regime with safety caps applied.
        """
        # Get base targets for regime (default to SIDEWAYS)
        targets = self.REGIME_TARGETS.get(regime, self.REGIME_TARGETS["SIDEWAYS"]).copy()
        
        # Apply safety caps
        for sector, (min_cap, max_cap) in self.SAFETY_CAPS.items():
            if sector in targets:
                targets[sector] = max(min_cap, min(max_cap, targets[sector]))
        
        return targets

    def get_dynamic_target(self, ticker: str, base_target: float, regime: Dict) -> float:
        """
        Calculate dynamic target allocation based on Market Regime.
        Uses the new regime targets system.
        """
        # Map ticker to sector
        sector = self._get_ticker_sector(ticker)
        
        # Get target from regime (if available) or use adjusted base
        regime_targets = regime.get("targets", {})
        if sector in regime_targets:
            return regime_targets[sector]
        
        # Fallback: apply old adjustment logic
        adjustment = regime.get("adjustments", {}).get(sector, 0.0)
        dynamic_target = base_target * (1 + adjustment)
        
        return round(dynamic_target, 1)
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """Map a ticker to its sector category."""
        ticker_upper = ticker.upper()
        
        # Crypto
        if any(c in ticker_upper for c in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "LINK"]):
            return "Crypto"
        
        # Technology
        tech_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "TSLA", "AMD", "INTC", "ASML"]
        if any(t in ticker_upper for t in tech_tickers):
            return "Technology"
        
        # ETF/Safe Haven
        if any(e in ticker_upper for e in ["EUNL", "IWDA", "VWCE", "GLD", "SLV", "TLT", "ETF"]):
            return "ETF"
        
        return "Other"

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
