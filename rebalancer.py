"""
Rebalancer Module - Daily Portfolio Health & Rebalancing Suggestions
====================================================================
Provides intelligent rebalancing analysis with AI-powered suggestions.
Runs daily at 7:00 AM CET before market open.
"""

import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("Rebalancer")

class Rebalancer:
    """
    Analyzes portfolio allocations and generates rebalancing suggestions.
    Uses dynamic target allocation based on portfolio composition.
    """
    
    # Default sector targets (can be customized per user in future)
    DEFAULT_TARGETS = {
        "Crypto": 30.0,
        "Technology": 25.0,
        "ETF": 25.0,
        "Consumer Cyclical": 10.0,
        "Other": 10.0
    }
    
    # Deviation threshold for alerts (percentage points)
    DEVIATION_THRESHOLD = 10.0  # Alert if sector is ±10% from target
    
    # Trade Republic Cost Structure
    TRADE_FEE = 1.0  # €1 per trade (buy or sell)
    CAPITAL_GAINS_TAX = 0.26  # 26% Italian capital gains tax on profits
    MIN_PROFITABLE_TRADE = 50.0  # Minimum trade size where fee is <2% of trade
    
    def __init__(self):
        from db_handler import DBHandler
        from market_data import MarketData
        from advisor import Advisor
        
        self.db = DBHandler()
        self.market = MarketData()
        self.advisor = Advisor()
        
        # Level 9: Active AI Manager
        from economist import Economist
        from strategy_manager import StrategyManager
        self.economist = Economist()
        self.strategy_manager = StrategyManager()
        
        # Gemini for AI suggestions
        self.api_key = os.environ.get("GEMINI_API_KEY")
    
    def get_portfolio_analysis(self) -> Dict:
        """
        Get comprehensive portfolio analysis with allocations.
        
        Returns:
            {
                "total_value": 10000.0,
                "assets": [
                    {"ticker": "BTC-USD", "value": 3000, "pnl_pct": 15.5, "allocation": 30.0, "sector": "Crypto"},
                    ...
                ],
                "sector_allocation": {"Crypto": 30.0, "Technology": 25.0, ...},
                "deviations": {"Crypto": +5.0, "Technology": -3.0, ...}  # vs target
            }
        """
        portfolio = self.db.get_portfolio()
        
        if not portfolio:
            return {"total_value": 0, "assets": [], "sector_allocation": {}, "deviations": {}}
        
        assets = []
        sector_values = {}
        total_value = 0.0
        
        for item in portfolio:
            ticker = item.get('ticker', 'UNKNOWN')
            qty = float(item.get('quantity', 0))
            avg_price = float(item.get('avg_price', 0))
            
            # Get current price in EUR
            current_price, _ = self.market.get_smart_price_eur(ticker)
            if current_price <= 0:
                current_price = avg_price  # Fallback
            
            value = qty * current_price
            total_value += value
            
            # Calculate PnL
            cost_basis = qty * avg_price
            pnl_pct = ((value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0.0
            
            # Get sector
            sector = self.advisor.get_sector(ticker)
            sector_values[sector] = sector_values.get(sector, 0.0) + value
            
            # Get RSI (for AI context)
            rsi = None
            try:
                tech_data = self.market.get_technical_summary(ticker)
                if "RSI:" in tech_data:
                    rsi_str = tech_data.split("RSI:")[1].split(",")[0].strip()
                    rsi = float(rsi_str) if rsi_str != "N/A" else None
            except:
                pass
            
            # Calculate unrealized gain and potential tax cost
            unrealized_gain = value - cost_basis
            potential_tax = unrealized_gain * self.CAPITAL_GAINS_TAX if unrealized_gain > 0 else 0
            net_if_sold = value - potential_tax - self.TRADE_FEE  # Net proceeds after selling
            
            assets.append({
                "ticker": ticker,
                "quantity": qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "value": value,
                "pnl_pct": pnl_pct,
                "pnl_eur": unrealized_gain,
                "potential_tax": potential_tax,
                "net_if_sold": net_if_sold,
                "sector": sector,
                "rsi": rsi
            })
        
        # Calculate allocations and deviations
        sector_allocation = {}
        deviations = {}
        
        for sector, value in sector_values.items():
            allocation = (value / total_value * 100) if total_value > 0 else 0
            sector_allocation[sector] = allocation
            
            # Calculate deviation from target
            target = self.DEFAULT_TARGETS.get(sector, self.DEFAULT_TARGETS.get("Other", 10.0))
            deviations[sector] = allocation - target
        
        # Add allocation % to each asset
        for asset in assets:
            asset["allocation"] = (asset["value"] / total_value * 100) if total_value > 0 else 0
        
        return {
            "total_value": total_value,
            "assets": sorted(assets, key=lambda x: -x["value"]),  # Sort by value desc
            "sector_allocation": sector_allocation,
            "deviations": deviations
        }
    
    def generate_rebalance_suggestions(self, analysis: Dict) -> List[str]:
        """
        Generate AI-powered rebalancing suggestions.
        
        Returns list of actionable suggestions.
        """
        if analysis["total_value"] == 0:
            return ["Portfolio is empty. Start accumulating assets."]
        
        suggestions = []
        
        # 1. Check sector deviations
        for sector, deviation in analysis["deviations"].items():
            if abs(deviation) > self.DEVIATION_THRESHOLD:
                if deviation > 0:
                    suggestions.append(f"⚠️ **{sector}** is overweight by {deviation:.1f}%. Consider trimming positions.")
                else:
                    suggestions.append(f"📉 **{sector}** is underweight by {abs(deviation):.1f}%. Look for opportunities to add.")
        
        # 2. Check individual asset concentration
        for asset in analysis["assets"]:
            if asset["allocation"] > 25:
                suggestions.append(f"🎯 **{asset['ticker']}** is {asset['allocation']:.1f}% of portfolio. Consider diversifying.")
        
        # 3. Check for high profit-taking opportunities
        for asset in analysis["assets"]:
            if asset["pnl_pct"] > 50 and asset["allocation"] > 10:
                suggestions.append(f"💰 **{asset['ticker']}** is up {asset['pnl_pct']:.1f}%. Consider taking 20% profit.")
        
        # 4. Check for tax-loss harvesting (Italy specific)
        for asset in analysis["assets"]:
            if asset["pnl_pct"] < -30:
                if asset["sector"] == "Crypto":
                    suggestions.append(f"📉 **{asset['ticker']}** is down {asset['pnl_pct']:.1f}%. Consider tax-loss harvest before 2026 (Crypto tax 26%→33%).")
                else:
                    suggestions.append(f"📉 **{asset['ticker']}** is down {asset['pnl_pct']:.1f}%. Potential tax-loss harvest opportunity.")
        
        # 5. Use AI for deeper analysis if available
        if self.api_key and len(suggestions) < 3:
            ai_suggestion = self._get_ai_suggestion(analysis)
            if ai_suggestion:
                suggestions.append(ai_suggestion)
        
        if not suggestions:
            suggestions.append("✅ Portfolio is well-balanced. No rebalancing needed.")
        
        return suggestions
    
    def _get_ai_suggestion(self, analysis: Dict) -> Optional[str]:
        """
        Get AI-generated ACTIONABLE rebalancing strategy.
        Returns specific trade recommendations.
        """
        try:
            from google import genai
            
            client = genai.Client(api_key=self.api_key)
            
            # 1. Portfolio summary with RSI and cost info
            def rsi_label(rsi):
                if rsi is None: return ""
                if rsi > 70: return f", RSI: {rsi:.0f} ⚠️OVERBOUGHT"
                if rsi < 30: return f", RSI: {rsi:.0f} 🔥OVERSOLD"
                return f", RSI: {rsi:.0f}"
            
            def cost_label(a):
                if a['pnl_eur'] > 0:
                    return f" [Se vendi: Tax €{a['potential_tax']:.0f} + Fee €1]"
                return " [Se vendi: Fee €1]"
            
            assets_summary = "\n".join([
                f"- {a['ticker']}: €{a['value']:.0f} ({a['allocation']:.1f}%), PnL: {a['pnl_pct']:+.1f}% (€{a['pnl_eur']:+.0f}), Sector: {a['sector']}{rsi_label(a.get('rsi'))}{cost_label(a)}"
                for a in analysis["assets"][:10]
            ])
            
            sector_summary = "\n".join([
                f"- {s}: {a:.1f}% (Target: {self.DEFAULT_TARGETS.get(s, 10):.0f}%)"
                for s, a in analysis["sector_allocation"].items()
            ])
            
            # 2. Get recent signals from DB
            signals_text = "Nessun segnale recente."
            try:
                recent = self.db.supabase.table("predictions") \
                    .select("ticker, sentiment, confidence_score, target_price") \
                    .gte("created_at", (datetime.now() - timedelta(days=7)).isoformat()) \
                    .order("created_at", desc=True) \
                    .limit(10) \
                    .execute()
                if recent.data:
                    signals_text = "\n".join([
                        f"- {s['ticker']}: {s['sentiment']} (Conf: {s['confidence_score']:.0%}, Target: {s.get('target_price', 'N/A')})"
                        for s in recent.data
                    ])
            except Exception as e:
                logger.warning(f"Could not fetch signals: {e}")
            
            # 3. Market context
            market_context = "N/A"
            try:
                from economist import Economist
                from whale_watcher import WhaleWatcher
                
                eco = Economist()
                whale = WhaleWatcher()
                
                macro = eco.get_dashboard_stats()
                whale_data = whale.get_dashboard_stats()
                market_status = eco.get_market_status()
                
                market_context = f"VIX: {macro.get('vix', 'N/A')}, Macro Risk: {macro.get('risk_level', 'N/A')}, Whale: {whale_data.get('status', 'N/A')}"
                market_hours = f"🇺🇸 US: {market_status['us_stocks']}, 🇪🇺 EU: {market_status['eu_stocks']}, ₿ Crypto: {market_status['crypto']}"
            except Exception as e:
                logger.warning(f"Market context failed: {e}")
                market_hours = "N/A"
            
            # 4. L2 Predictive Context: Sector Rotation + Market Regime
            l2_context = ""
            try:
                from sector_rotation import SectorRotationTracker
                from market_regime import MarketRegimeClassifier
                
                # Sector Rotation
                rotation = SectorRotationTracker().analyze()
                rotation_text = f"Signal: {rotation.get('rotation_signal')} ({rotation.get('confidence'):.0%}). Leading: {', '.join(rotation.get('leading', []))}. Lagging: {', '.join(rotation.get('lagging', []))}"
                
                # Market Regime
                regime = MarketRegimeClassifier().classify()
                regime_text = f"{regime.get('regime')} ({regime.get('confidence'):.0%}) - {regime.get('recommendation')}"
                
                l2_context = f"""
**L2 PREDICTIVE (Auto-Updated):**
- Sector Rotation: {rotation_text}
- Market Regime: {regime_text}
"""
                logger.info(f"Rebalance: L2 context added - {rotation.get('rotation_signal')}, {regime.get('regime')}")
            except Exception as e:
                logger.warning(f"L2 context failed for rebalance: {e}")
            
            # 5. L3 Pattern Recognition for portfolio assets
            l3_context = ""
            try:
                from pattern_recognition import PatternRecognizer
                pr = PatternRecognizer()
                
                pattern_findings = []
                for asset in analysis["assets"][:5]:  # Top 5 by value
                    ticker = asset['ticker']
                    bias, modifier = pr.get_pattern_bias(ticker)
                    if bias != "NEUTRAL":
                        patterns = pr.detect_patterns(ticker)
                        if patterns:
                            top_pattern = patterns[0]
                            pattern_findings.append(
                                f"- {ticker}: {top_pattern.pattern_type.value} ({bias}, {int(top_pattern.confidence*100)}% conf) → Target: {top_pattern.target_move_pct:+.1f}%"
                            )
                
                if pattern_findings:
                    l3_context = f"""
**L3 PATTERN RECOGNITION:**
{chr(10).join(pattern_findings)}
"""
                    logger.info(f"Rebalance: L3 patterns found for {len(pattern_findings)} assets")
            except Exception as e:
                logger.warning(f"L3 context failed for rebalance: {e}")
            
            # 6. Strategy Governance Context (Level 9)
            strategy_context = ""
            regime_desc = "NEUTRAL"
            risk_level = "UNKNOWN"
            
            try:
                # Get Macro Regime
                regime_data = self.strategy_manager.get_market_regime(self.economist)
                regime_desc = regime_data.get('description', 'NEUTRAL')
                risk_level = regime_data.get('risk_level', 'UNKNOWN')
                
                strategy_lines = []
                for asset in analysis["assets"][:10]:  # Top 10 assets
                    ticker = asset['ticker']
                    pnl = asset.get('pnl_pct', 0)
                    alloc = asset.get('allocation', 0)
                    
                    # Calculate Dynamic Target
                    rule = self.strategy_manager.get_rule(ticker)
                    base_target = rule.target_allocation_pct if rule else 10.0
                    dynamic_target = self.strategy_manager.get_dynamic_target(ticker, base_target, regime_data)
                    
                    ctx = self.strategy_manager.get_strategy_context(ticker, pnl, alloc)
                    
                    # Inject Dynamic Target info if different from base
                    if dynamic_target != base_target:
                        arrow = "⬆️" if dynamic_target > base_target else "⬇️"
                        ctx += f"\n   ⚡ **Dynamic Target ({regime_desc}):** {base_target}% -> {dynamic_target}% {arrow}"
                    
                    if "No specific rule" not in ctx:
                        strategy_lines.append(ctx)
                
                if strategy_lines:
                    strategy_context = "\n\n**🛡️ USER STRATEGY RULES (MANDATORY):**\n" + "\n".join(strategy_lines)
                    strategy_context += "\n**⚠️ CRITICAL: You MUST respect these rules. Do NOT suggest SELL for LONG_TERM assets!**"
                    logger.info(f"Rebalance: Strategy context added for {len(strategy_lines)} assets")
            except Exception as e:
                logger.warning(f"Strategy context failed for rebalance: {e}")
            
            # 7. Portfolio Backtest & Correlation Context (Level 11)
            backtest_context = ""
            try:
                from portfolio_backtest import PortfolioBacktest
                bt = PortfolioBacktest()
                
                # Prepare portfolio data for backtest
                bt_portfolio = []
                for asset in analysis["assets"]:
                    bt_portfolio.append({
                        "ticker": asset['ticker'],
                        "quantity": asset.get('qty', 0),
                        "avg_price_eur": asset.get('avg_price', 0),
                        "current_price": asset.get('price', asset.get('avg_price', 0))
                    })
                
                # Run 90-day backtest (shorter for speed)
                bt_results = bt.run_backtest(bt_portfolio, period_days=90)
                
                if "error" not in bt_results:
                    sharpe = bt_results.get('sharpe_ratio', 0)
                    max_dd = bt_results.get('max_drawdown_pct', 0)
                    win_rate = bt_results.get('win_rate', 0)
                    
                    sharpe_status = "🟢 GOOD" if sharpe >= 1.0 else "🟡 MEDIOCRE" if sharpe >= 0 else "🔴 POOR"
                    dd_status = "🟢 SAFE" if max_dd >= -15 else "🟡 MODERATE" if max_dd >= -25 else "🔴 HIGH RISK"
                    
                    backtest_context = f"""
**📊 PORTFOLIO HISTORICAL PERFORMANCE (90 days):**
- Sharpe Ratio: {sharpe:.2f} ({sharpe_status})
- Max Drawdown: {max_dd:.1f}% ({dd_status})
- Win Rate: {win_rate:.0%}
"""
                    logger.info(f"Rebalance: Backtest context added (Sharpe: {sharpe:.2f})")
                
                # Add Correlation Analysis
                tickers = [asset['ticker'] for asset in analysis["assets"][:10]]
                if len(tickers) >= 2:
                    corr_data = self.market_data.calculate_correlation_matrix(tickers)
                    if corr_data.get('high_correlation_pairs'):
                        backtest_context += "\n**⚠️ HIGH CORRELATION WARNING (>70%):**\n"
                        for t1, t2, corr in corr_data['high_correlation_pairs'][:3]:
                            backtest_context += f"- {t1} ↔ {t2}: {corr:.0%} (consider diversifying)\n"
                    
                    div_score = corr_data.get('diversification_score', 50)
                    div_status = "🟢 GOOD" if div_score >= 70 else "🟡 MEDIOCRE" if div_score >= 50 else "🔴 POOR"
                    backtest_context += f"- Diversification Score: {div_score}/100 ({div_status})\n"
                    
            except Exception as e:
                logger.warning(f"Backtest context failed for rebalance: {e}")
            
            prompt = f"""
            Sei un Portfolio Manager italiano focalizzato sulla MASSIMIZZAZIONE DEI PROFITTI NETTI (Post-Tax & Fees).
            
            **STRUTTURA COSTI (Trade Republic Italia):**
            - Commissione: €1 fisso per ogni ordine (BUY o SELL)
            - Tasse: 26% sulle plusvalenze (Capital Gains Tax)
            - Soglia minima convenienza: Un trade deve generare valore NETTO > dei costi.
            - Esempio: Vendere €100 con €10 gain non conviene (Tasse €2.6 + Fee €1 = €3.6 di costi, 36% del gain eroso).
            
            **ORARI MERCATO ({market_status.get('current_time_italy', 'N/A')}):**
            {market_hours}
            
            **PORTAFOGLIO ATTUALE:** €{analysis['total_value']:.0f}
            {assets_summary}
            
            **ALLOCAZIONE SETTORI:**
            {sector_summary}
            
            **SEGNALI RECENTI (ultimi 7gg) - PRIORITÀ ALTA:**
            {signals_text}
            
            **CONTESTO MERCATO:**
            {market_context}
            
            {l2_context}
            
            {l3_context}
            
            **🔬 MACRO REGIME ANALYSIS (DeepSeek R1):**
            Based on VIX and Fed data from Economist:
            - Regime: {regime_desc}
            - Risk Level: {risk_level}
            
            **🛡️ DYNAMIC STRATEGY RULES (MANDATORY):**
            {strategy_context}
            
            {backtest_context}
            
            **OBIETTIVO PRINCIPALE: MASSIMIZZARE I PROFITTI NETTI**
            
            **ISTRUZIONI:**
            1. Genera MAX 3 azioni concrete nel formato:
               🟢 BUY €XXX TICKER (NUOVO) - Motivo breve ← SOLO per asset NON in portafoglio
               🟢 ACCUMULATE €XXX TICKER - Motivo breve ← SOLO per asset GIÀ in portafoglio
               🟡 HOLD TICKER - Motivo breve
               🔴 TRIM XX% TICKER - Motivo breve (Net profit after tax/fees: €XX)
            
            2. Poi aggiungi UNA strategia sintetica (1 frase) per €500 di capitale fresco.
            
            **REGOLE CRITICHE PER MASSIMIZZARE PROFITTI:**
            - **COST CHECK (FONDAMENTALE):** NON suggerire SELL/TRIM se il costo totale (26% Tax + €1 Fee) supera il 50% del profitto, a meno che non sia per stop-loss o rotazione critica.
            - **PRIORITÀ 1:** Segui i SEGNALI RECENTI ad alta confidenza (BUY/ACCUMULATE ≥75%)
            - **PRIORITÀ 2:** Accumula asset con RSI OVERSOLD (<30)
            - **PRIORITÀ 3:** Trimma asset con RSI OVERBOUGHT (>70) SOLO SE il PnL netto giustifica la tassa.
            - **PRIORITÀ 4:** (Solo dopo) Considera il bilanciamento settoriale
            - **PRIORITÀ 5:** Se Sharpe < 0 o Diversification Score < 60, suggerisci di ridurre asset altamente correlati
            
            - **TICKER SUGGERIBILI:** Asset nel portfolio: {', '.join([a['ticker'] for a in analysis['assets']])}
            - **ECCEZIONE:** Se un segnale recente (BUY ≥80% conf) è per un ticker NON nel portfolio, suggeriscilo comunque indicando "(NUOVO)"
            - Per capitale fresco, indica asset specifici se ci sono segnali forti, altrimenti settori
            - NON suggerire SELL totale (solo TRIM parziale per prendere profitti)
            
            Rispondi SOLO con le azioni, senza preamboli.
            """
            
            # Use Brain's fallback system (respects APP_MODE: PREPROD/PROD)
            # Use 'deepseek-reasoner' (R1) for complex logic, fallback to Gemini
            from brain import Brain
            brain = Brain()
            # DeepSeek R1 enabled for GitHub Actions (Long-running async task)
            response_text = brain._generate_with_fallback(prompt, json_mode=False, model="deepseek-reasoner")
            
            return response_text.strip()
            
        except Exception as e:
            logger.warning(f"AI suggestion failed: {e}")
            return None
    
    def format_rebalance_report(self) -> str:
        """
        Generate formatted rebalancing report for Telegram.
        """
        analysis = self.get_portfolio_analysis()
        
        if analysis["total_value"] == 0:
            return "📊 **Portfolio Rebalancing**\n\n❌ Portafoglio vuoto."
        
        report = "📊 **Portfolio Rebalancing Strategy**\n"
        report += "━" * 28 + "\n\n"
        
        # Total value
        report += f"💰 **Valore Totale:** €{analysis['total_value']:,.0f}\n\n"
        
        # Macro Regime & Dynamic Targets (Level 9)
        try:
            regime_data = self.strategy_manager.get_market_regime(self.economist)
            regime_desc = regime_data.get('description', 'NEUTRAL')
            risk_info = regime_data.get('risk_level', 'UNKNOWN')
            
            # Handle tuple (level, reason) from enhanced check_risk_level
            if isinstance(risk_info, tuple):
                risk_level, risk_reason = risk_info
            else:
                risk_level = risk_info
                risk_reason = ""
            
            report += f"🔬 **MACRO REGIME:** {regime_desc}\n"
            report += f"   📊 Risk Level: {risk_level}"
            if risk_reason:
                report += f" | {risk_reason}"
            report += "\n"
            
            # Show Dynamic Targets for top assets
            dynamic_lines = []
            for asset in analysis["assets"][:6]:
                ticker = asset['ticker']
                rule = self.strategy_manager.get_rule(ticker)
                if rule:
                    base_target = rule.target_allocation_pct
                    dynamic_target = self.strategy_manager.get_dynamic_target(ticker, base_target, regime_data)
                    
                    if rule.strategy_type.value == "LONG_TERM":
                        dynamic_lines.append(f"   🔵 {ticker}: {base_target}% (LONG_TERM)")
                    elif dynamic_target != base_target:
                        arrow = "⬆️" if dynamic_target > base_target else "⬇️"
                        dynamic_lines.append(f"   ⚡ {ticker}: {base_target}% → {dynamic_target}% {arrow}")
                    else:
                        dynamic_lines.append(f"   🟢 {ticker}: {base_target}%")
            
            if dynamic_lines:
                report += "   ⚡ **Dynamic Targets:**\n"
                for line in dynamic_lines:
                    report += f"{line}\n"
            
            report += "\n"
        except Exception as e:
            logger.warning(f"Macro regime display failed: {e}")
        
        # AI Strategy (FIRST - most important)
        ai_strategy = self._get_ai_suggestion(analysis)
        if ai_strategy:
            report += "🎯 **PIANO D'AZIONE (Hybrid AI - R1):**\n"
            report += ai_strategy + "\n\n"
        
        # Sector allocation (compact)
        report += "📈 **Allocazione Settori:**\n"
        for sector, alloc in sorted(analysis["sector_allocation"].items(), key=lambda x: -x[1]):
            target = self.DEFAULT_TARGETS.get(sector, 10)
            deviation = analysis["deviations"].get(sector, 0)
            
            emoji = "🟢" if abs(deviation) < self.DEVIATION_THRESHOLD else "🟡" if deviation > 0 else "🔴"
            report += f"  {emoji} {sector}: {alloc:.1f}% (target: {target:.0f}%)\n"
        
        # Top 5 assets
        report += "\n📦 **Top Holdings:**\n"
        for asset in analysis["assets"][:5]:
            pnl_emoji = "🟢" if asset["pnl_pct"] >= 0 else "🔴"
            report += f"  {pnl_emoji} **{asset['ticker']}**: €{asset['value']:.0f} ({asset['allocation']:.1f}%) | {asset['pnl_pct']:+.1f}%\n"
        
        # Rule-based suggestions (secondary)
        suggestions = self.generate_rebalance_suggestions(analysis)
        if suggestions and suggestions[0] != "✅ Portfolio is well-balanced. No rebalancing needed.":
            report += "\n💡 **Note Aggiuntive:**\n"
            for s in suggestions[:3]:  # Max 3
                report += f"{s}\n"
        
        report += "\n" + "━" * 28
        report += f"\n_Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        
        return report
    
    def get_flash_recommendation(self) -> Optional[str]:
        """
        Get a single cost-effective rebalancing tip for the Hunt report.
        Only returns if there's a strong opportunity covering fees+tax.
        Uses DeepSeek Reasoner for maximum precision if available.
        """
        try:
            analysis = self.get_portfolio_analysis()
            if analysis["total_value"] == 0: return None
            
            # Simple rule-based check for obvious opportunities
            for asset in analysis["assets"]:
                # Check for Flash TRIM (Profit > €50, RSI > 75)
                rsi = asset.get("rsi")
                if rsi is None: rsi = 50
                
                pnl = asset.get("pnl_eur")
                if pnl is None: pnl = 0

                # If rule based trigger found, pass to AI for confirmation
                if pnl > 50 and rsi > 75:
                    tax = asset.get("potential_tax")
                    if tax is None: tax = 0
                    
                    cost = tax + self.TRADE_FEE
                    net_profit = pnl - cost
                    if net_profit > 20: # Worth doing
                        # Ask DeepSeek Reasoner for a one-line strategic confirmation
                        try:
                            prompt = (
                                f"I am considering FLASH TRIMMING {asset['ticker']} because RSI is {rsi:.0f} (overbought) "
                                f"and I have €{pnl:.0f} profit (€{net_profit:.0f} net after tax). "
                                f"Total portfolio value: €{analysis['total_value']:.0f}. "
                                "Is this a smart move? Answer in 1 short sentence starting with '💡 FLASH TIP:'."
                            )
                            from brain import Brain
                            return Brain()._generate_with_fallback(prompt, json_mode=False, model="deepseek-reasoner").strip()
                        except:
                            # Fallback to simple rule text if AI fails
                            return f"📉 **FLASH TRIM Opportunity**: Vendi parte di {asset['ticker']} (RSI {rsi:.0f}). Net Gain: ~€{net_profit:.0f} (dopo fees/tax)."
            
            # Check for Flash BUY (Strong signal + Cash available)
            # This is handled by main hunt logic, so we focus on Portfolio Trim/Rotation here
            
            return None
        except Exception as e:
            logger.warning(f"Flash recommendation failed: {e}")
            return None

    async def run_daily(self):
        """
        Run daily rebalancing analysis and send to Telegram.
        Called by GitHub Actions.
        If TARGET_CHAT_ID is set in env, sends only to that user.
        """
        from telegram_bot import TelegramNotifier
        
        logger.info("Running daily rebalancing analysis...")
        
        # Check for Targeted Execution (e.g. from /rebalance command)
        target_chat_id = os.environ.get("TARGET_CHAT_ID")
        
        try:
            report = self.format_rebalance_report()
            
            notifier = TelegramNotifier()
            
            if target_chat_id:
                logger.info(f"Sending targeted report to {target_chat_id}")
                await notifier.send_message(chat_id=target_chat_id, message=report)
            else:
                # Default: Broadcast to alert list (Daily Schedule)
                await notifier.send_alert(report)
            
            # Log to DB
            self.db.log_system_event("INFO", "Rebalancer", "Daily rebalancing report sent")
            
            logger.info("Daily rebalancing report sent successfully")
            
        except Exception as e:
            logger.error(f"Daily rebalancing failed: {e}")
            self.db.log_system_event("ERROR", "Rebalancer", f"Failed: {e}")


# CLI entry point for GitHub Actions
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    rebalancer = Rebalancer()
    asyncio.run(rebalancer.run_daily())
