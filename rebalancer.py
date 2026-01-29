"""
Rebalancer Module - Daily Portfolio Health & Rebalancing Suggestions
====================================================================
Provides intelligent rebalancing analysis with AI-powered suggestions.
Runs daily at 7:00 AM CET before market open.
"""

import logging
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from datetime import timezone as _timezone
    ZoneInfo = lambda x: _timezone(timedelta(hours=1)) if "Europe" in x else _timezone.utc

# [PHASE C.6] Ensure nest_asyncio is applied for nested Council calls
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

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
        
        # [PHASE C.6] Switch to higher-quota model for complex analysis
        # gemini-2.0-flash-lite has only 20 RPM, gemini-1.5-flash has 1500 RPM.
        self.ai_model = "gemini-1.5-flash"
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
        Generate quantitative rebalancing suggestions based on deviations.
        """
        total_value = analysis["total_value"]
        if total_value == 0:
            return ["Portfolio is empty. Start accumulating assets."]
        
        suggestions = []
        buys = []
        sells = []
        
        # 1. Sector Balancer (Quantitative)
        for sector, deviation in analysis["deviations"].items():
            # Calculate Euro amount needed to fix the deviation
            target_pct = self.DEFAULT_TARGETS.get(sector, 10.0)
            target_value = total_value * (target_pct / 100.0)
            current_value = total_value * (analysis.get("sector_allocation", {}).get(sector, 0.0) / 100.0)
            diff_eur = target_value - current_value
            
            if abs(deviation) >= self.DEVIATION_THRESHOLD or abs(diff_eur) >= 200:
                if diff_eur < -50: # Overweight -> Sell
                    assets_in_sector = [a for a in analysis["assets"] if a["sector"] == sector]
                    if assets_in_sector:
                        # [FIX] Exclude LONG_TERM assets from being sold
                        tradable_assets = []
                        for a in assets_in_sector:
                            rule = self.strategy_manager.get_rule(a['ticker'])
                            if rule and rule.strategy_type.value == "LONG_TERM":
                                continue
                            tradable_assets.append(a)
                        
                        if tradable_assets:
                            # Suggest trimming the largest one or one with high PnL
                            best_to_trim = sorted(tradable_assets, key=lambda x: -x["pnl_pct"])[0]
                            sells.append(f"🔴 **Vendi €{abs(diff_eur):.0f}** di **{best_to_trim['ticker']}** ({sector} è in eccesso del {abs(deviation):.1f}%)")
                        else:
                            logger.info(f"Rebalancer: Skipping sell suggestion for {sector} (all assets are LONG_TERM)")
                elif diff_eur > 50: # Underweight -> Buy
                    buys.append(f"🟢 **Compra €{diff_eur:.0f}** in **{sector}** (manca il {abs(deviation):.1f}% al target)")

        # 2. Individual Asset Concentration
        for asset in analysis["assets"]:
            if asset["allocation"] >= 30:
                # [FIX] Exclude LONG_TERM assets from concentration trim
                rule = self.strategy_manager.get_rule(asset['ticker'])
                if rule and rule.strategy_type.value == "LONG_TERM":
                    logger.info(f"Rebalancer: Skipping concentration trim for {asset['ticker']} (LONG_TERM)")
                    continue
                    
                target_alloc = 20.0
                trim_eur = asset["value"] - (total_value * (target_alloc / 100.0))
                if trim_eur >= 100:
                    sells.append(f"🔴 **Riduci €{trim_eur:.0f}** di **{asset['ticker']}** (troppa concentrazione: {asset['allocation']:.1f}%)")

        # 3. Specific Ops (Tax harvesting, Profit taking)
        for asset in analysis["assets"]:
            if asset["pnl_pct"] <= -35 and asset["value"] >= 100:
                suggestions.append(f"📉 **Tax-Loss Harvesting**: Considera di vendere e ricomprare **{asset['ticker']}** ({asset['pnl_pct']:.1f}%) per scaricare minusvalenze.")
            elif asset["pnl_pct"] >= 50 and asset["allocation"] >= 10:
                take_profit = asset["value"] * 0.2
                suggestions.append(f"💰 **Take Profit**: Vendi **€{take_profit:.0f}** di **{asset['ticker']}** (+{asset['pnl_pct']:.1f}%) per mettere al sicuro i guadagni.")

        # Combine items
        final_list = sells[:3] + buys[:3] + suggestions[:5]
        
        if not final_list:
            final_list.append("✅ Il portafoglio è ben bilanciato. Non sono necessarie azioni urgenti.")
        
        return final_list
    
    def _format_regime_targets(self, regime_data: Dict) -> str:
        """
        Format dynamic sector targets for AI prompt, comparing to current allocation.
        """
        targets = regime_data.get("targets", {})
        regime = regime_data.get("regime", "SIDEWAYS")
        
        # Get current sector allocations from portfolio
        analysis = self.get_portfolio_analysis()
        current_sectors = analysis.get("sector_allocation", {})
        
        lines = []
        for sector, target in targets.items():
            current = current_sectors.get(sector, 0.0)
            diff = current - target
            
            if abs(diff) < 2:  # Within tolerance
                status = "✅"
                action = "OK"
            elif diff > 0:  # Overweight
                status = "🔴"
                action = f"TRIM {abs(diff):.0f}%"
            else:  # Underweight
                status = "🟢"
                action = f"BUY +{abs(diff):.0f}%"
            
            lines.append(f"  {status} {sector}: {current:.0f}% → Target: {target:.0f}% ({action})")
        
        header = f"[{regime}] Dynamic Sector Targets:"
        return header + "\n" + "\n".join(lines)
    
    def _get_ai_suggestion(self, analysis: Dict) -> Optional[str]:
        """
        Get AI-generated ACTIONABLE rebalancing strategy.
        Returns specific trade recommendations.
        """
        try:
            # 1. Portfolio summary with RSI and cost info
            def rsi_label(rsi):
                if rsi is None: return ""
                if rsi > 70: return f", RSI: {rsi:.0f} ⚠️OVERBOUGHT"
                if rsi < 30: return f", RSI: {rsi:.0f} 🔥OVERSOLD"
                return f", RSI: {rsi:.0f}"
            
            def cost_label(a):
                if a['pnl_eur'] > 10:
                    # Realistic net profit for 10% of position
                    # (Profit of 10% portion) - (Tax on that 10% portion) - (Fee 1€)
                    net_prof_10 = (a['pnl_eur'] * 0.1) - (a['potential_tax'] * 0.1) - 1.0
                    if net_prof_10 > 0:
                        return f" [Profitto netto stimato per ogni 10% venduto: €{net_prof_10:.1f}]"
                return " [Se vendi: Fee €1 + eventuale tax 26% su profitto]"
            
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
                macro = eco.get_dashboard_stats()
                whale_context_str = whale.analyze_flow()
                market_status = eco.get_market_status()
                
                market_context = f"VIX: {macro.get('vix', 'N/A')}, Macro Risk: {macro.get('risk_level', 'N/A')}\n\n{whale_context_str}"
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
                        "quantity": asset.get('quantity', asset.get('qty', 0)),
                        "avg_price_eur": asset.get('avg_price_eur', asset.get('avg_price', 0)),
                        "current_price": asset.get('price', asset.get('current_price', asset.get('avg_price', 0)))
                    })
                
                logger.info(f"Rebalance: Running backtest for {len(bt_portfolio)} assets...")
                
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
                    logger.info(f"Rebalance: Backtest context added (Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.1f}%)")
                else:
                    logger.warning(f"Rebalance: Backtest returned error: {bt_results.get('error')}")
                
                # Add Correlation Analysis
                tickers = [asset['ticker'] for asset in analysis["assets"][:10]]
                if len(tickers) >= 2:
                    corr_data = self.market.calculate_correlation_matrix(tickers)
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
            
            **ORARI MERCATO ({market_status.get('current_time_italy', 'N/A')}):**
            {market_hours}
            
            **PORTAFOGLIO ATTUALE (ASSET POSSEDUTI):**
            €{analysis['total_value']:.0f} totali.
            ASSET: {', '.join([a['ticker'] for a in analysis['assets']]) if analysis['assets'] else 'NESSUNO (Liquidità)'}
            
            **ALLOCAZIONE SETTORI:**
            {sector_summary}
            
            **SEGNALI RECENTI (ultimi 7gg) - PRIORITÀ ALTA:**
            {signals_text}
            
            **CONTESTO MERCATO:**
            {market_context}
            
            {l2_context}
            
            {l3_context}
            
            **🔬 MACRO REGIME ANALYSIS (Level 9 - Active AI Manager):**
            - Regime: {regime_desc}
            - Risk Level: {risk_level}
            - Confidence: {regime_data.get('confidence', 0.5):.0%}
            - Signals: {', '.join(regime_data.get('signals', ['N/A'])[:3])}
            
            **📊 DYNAMIC SECTOR TARGETS (BASED ON CURRENT REGIME):**
            {self._format_regime_targets(regime_data)}
            
            **⚠️ YOUR JOB: Move the portfolio TOWARDS these targets!**
            - If a sector is ABOVE target → suggest TRIM
            - If a sector is BELOW target → suggest BUY/ACCUMULATE
            
            **🛡️ DYNAMIC STRATEGY RULES (MANDATORY):**
            {strategy_context}
            
            {backtest_context}
            
            **OBIETTIVO PRINCIPALE: MASSIMIZZARE I PROFITTI NETTI**
            
            **ISTRUZIONI:**
            1. Genera MAX 3 azioni concrete nel formato:
               🟢 BUY €XXX TICKER (NUOVO) - Motivo breve ← SOLO per asset NON in portafoglio
               🟢 ACCUMULATE €XXX TICKER - Motivo breve ← SOLO per asset GIÀ in portafoglio
               🟡 HOLD TICKER - Motivo breve ← VIETATO per asset NON in portafoglio
               🔴 TRIM XX% TICKER - Motivo breve (Profitto netto stimato: €[FAI_IL_CALCOLO])
            
            2. Poi aggiungi UNA strategia sintetica (1 frase) per €500 di capitale fresco.
            
            **REGOLE CRITICHE PER MASSIMIZZARE PROFITTI:**
            - **COST CHECK (FONDAMENTALE):** NON suggerire SELL/TRIM se il costo totale (26% Tax + €1 Fee) supera il 50% del profitto.
            - **PRIORITÀ 1:** Segui i SEGNALI RECENTI ad alta confidenza (BUY/ACCUMULATE ≥75%)
            - **PRIORITÀ 2:** Accumula asset con RSI OVERSOLD (<30)
            - **PRIORITÀ 3:** Trimma asset con RSI OVERBOUGHT (>70) SOLO SE il PnL netto giustifica la tassa.
            
            - **⏰ ORARI MERCATO (FONDAMENTALE):**
              - Se mercato EU è CLOSED: NON suggerire BUY/SELL per ETF europei → HOLD o "domani"
              - Se mercato US è CLOSED: NON suggerire BUY/SELL per stocks USA → HOLD o "riapertura"
              - Crypto sempre aperti.
            
            - **COERENZA PORTAFOGLIO (STRETTA):**
              - NON suggerire MAI "HOLD" o "TRIM" per un ticker che NON è nella lista "ASSET POSSEDUTI".
              - Se vuoi suggerire un asset nuovo, DEVE essere "BUY" (se conviene). Se non conviene comprare, IGNORALO (non dire HOLD).
            
            - **CALCOLO MATEMATICO (OBBLIGATORIO):**
              Per i suggerimenti '🔴 TRIM', usa il valore 'Profitto netto stimato per ogni 10% venduto' fornito nel contesto per calcolare il valore finale. 
              Esempio: Se suggerisci TRIM 20% e il profitto per ogni 10% è €30, scrivi '(Profitto netto stimato: €60)'. NON scrivere mai '€XX'.
            
            Rispondi SOLO con le azioni, senza preamboli.
            """
            
            # Use Brain's fallback system (OpenRouter auto-selects best free model)
            from brain import Brain
            brain = Brain()
            # OpenRouter will auto-select best reasoning model (DeepSeek R1 if available)
            logger.info("Calling Brain for AI strategy...")
            response_text = brain._generate_with_fallback(prompt, json_mode=False, task_type="rebalance")
            
            # Save brain reference for AI footer
            self._last_brain = brain
            
            # Check if response is valid - if empty, try Gemini fallback
            if not response_text or not response_text.strip():
                logger.warning(f"Primary AI returned empty, trying Gemini fallback...")
                try:
                    response_text = brain._call_gemini_with_tiered_fallback(prompt, json_mode=False)
                    if response_text and response_text.strip():
                        logger.info(f"Gemini fallback succeeded: {len(response_text)} chars")
                        
                        # Update tracking for Gemini fallback
                        brain.last_run_details = {
                            "model": "gemini-2.5-flash (Fallback)",
                            "usage": {"total_tokens": "N/A"},
                            "provider": "Google Direct (Fallback)"
                        }
                        
                        # Log fallback usage to DB
                        try:
                            from db_handler import DBHandler
                            db = DBHandler()
                            db.increment_api_counter("gemini_fallback")
                            db.log_model_used("gemini-2.5-flash-fallback")
                        except: pass
                    else:
                        logger.warning("Gemini fallback also returned empty")
                        return None
                except Exception as gemini_err:
                    logger.error(f"Gemini fallback failed: {gemini_err}")
                    return None
            
            logger.info(f"AI response received: {len(response_text)} chars")
            
            # --- CRITIC AGENT VALIDATION (New in Phase A.2) ---
            # Enforce consistency between Regime and Strategy
            try:
                from critic import Critic
                critic = Critic()
                regime_desc = regime_data.get('description', 'NEUTRAL')
                portfolio_val = analysis.get('total_value', 0)
                held_assets = [a['ticker'] for a in analysis['assets']]
                
                # Overwrite response with Critic's verified version
                valid_response = critic.critique_rebalance_strategy(response_text, regime_desc, portfolio_val, held_assets)
                
                if valid_response and valid_response != response_text:
                    response_text = valid_response
                    logger.info("Strategy refined by Critic Agent.")
                
                # [PHASE C.2] COUNCIL CONSENSUS (Adversarial Debate for Rebalance)
                import asyncio
                try:
                    portfolio_summary = f"Value: €{portfolio_val:.0f}\nAssets: {', '.join(held_assets)}"
                    
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()

                    response_text = loop.run_until_complete(
                        brain.council.get_strategy_consensus(portfolio_summary, response_text)
                    )
                except Exception as council_err:
                    logger.warning(f"Council strategy consensus failed: {council_err}")
                
            except Exception as critic_err:
                logger.error(f"Critic validation failed: {critic_err}")
                # Continue with original response_text if Critic fails
            
            # [V12] Save AI suggestions for learning
            self._save_suggestions_to_db(response_text, analysis, regime_data)
            
            return response_text.strip()

            
        except Exception as e:
            logger.error(f"AI suggestion failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _save_suggestions_to_db(self, response_text: str, analysis: dict, regime_data: dict):
        """
        Parse AI response and save individual suggestions to rebalancer_history for learning.
        """
        try:
            import re
            
            # Patterns to match AI suggestions
            patterns = [
                r'🟢\s*(BUY|ACCUMULATE)\s*€?(\d+)\s+(\w+[-\w]*)',
                r'🔴\s*(TRIM|SELL)\s*(\d+)%?\s*(\w+[-\w]*)',
                r'🟡\s*(HOLD)\s+(\w+[-\w]*)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 3:
                        action = match[0].upper()
                        amount = float(match[1]) if match[1].isdigit() else None
                        ticker = match[2].upper() if len(match) > 2 else match[1].upper()
                    elif len(match) == 2:
                        action = match[0].upper()
                        ticker = match[1].upper()
                        amount = None
                    else:
                        continue
                    
                    # Find asset in portfolio for context
                    asset_info = next((a for a in analysis.get('assets', []) 
                                      if a.get('ticker', '').upper() == ticker), {})
                    
                    self.db.save_rebalancer_suggestion(
                        ticker=ticker,
                        action=action,
                        amount=amount,
                        regime=regime_data.get('regime', 'UNKNOWN'),
                        sector_rotation=regime_data.get('sector_rotation', 'UNKNOWN'),
                        ticker_rsi=asset_info.get('rsi'),
                        ticker_pnl_pct=asset_info.get('pnl_pct'),
                        portfolio_value=analysis.get('total_value'),
                        price_at_suggestion=asset_info.get('current_price')
                    )
        except Exception as e:
            logger.warning(f"Failed to save suggestions for learning: {e}")
    
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
            logger.info(f"AI Strategy received ({len(ai_strategy)} chars)")
            # Sanitize markdown: escape ALL problematic Telegram Markdown characters
            # Telegram MarkdownV1 treats these as special: _ * ` [
            ai_strategy_safe = ai_strategy
            # Remove or escape underscores (most common issue)
            ai_strategy_safe = ai_strategy_safe.replace("_", " ")
            # Escape backticks (can break code blocks)
            ai_strategy_safe = ai_strategy_safe.replace("`", "'")
            # Remove square brackets (link syntax)
            ai_strategy_safe = ai_strategy_safe.replace("[", "(").replace("]", ")")
            
            report += "🎯 **PIANO D'AZIONE (Hybrid AI Strategy):**\n"
            report += ai_strategy_safe + "\n\n"
            fallback_used = False
        else:
            # AI failed - generate fallback recommendations from rules
            logger.warning("AI Strategy returned None - using rule-based fallback")
            suggestions = self.generate_rebalance_suggestions(analysis)
            if suggestions and len(suggestions) > 0:
                report += "🎯 **PIANO D'AZIONE (Fallback):**\n"
                for s in suggestions[:3]:
                    report += f"{s}\n"
                report += "\n"
                fallback_used = True
            else:
                report += "🎯 **PIANO D'AZIONE:** Portfolio bilanciato, nessuna azione richiesta.\n\n"
                fallback_used = True
        
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
        if not fallback_used:
            suggestions = self.generate_rebalance_suggestions(analysis)
            if suggestions and suggestions[0] != "✅ Portfolio is well-balanced. No rebalancing needed.":
                report += "\n💡 **Note Aggiuntive:**\n"
                for s in suggestions[:3]:  # Max 3
                    report += f"{s}\n"
        
        report += "\n" + "━" * 28
        report += f"\n_Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        
        # Add AI Model Footer
        try:
            if hasattr(self, '_last_brain') and self._last_brain:
                details = self._last_brain.last_run_details
                if details:
                    model_name = details.get('model', 'Unknown').split('/')[-1].replace(':free', '')
                    usage = details.get('usage', {})
                    total_tok = usage.get('total_tokens', 'N/A') if isinstance(usage, dict) else str(usage)
                    report += f"\n\n🤖 AI: `{model_name}` | 🎟️ `{total_tok}`"
        except: pass
        
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
                            return Brain()._generate_with_fallback(prompt, json_mode=False, task_type="flash_rebalance").strip()
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
