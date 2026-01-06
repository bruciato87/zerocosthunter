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
    
    def __init__(self):
        from db_handler import DBHandler
        from market_data import MarketData
        from advisor import Advisor
        
        self.db = DBHandler()
        self.market = MarketData()
        self.advisor = Advisor()
        
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
            
            assets.append({
                "ticker": ticker,
                "quantity": qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "value": value,
                "pnl_pct": pnl_pct,
                "sector": sector
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
            
            # 1. Portfolio summary
            assets_summary = "\n".join([
                f"- {a['ticker']}: €{a['value']:.0f} ({a['allocation']:.1f}%), PnL: {a['pnl_pct']:+.1f}%, Sector: {a['sector']}"
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
                
                macro = eco.get_macro_summary()
                whale_data = whale.get_whale_activity()
                
                market_context = f"VIX: {macro.get('vix', 'N/A')}, Fed: {macro.get('fed_sentiment', 'N/A')}, Whale: {whale_data.get('strategy_hint', 'N/A')}"
            except Exception as e:
                logger.warning(f"Market context failed: {e}")
            
            prompt = f"""
            Sei un Portfolio Manager italiano. Genera un PIANO DI RIBILANCIAMENTO concreto.
            
            **PORTAFOGLIO ATTUALE:** €{analysis['total_value']:.0f}
            {assets_summary}
            
            **ALLOCAZIONE SETTORI:**
            {sector_summary}
            
            **SEGNALI RECENTI (ultimi 7gg):**
            {signals_text}
            
            **CONTESTO MERCATO:**
            {market_context}
            
            **ISTRUZIONI:**
            1. Genera MAX 3 azioni concrete nel formato:
               🟢 BUY €XXX TICKER - Motivo breve
               🟡 HOLD TICKER - Motivo breve
               🔴 TRIM XX% TICKER - Motivo breve
            
            2. Poi aggiungi UNA strategia sintetica (1 frase) per €500 di capitale fresco.
            
            **REGOLE:**
            - Considera tasse italiane 2026 (Crypto 33%)
            - NON suggerire SELL totale (solo TRIM parziale)
            - PRIORIZZA asset con segnali BUY recenti
            - Se ETF > 40%, suggerisci redistribuzione
            
            Rispondi SOLO con le azioni, senza preamboli.
            """
            
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            
            return response.text.strip()
            
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
        
        # AI Strategy (FIRST - most important)
        ai_strategy = self._get_ai_suggestion(analysis)
        if ai_strategy:
            report += "🎯 **PIANO D'AZIONE:**\n"
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
    
    async def run_daily(self):
        """
        Run daily rebalancing analysis and send to Telegram.
        Called by GitHub Actions.
        """
        from telegram_bot import TelegramNotifier
        
        logger.info("Running daily rebalancing analysis...")
        
        try:
            report = self.format_rebalance_report()
            
            notifier = TelegramNotifier()
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
