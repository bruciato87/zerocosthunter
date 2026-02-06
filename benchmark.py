"""
Benchmark Module - The Hedge Fund Manager (Phase 17)
=====================================================
Compares portfolio performance against market benchmarks (S&P500, BTC).
"""

import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger("Benchmark")


class Benchmark:
    """Portfolio performance comparison against market benchmarks."""
    
    # Benchmark tickers
    BENCHMARKS = {
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    }
    
    def __init__(self):
        from db_handler import DBHandler
        from market_data import MarketData
        self.db = DBHandler()
        self.market = MarketData()
    
    async def get_benchmark_performance_async(self, period_days: int = 30):
        """Async version of benchmark performance fetch."""
        import asyncio
        return await asyncio.to_thread(self.get_benchmark_performance, period_days)

    def get_benchmark_performance(self, period_days: int = 30) -> Dict[str, Dict]:
        """
        Get performance of all benchmarks over a period.
        
        Returns:
            {
                "S&P500": {"start_price": 4500, "end_price": 4600, "return_pct": 2.22},
                ...
            }
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        results = {}
        
        for name, ticker in self.BENCHMARKS.items():
            try:
                data = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=False
                )
                
                if len(data) < 2:
                    continue
                
                # Handle MultiIndex columns
                if hasattr(data.columns, 'levels'):
                    close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
                else:
                    close = data['Close']
                
                start_price = float(close.iloc[0])
                end_price = float(close.iloc[-1])
                return_pct = ((end_price - start_price) / start_price) * 100
                
                results[name] = {
                    "start_price": start_price,
                    "end_price": end_price,
                    "return_pct": return_pct,
                    "period_days": period_days
                }
                
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
        
        return results
    
    def get_portfolio_performance(self, period_days: int = 30) -> Dict:
        """
        Calculate portfolio performance over a period.
        Uses trade history to estimate starting value.
        
        Returns:
            {
                "start_value": 1000,
                "current_value": 1150,
                "return_pct": 15.0,
                "period_days": 30
            }
        """
        try:
            portfolio = self.db.get_portfolio()
            if not portfolio:
                return {"error": "No portfolio data"}
            
            # Calculate current value
            current_value = 0.0
            cost_basis = 0.0
            
            for p in portfolio:
                ticker = p.get('ticker', '')
                qty = p.get('quantity', 0)
                avg_price = p.get('avg_price', 0)
                
                # Get current price in EUR
                current_price, _ = self.market.get_smart_price_eur(ticker)
                
                if current_price and current_price > 0:
                    current_value += qty * current_price
                    cost_basis += qty * avg_price
            
            if cost_basis <= 0:
                return {"error": "No cost basis data"}
            
            # Overall return since inception
            total_return_pct = ((current_value - cost_basis) / cost_basis) * 100
            
            # For period-specific return, we'd need historical snapshots
            # For now, use total return as approximation
            return {
                "cost_basis": cost_basis,
                "current_value": current_value,
                "return_pct": total_return_pct,
                "period_days": period_days,
                "note": "Total return since inception"
            }
            
        except Exception as e:
            logger.error(f"Portfolio performance calc failed: {e}")
            return {"error": str(e)}
    
    def compare_vs_benchmarks(self, period_days: int = 30) -> Dict:
        """
        Compare portfolio against all benchmarks.
        
        Returns:
            {
                "portfolio": {...},
                "benchmarks": {...},
                "beating": ["S&P500", "NASDAQ"],  # List of benchmarks portfolio beats
                "losing_to": ["Bitcoin"]
            }
        """
        portfolio_perf = self.get_portfolio_performance(period_days)
        benchmark_perf = self.get_benchmark_performance(period_days)
        
        if "error" in portfolio_perf:
            return {"error": portfolio_perf["error"]}
        
        portfolio_return = portfolio_perf.get("return_pct", 0)
        
        beating = []
        losing_to = []
        
        for name, data in benchmark_perf.items():
            bench_return = data.get("return_pct", 0)
            if portfolio_return > bench_return:
                beating.append(name)
            else:
                losing_to.append(name)
        
        return {
            "portfolio": portfolio_perf,
            "benchmarks": benchmark_perf,
            "beating": beating,
            "losing_to": losing_to,
            "period_days": period_days
        }
    
    def get_top_movers(self, limit: int = 5) -> Dict[str, List]:
        """
        Get top gainers and losers from portfolio.
        
        Returns:
            {
                "gainers": [{"ticker": "BTC", "pnl_pct": 25.5}, ...],
                "losers": [{"ticker": "AAPL", "pnl_pct": -5.2}, ...]
            }
        """
        try:
            portfolio = self.db.get_portfolio()
            if not portfolio:
                return {"gainers": [], "losers": []}
            
            movers = []
            
            for p in portfolio:
                ticker = p.get('ticker', '')
                qty = p.get('quantity', 0)
                avg_price = p.get('avg_price', 0)
                
                if avg_price <= 0:
                    continue
                
                current_price, _ = self.market.get_smart_price_eur(ticker)
                
                if current_price and current_price > 0:
                    pnl_pct = ((current_price - avg_price) / avg_price) * 100
                    value = qty * current_price
                    
                    movers.append({
                        "ticker": ticker,
                        "pnl_pct": pnl_pct,
                        "value": value,
                        "quantity": qty
                    })
            
            # Sort by PnL
            movers.sort(key=lambda x: x["pnl_pct"], reverse=True)
            
            gainers = [m for m in movers if m["pnl_pct"] > 0][:limit]
            losers = [m for m in movers if m["pnl_pct"] < 0][-limit:][::-1]  # Worst first
            
            return {
                "gainers": gainers,
                "losers": losers
            }
            
        except Exception as e:
            logger.error(f"Top movers calc failed: {e}")
            return {"gainers": [], "losers": []}
    
    async def format_benchmark_report_async(self, period_days: int = 30) -> str:
        """Async version of report generation with parallel fetching."""
        import asyncio
        
        # 1. Fetch Benchmarks in parallel
        # 2. Fetch Portfolio Performance (which uses market data cache)
        # 3. Combine
        
        bench_task = self.get_benchmark_performance_async(period_days)
        # Note: get_portfolio_performance is currently sync but market.get_smart_price_eur is cached
        # Let's run it in thread to be safe
        port_task = asyncio.to_thread(self.get_portfolio_performance, period_days)
        
        benchmarks, portfolio = await asyncio.gather(bench_task, port_task)
        
        if "error" in portfolio:
            return f"⚠️ Errore: {portfolio['error']}"
        
        # Header
        report = f"📊 **Performance Report** (Last {period_days} days)\n\n"
        
        # Portfolio Performance
        port_return = portfolio.get("return_pct", 0)
        port_value = portfolio.get("current_value", 0)
        port_emoji = "🟢" if port_return >= 0 else "🔴"
        
        report += f"{port_emoji} **Il Tuo Portfolio:** {port_return:+.2f}%\n"
        report += f"💰 Valore: €{port_value:,.2f}\n\n"
        
        # Benchmarks
        report += "📈 **Benchmarks:**\n"
        for name, data in benchmarks.items():
            bench_return = data.get("return_pct", 0)
            emoji = "🟢" if bench_return >= 0 else "🔴"
            vs = "✅" if port_return > bench_return else "❌"
            report += f"{emoji} {name}: {bench_return:+.2f}% {vs}\n"
            
        # Summary
        beating = []
        losing_to = []
        for name, data in benchmarks.items():
            if port_return > data.get("return_pct", 0):
                beating.append(name)
            else:
                losing_to.append(name)
        
        report += "\n"
        if beating:
            report += f"🏆 **Batti:** {', '.join(beating)}\n"
        if losing_to:
            report += f"📉 **Perdi vs:** {', '.join(losing_to)}\n"
            
        # Top Movers (uses cache)
        movers = self.get_top_movers(3)
        if movers["gainers"]:
            report += "\n🚀 **Top Gainers:**\n"
            for g in movers["gainers"]:
                report += f"  • {g['ticker']}: {g['pnl_pct']:+.1f}%\n"
        if movers["losers"]:
            report += "\n📉 **Top Losers:**\n"
            for l in movers["losers"]:
                report += f"  • {l['ticker']}: {l['pnl_pct']:+.1f}%\n"
        
        return report

    def format_benchmark_report(self, period_days: int = 30) -> str:
        """Generate formatted benchmark comparison report for Telegram."""
        comparison = self.compare_vs_benchmarks(period_days)
        
        if "error" in comparison:
            return f"⚠️ Errore: {comparison['error']}"
        
        portfolio = comparison["portfolio"]
        benchmarks = comparison["benchmarks"]
        
        # Header
        report = f"📊 **Performance Report** (Last {period_days} days)\n\n"
        
        # Portfolio Performance
        port_return = portfolio.get("return_pct", 0)
        port_value = portfolio.get("current_value", 0)
        port_emoji = "🟢" if port_return >= 0 else "🔴"
        
        report += f"{port_emoji} **Il Tuo Portfolio:** {port_return:+.2f}%\n"
        report += f"💰 Valore: €{port_value:,.2f}\n\n"
        
        # Benchmarks
        report += "📈 **Benchmarks:**\n"
        for name, data in benchmarks.items():
            bench_return = data.get("return_pct", 0)
            emoji = "🟢" if bench_return >= 0 else "🔴"
            
            # Check if beating
            vs = "✅" if port_return > bench_return else "❌"
            report += f"{emoji} {name}: {bench_return:+.2f}% {vs}\n"
        
        # Summary
        beating = comparison.get("beating", [])
        losing_to = comparison.get("losing_to", [])
        
        report += "\n"
        if beating:
            report += f"🏆 **Batti:** {', '.join(beating)}\n"
        if losing_to:
            report += f"📉 **Perdi vs:** {', '.join(losing_to)}\n"
        
        # Top Movers
        movers = self.get_top_movers(3)
        
        if movers["gainers"]:
            report += "\n🚀 **Top Gainers:**\n"
            for g in movers["gainers"][:3]:
                report += f"  • {g['ticker']}: {g['pnl_pct']:+.1f}%\n"
        
        if movers["losers"]:
            report += "\n📉 **Top Losers:**\n"
            for l in movers["losers"][:3]:
                report += f"  • {l['ticker']}: {l['pnl_pct']:+.1f}%\n"
        
        return report
    
    def generate_weekly_summary(self) -> str:
        """
        Generate comprehensive weekly portfolio summary.
        Includes all assets, sector breakdown, and AI outlook.
        """
        import os
        from datetime import datetime
        
        try:
            portfolio = self.db.get_portfolio()
            if not portfolio:
                return "📑 **Weekly Summary**\n\n❌ Portafoglio vuoto."
            
            # Calculate all asset values
            assets = []
            sector_values = {}
            total_value = 0.0
            total_cost = 0.0
            
            for p in portfolio:
                ticker = p.get('ticker', 'UNKNOWN')
                qty = float(p.get('quantity', 0))
                avg_price = float(p.get('avg_price', 0))
                
                current_price, _ = self.market.get_smart_price_eur(ticker)
                if current_price <= 0:
                    current_price = avg_price
                
                value = qty * current_price
                cost = qty * avg_price
                total_value += value
                total_cost += cost
                
                pnl_pct = ((value - cost) / cost * 100) if cost > 0 else 0
                
                # Get sector
                from advisor import Advisor
                adv = Advisor()
                sector = adv.get_sector(ticker)
                sector_values[sector] = sector_values.get(sector, 0.0) + value
                
                assets.append({
                    "ticker": ticker,
                    "value": value,
                    "pnl_pct": pnl_pct,
                    "allocation": 0,  # Calculate after total
                    "sector": sector
                })
            
            # Calculate allocations
            for asset in assets:
                asset["allocation"] = (asset["value"] / total_value * 100) if total_value > 0 else 0
            
            # Sort by value
            assets.sort(key=lambda x: -x["value"])
            
            # Build report
            total_pnl = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
            pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
            
            report = "📑 **WEEKLY PORTFOLIO SUMMARY**\n"
            report += "━" * 28 + "\n\n"
            
            # Overview
            report += f"💰 **Valore Totale:** €{total_value:,.0f}\n"
            report += f"{pnl_emoji} **P/L Totale:** {total_pnl:+.2f}%\n\n"
            
            # All Assets
            report += "📦 **Tutti gli Asset:**\n"
            for asset in assets:
                emoji = "🟢" if asset["pnl_pct"] >= 0 else "🔴"
                report += f"  {emoji} **{asset['ticker']}**: €{asset['value']:.0f} ({asset['allocation']:.1f}%) | {asset['pnl_pct']:+.1f}%\n"
            
            # Sector Breakdown
            report += "\n📊 **Settori:**\n"
            sector_pcts = {s: (v / total_value * 100) if total_value > 0 else 0 for s, v in sector_values.items()}
            for sector, pct in sorted(sector_pcts.items(), key=lambda x: -x[1]):
                report += f"  • {sector}: {pct:.1f}%\n"
            
            # Benchmarks comparison (7 days)
            report += "\n📈 **vs Benchmarks (7d):**\n"
            benchmarks = self.get_benchmark_performance(7)
            for name, data in benchmarks.items():
                ret = data.get("return_pct", 0)
                vs = "✅" if total_pnl > ret else "❌"
                report += f"  {vs} {name}: {ret:+.2f}%\n"
            
            # AI Outlook
            ai_outlook = self._get_ai_weekly_outlook(assets, sector_pcts, total_pnl)
            if ai_outlook:
                report += f"\n🔮 **Outlook Settimanale:**\n{ai_outlook}\n"
            
            report += "\n" + "━" * 28
            report += f"\n_Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
            
            return report
            
        except Exception as e:
            logger.error(f"Weekly summary failed: {e}")
            return f"⚠️ Errore generazione summary: {e}"
    
    def _get_ai_weekly_outlook(self, assets: List[Dict], sectors: Dict, total_pnl: float) -> Optional[str]:
        """Generate AI weekly outlook."""
        import os
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        
        try:
            from google import genai
            
            client = genai.Client(api_key=api_key)
            
            # Prepare context
            top_assets = "\n".join([
                f"- {a['ticker']}: {a['pnl_pct']:+.1f}%, {a['allocation']:.0f}% allocation"
                for a in assets[:5]
            ])
            
            sector_str = "\n".join([f"- {s}: {p:.0f}%" for s, p in sectors.items()])
            
            prompt = f"""
            Sei un consulente finanziario. Scrivi un BREVE outlook settimanale (MAX 3 righe) per questo portfolio.
            
            Performance Totale: {total_pnl:+.1f}%
            
            Top Asset:
            {top_assets}
            
            Settori:
            {sector_str}
            
            Regole:
            - MAX 3 righe
            - Italiano
            - Sii specifico sui singoli asset
            - Menziona rischi e opportunità
            """
            
            # Use Brain's fallback system (respects APP_MODE: PREPROD/PROD)
            from brain import Brain
            brain = Brain()
            response_text = brain._generate_with_fallback(prompt, json_mode=False)
            
            return response_text.strip()
            
        except Exception as e:
            logger.warning(f"AI outlook failed: {e}")
            return None
    
    async def send_weekly_summary(self):
        """
        Send weekly summary to Telegram.
        Called by GitHub Actions on Sunday.
        """
        from telegram_bot import TelegramNotifier
        
        logger.info("Generating weekly summary...")
        
        try:
            report = self.generate_weekly_summary()
            
            notifier = TelegramNotifier()
            await notifier.send_alert(report)
            
            # Log to DB
            self.db.log_system_event("INFO", "Benchmark", "Weekly summary sent")
            
            logger.info("Weekly summary sent successfully")
            
        except Exception as e:
            logger.error(f"Weekly summary failed: {e}")
            self.db.log_system_event("ERROR", "Benchmark", f"Weekly summary failed: {e}")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    bench = Benchmark()
    
    print("="*50)
    print("Benchmark Performance (30 days)")
    print("="*50)
    
    benchmarks = bench.get_benchmark_performance(30)
    for name, data in benchmarks.items():
        print(f"  {name}: {data['return_pct']:+.2f}%")
    
    print("\n" + "="*50)
    print("Portfolio vs Benchmarks")
    print("="*50)
    
    report = bench.format_benchmark_report(30)
    print(report)

