import logging
import datetime
from typing import List, Dict, Tuple, Optional
from db_handler import DBHandler

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """
    Analyzes past AI suggestions and determines their success rate.
    Provides context for 'Lessons Learned' to improve future AI prompts.
    """
    
    def __init__(self, db: Optional[DBHandler] = None):
        self.db = db or DBHandler()
        
    def log_suggestion(self, ticker: str, action: str, price: float, context: Optional[Dict] = None):
        """Record a new AI suggestion for later evaluation."""
        try:
            self.db.supabase.table("ai_performance_logs").insert({
                "ticker": ticker.upper(),
                "action_type": action.upper(),
                "price_at_suggestion": price,
                "suggestion_data": context or {}
            }).execute()
        except Exception as e:
            logger.debug(f"Failed to log AI suggestion: {e}")

    def evaluate_performance(self, horizon_days: int = 7):
        """
        Scan unevaluated suggestions older than horizon_days and mark as Win/Loss.
        """
        try:
            # 1. Fetch suggestions pending evaluation
            since = (datetime.datetime.now() - datetime.timedelta(days=horizon_days)).isoformat()
            resp = self.db.supabase.table("ai_performance_logs") \
                .select("*") \
                .is_("evaluated_at", "null") \
                .lt("created_at", since) \
                .execute()
            
            pending = resp.data or []
            if not pending:
                return 0
                
            from market_data import MarketData
            market = MarketData()
            
            evaluated_count = 0
            for item in pending:
                ticker = item['ticker']
                price_then = float(item['price_at_suggestion'])
                action = item['action_type']
                
                # Fetch current price
                current_price = market.get_current_price(ticker)
                if current_price <= 0: continue
                
                perf = (current_price - price_then) / price_then
                
                # Logic:
                # BUY/ACCUMULATE -> Win if perf > 0.01 (1%)
                # TRIM/SELL -> Win if perf < -0.01 (-1%)
                is_win = False
                if action in ['BUY', 'ACCUMULATE']:
                    is_win = perf > 0.01
                elif action in ['TRIM', 'SELL']:
                    is_win = perf < -0.01
                
                # Update DB
                self.db.supabase.table("ai_performance_logs").update({
                    "current_price": current_price,
                    "performance_pct": perf,
                    "is_win": is_win,
                    "evaluated_at": datetime.datetime.now().isoformat()
                }).eq("id", item['id']).execute()
                
                evaluated_count += 1
                
            return evaluated_count
        except Exception as e:
            logger.error(f"Feedback evaluation failed: {e}")
            return 0

    def get_lessons_learned(self, limit: int = 10) -> str:
        """Generate a text summary of recent performance for the AI prompt."""
        try:
            resp = getattr(self.db.supabase.table("ai_performance_logs").select("action_type, is_win"), "not") \
                .is_("evaluated_at", "null") \
                .order("evaluated_at", desc=True) \
                .limit(50) \
                .execute()
            
            data = resp.data or []
            if not data:
                return "Nessuna lezione appresa disponibile. Continua con la strategia standard."
                
            wins = sum(1 for i in data if i['is_win'])
            total = len(data)
            rate = (wins/total) if total > 0 else 0
            
            summary = f"Recent AI Accuracy: {rate:.0%} ({wins}/{total} successi).\n"
            
            # Action specific stats
            buy_wins = sum(1 for i in data if i['action_type'] in ['BUY', 'ACCUMULATE'] and i['is_win'])
            buy_total = sum(1 for i in data if i['action_type'] in ['BUY', 'ACCUMULATE'])
            if buy_total > 5:
                buy_rate = buy_wins / buy_total
                if buy_rate < 0.4:
                    summary += "⚠️ ATTENZIONE: I suggerimenti BUY hanno avuto scarso successo recentemente. Sii più selettivo.\n"
                elif buy_rate > 0.7:
                    summary += "🚀 I suggerimenti BUY sono molto affidabili ora.\n"
                    
            return summary
        except Exception as e:
            return "Errore nel recupero delle lezioni apprese."
