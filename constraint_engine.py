import logging
import numpy as np
from typing import List, Dict, Tuple
from db_handler import DBHandler

logger = logging.getLogger(__name__)

class ConstraintEngine:
    """
    Enforces quantitative portfolio constraints and risk guardrails.
    Ensures that AI suggestions do not violate safety limits.
    """
    
    def __init__(self, db: DBHandler = None):
        self.db = db or DBHandler()
        self.MAX_TICKER_EXPOSURE = 0.15 # Max 15% per single ticker
        self.MAX_SECTOR_EXPOSURE = 0.40 # Max 40% per sector (redundant with rebalance targets, but acts as a hard limit)
        self.CORRELATION_THRESHOLD = 0.85 # Alert/Block if correl > 0.85

    def validate_action(
        self,
        action_type: str,
        ticker: str,
        amount_eur: float,
        portfolio: List[Dict],
        sector: str = None
    ) -> Tuple[bool, str]:
        """
        Validate if an action (BUY/ACCUMULATE/TRIM) violates any constraints.
        
        Args:
            action_type: 'BUY', 'ACCUMULATE', 'TRIM', 'SELL'
            ticker: Asset ticker
            amount_eur: Controvalore dell'operazione
            portfolio: Current holdings list
            
        Returns:
            Tuple (is_valid, reason)
        """
        if action_type not in ['BUY', 'ACCUMULATE']:
            return True, "" # Sales/Holds are generally fine for risk reduction
            
        def _position_value(asset: Dict) -> float:
            if asset is None:
                return 0.0
            if asset.get('value') is not None:
                return float(asset.get('value') or 0.0)
            qty = float(asset.get('quantity', 0) or 0)
            price = float(asset.get('current_price', asset.get('avg_price', 0)) or 0)
            if qty > 0 and price > 0:
                return qty * price
            # Last-resort fallback for sparse test fixtures.
            return float(asset.get('avg_price', 0) or 0)

        total_value = sum(_position_value(a) for a in portfolio)
        total_value += amount_eur # New estimated total
        
        # 1. Ticker Exposure Check
        current_ticker_val = next((_position_value(a) for a in portfolio if a.get('ticker') == ticker), 0.0)
        new_ticker_val = current_ticker_val + amount_eur
        
        exposure = new_ticker_val / total_value if total_value > 0 else 1.0
        
        if exposure > self.MAX_TICKER_EXPOSURE:
            return False, f"Esposizione su {ticker} supererebbe il limite del {self.MAX_TICKER_EXPOSURE:.0%} (attuale: {exposure:.1%})"
            
        # 2. Sector Exposure Check (only when sector context is available)
        if sector:
            current_sector_val = sum(
                _position_value(a)
                for a in portfolio
                if a.get('sector') == sector
            )
            new_sector_val = current_sector_val + amount_eur
            sector_exposure = new_sector_val / total_value if total_value > 0 else 1.0
            if sector_exposure > self.MAX_SECTOR_EXPOSURE:
                return False, (
                    f"Esposizione settore {sector} supererebbe il limite del "
                    f"{self.MAX_SECTOR_EXPOSURE:.0%} (attuale: {sector_exposure:.1%})"
                )
        
        return True, ""

    def get_risk_adjusted_size(self, ticker: str, base_amount: float, volatility_score: float) -> float:
        """
        Scale position size based on volatility. 
        Higher volatility (ATR%) -> Lower size.
        """
        # Simple Inverse Volatility Weighting
        # vol_score 0 to 10 (10 = hyper-volatile crypto)
        scaling_factor = max(0.5, 1.0 - (volatility_score / 20.0))
        return base_amount * scaling_factor
