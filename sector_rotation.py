"""
Sector Rotation Tracker - L2 Predictive System
===============================================
Tracks relative performance of sectors to detect capital flows.
"""

import logging
import yfinance as yf
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger("SectorRotation")


class SectorRotationTracker:
    """
    Tracks relative performance of sectors to detect rotation patterns.
    
    Key sectors:
    - Technology (QQQ)
    - Energy (XLE)
    - Financials (XLF)
    - Healthcare (XLV)
    - Utilities (XLU) - Defensive
    - Crypto (BTC)
    
    Detects rotation signals: Risk-On vs Risk-Off
    """
    
    # Sector ETF mapping
    SECTORS = {
        "Technology": "QQQ",
        "S&P 500": "SPY",
        "Energy": "XLE",
        "Financials": "XLF",
        "Healthcare": "XLV",
        "Utilities": "XLU",  # Defensive
        "Crypto": "BTC-USD",
        "Small Caps": "IWM"
    }
    
    # Sector classification
    RISK_ON_SECTORS = ["Technology", "Crypto", "Small Caps"]
    RISK_OFF_SECTORS = ["Utilities", "Healthcare"]
    
    def __init__(self):
        self._cache = {}
        self._cache_time = None
        self.CACHE_TTL_MINUTES = 60
        logger.info("SectorRotationTracker initialized")
    
    def _get_sector_performance(self, ticker: str, period: str = "7d") -> Dict:
        """Get performance for a sector ETF."""
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            
            if data.empty or len(data) < 2:
                return {"error": f"No data for {ticker}"}
            
            # Handle MultiIndex
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                close = data['Close']
            
            start_price = float(close.iloc[0])
            end_price = float(close.iloc[-1])
            change_pct = ((end_price - start_price) / start_price) * 100
            
            return {
                "ticker": ticker,
                "start": start_price,
                "end": end_price,
                "change_pct": round(change_pct, 2)
            }
        except Exception as e:
            logger.warning(f"Failed to get performance for {ticker}: {e}")
            return {"error": str(e)}
    
    def analyze(self, period: str = "7d") -> Dict:
        """
        Analyze sector rotation for the given period.
        
        Returns:
            {
                "leading": ["Technology", "Crypto"],
                "lagging": ["Utilities", "Energy"],
                "rotation_signal": "RISK_ON" | "RISK_OFF" | "NEUTRAL",
                "confidence": 0.0-1.0,
                "sector_performance": {...}
            }
        """
        # Check cache
        if self._cache and self._cache_time:
            age_minutes = (datetime.now() - self._cache_time).seconds / 60
            if age_minutes < self.CACHE_TTL_MINUTES:
                return self._cache
        
        performances = {}
        
        # Get performance for each sector
        for sector_name, ticker in self.SECTORS.items():
            perf = self._get_sector_performance(ticker, period)
            if "error" not in perf:
                performances[sector_name] = perf["change_pct"]
        
        if not performances:
            return {"error": "Failed to fetch sector data"}
        
        # Sort by performance
        sorted_sectors = sorted(performances.items(), key=lambda x: x[1], reverse=True)
        
        # Identify leading and lagging
        leading = [s[0] for s in sorted_sectors[:3]]
        lagging = [s[0] for s in sorted_sectors[-3:]]
        
        # Calculate risk-on vs risk-off score
        risk_on_perf = sum([performances.get(s, 0) for s in self.RISK_ON_SECTORS if s in performances])
        risk_off_perf = sum([performances.get(s, 0) for s in self.RISK_OFF_SECTORS if s in performances])
        
        # Determine rotation signal
        diff = risk_on_perf - risk_off_perf
        if diff > 3:
            rotation_signal = "RISK_ON"
            confidence = min(0.95, 0.6 + abs(diff) / 20)
        elif diff < -3:
            rotation_signal = "RISK_OFF"
            confidence = min(0.95, 0.6 + abs(diff) / 20)
        else:
            rotation_signal = "NEUTRAL"
            confidence = 0.5
        
        # Create ranking list for dashboard
        ranking_list = [
            {"sector": s[0], "momentum_score": s[1]} 
            for s in sorted_sectors
        ]

        result = {
            "leading": leading,
            "lagging": lagging,
            "rotation_signal": rotation_signal,
            "confidence": round(confidence, 2),
            "sector_performance": performances,
            "ranking": ranking_list,
            "risk_on_score": round(risk_on_perf, 2),
            "risk_off_score": round(risk_off_perf, 2),
            "period": period,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        self._cache = result
        self._cache_time = datetime.now()
        
        logger.info(f"Sector Rotation: {rotation_signal} ({confidence:.0%}). "
                   f"Leading: {', '.join(leading)}. Lagging: {', '.join(lagging)}")
        
        return result
    
    def get_recommendation_for_sector(self, sector: str) -> str:
        """Get recommendation for a specific sector based on rotation."""
        analysis = self.analyze()
        
        if "error" in analysis:
            return "HOLD"
        
        if sector in analysis["leading"]:
            return "OVERWEIGHT"
        elif sector in analysis["lagging"]:
            return "UNDERWEIGHT"
        else:
            return "NEUTRAL"
    
    def format_for_ai(self) -> str:
        """Format sector rotation for AI context."""
        analysis = self.analyze()
        
        if "error" in analysis:
            return "Sector Rotation: Not available"
        
        perf_str = "\n".join([f"  - {k}: {v:+.1f}%" for k, v in 
                             sorted(analysis['sector_performance'].items(), 
                                   key=lambda x: x[1], reverse=True)])
        
        return (
            f"Sector Rotation Analysis ({analysis['period']}):\n"
            f"- Signal: {analysis['rotation_signal']} ({analysis['confidence']:.0%})\n"
            f"- Leading: {', '.join(analysis['leading'])}\n"
            f"- Lagging: {', '.join(analysis['lagging'])}\n"
            f"- Performance:\n{perf_str}"
        )


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tracker = SectorRotationTracker()
    
    print("\n=== 7-Day Sector Rotation ===")
    result = tracker.analyze("7d")
    print(f"Signal: {result['rotation_signal']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Leading: {result['leading']}")
    print(f"Lagging: {result['lagging']}")
    print(f"\nPerformance:")
    for sector, perf in sorted(result['sector_performance'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {sector}: {perf:+.2f}%")
