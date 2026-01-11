"""
Support/Resistance AI - L2 Predictive System
=============================================
Identifies key support and resistance levels and alerts when price approaches.
"""

import logging
import yfinance as yf
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger("SupportResistance")


class SupportResistanceAI:
    """
    Identifies key support/resistance levels using:
    - Recent swing highs/lows (local minima/maxima)
    - Volume profile (high volume = strong level)
    - Round numbers (psychological levels)
    
    Returns levels + distance from current price.
    """
    
    def __init__(self):
        self._cache = {}
        self.CACHE_TTL_MINUTES = 60
        logger.info("SupportResistanceAI initialized")
    
    def _find_swing_points(self, close: list, window: int = 5) -> tuple:
        """Find local minima and maxima in price data."""
        highs = []
        lows = []
        
        for i in range(window, len(close) - window):
            # Check if this is a local high
            if all(close[i] >= close[i-j] for j in range(1, window+1)) and \
               all(close[i] >= close[i+j] for j in range(1, window+1)):
                highs.append(float(close[i]))
            
            # Check if this is a local low
            if all(close[i] <= close[i-j] for j in range(1, window+1)) and \
               all(close[i] <= close[i+j] for j in range(1, window+1)):
                lows.append(float(close[i]))
        
        return highs, lows
    
    def _get_round_numbers(self, current_price: float) -> List[float]:
        """Get psychologically significant round numbers near current price."""
        round_numbers = []
        
        # Determine magnitude
        if current_price > 10000:
            step = 5000  # BTC levels
        elif current_price > 1000:
            step = 100
        elif current_price > 100:
            step = 10
        elif current_price > 10:
            step = 5
        else:
            step = 1
        
        # Find round numbers within ±30% of current price
        lower = current_price * 0.7
        upper = current_price * 1.3
        
        level = (lower // step) * step
        while level <= upper:
            if level > 0:
                round_numbers.append(level)
            level += step
        
        return round_numbers
    
    def _cluster_levels(self, levels: List[float], tolerance_pct: float = 2.0) -> List[float]:
        """Cluster nearby levels into single levels (average of cluster)."""
        if not levels:
            return []
        
        sorted_levels = sorted(set(levels))
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If level is within tolerance of cluster average, add to cluster
            cluster_avg = sum(current_cluster) / len(current_cluster)
            if abs(level - cluster_avg) / cluster_avg * 100 <= tolerance_pct:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        clusters.append(sum(current_cluster) / len(current_cluster))
        return [round(c, 2) for c in clusters]
    
    def get_levels(self, ticker: str, period: str = "6mo") -> Dict:
        """
        Get support and resistance levels for a ticker.
        
        Returns:
            {
                "support": [level1, level2, ...],
                "resistance": [level1, level2, ...],
                "current_price": float,
                "nearest_support": float,
                "nearest_resistance": float,
                "support_distance_pct": float,
                "resistance_distance_pct": float,
                "zone": "near_support" | "near_resistance" | "neutral"
            }
        """
        try:
            # Handle crypto tickers
            yf_ticker = ticker
            crypto_list = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'RENDER', 'DOGE']
            base = ticker.replace('-USD', '').replace('-EUR', '')
            if base in crypto_list and not yf_ticker.endswith('-USD'):
                yf_ticker = f"{base}-USD"
            
            data = yf.download(yf_ticker, period=period, progress=False, auto_adjust=True)
            
            if data.empty or len(data) < 20:
                return {"error": f"Insufficient data for {ticker}"}
            
            # Handle MultiIndex
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                close = data['Close']
            
            close_list = close.tolist()
            current_price = float(close.iloc[-1])
            
            # Find swing points
            highs, lows = self._find_swing_points(close_list)
            
            # Add round numbers
            round_nums = self._get_round_numbers(current_price)
            
            # Combine and cluster
            all_resistance = [h for h in highs if h > current_price] + \
                            [r for r in round_nums if r > current_price]
            all_support = [l for l in lows if l < current_price] + \
                         [r for r in round_nums if r < current_price]
            
            # Cluster levels
            support_levels = self._cluster_levels(all_support)
            resistance_levels = self._cluster_levels(all_resistance)
            
            # Sort: support descending (nearest first), resistance ascending (nearest first)
            support_levels = sorted(support_levels, reverse=True)[:5]
            resistance_levels = sorted(resistance_levels)[:5]
            
            # Calculate distances
            nearest_support = support_levels[0] if support_levels else current_price * 0.9
            nearest_resistance = resistance_levels[0] if resistance_levels else current_price * 1.1
            
            support_distance_pct = ((current_price - nearest_support) / current_price) * 100
            resistance_distance_pct = ((nearest_resistance - current_price) / current_price) * 100
            
            # Determine zone
            if support_distance_pct < 3:
                zone = "near_support"
            elif resistance_distance_pct < 3:
                zone = "near_resistance"
            else:
                zone = "neutral"
            
            result = {
                "ticker": ticker,
                "support": support_levels,
                "resistance": resistance_levels,
                "current_price": round(current_price, 2),
                "nearest_support": round(nearest_support, 2),
                "nearest_resistance": round(nearest_resistance, 2),
                "support_distance_pct": round(support_distance_pct, 2),
                "resistance_distance_pct": round(resistance_distance_pct, 2),
                "zone": zone,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"S/R for {ticker}: Support {nearest_support:.2f} ({support_distance_pct:.1f}% away), "
                       f"Resistance {nearest_resistance:.2f} ({resistance_distance_pct:.1f}% away)")
            
            return result
            
        except Exception as e:
            logger.warning(f"S/R analysis failed for {ticker}: {e}")
            return {"error": str(e)}
    
    def format_for_ai(self, ticker: str) -> str:
        """Format S/R levels for AI context."""
        levels = self.get_levels(ticker)
        
        if "error" in levels:
            return f"S/R Analysis: Not available for {ticker}"
        
        support_str = ", ".join([f"€{s:.2f}" for s in levels['support'][:3]])
        resistance_str = ", ".join([f"€{r:.2f}" for r in levels['resistance'][:3]])
        
        return (
            f"S/R Analysis:\n"
            f"- Current: €{levels['current_price']:.2f}\n"
            f"- Support: {support_str} (nearest {levels['support_distance_pct']:.1f}% below)\n"
            f"- Resistance: {resistance_str} (nearest {levels['resistance_distance_pct']:.1f}% above)\n"
            f"- Zone: {levels['zone']}"
        )


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sr = SupportResistanceAI()
    
    for ticker in ["BTC-USD", "NVDA", "ETH-USD"]:
        print(f"\n=== {ticker} ===")
        result = sr.get_levels(ticker)
        if "error" not in result:
            print(f"Current: €{result['current_price']:.2f}")
            print(f"Support: {result['support'][:3]}")
            print(f"Resistance: {result['resistance'][:3]}")
            print(f"Zone: {result['zone']}")
        else:
            print(f"Error: {result['error']}")
