"""
Pattern Recognition Engine - Level 3 Intelligence
==================================================
Detects classic chart patterns to anticipate price movements.
Integrates with Brain, Analyze, and Rebalance modules.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("PatternRecognizer")

class PatternType(Enum):
    HEAD_AND_SHOULDERS = "Head & Shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "Inverse H&S"
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    BULL_FLAG = "Bull Flag"
    BEAR_FLAG = "Bear Flag"
    ASCENDING_WEDGE = "Ascending Wedge"
    DESCENDING_WEDGE = "Descending Wedge"
    TRIANGLE_ASCENDING = "Ascending Triangle"
    TRIANGLE_DESCENDING = "Descending Triangle"

class PatternSignal(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

@dataclass
class DetectedPattern:
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float  # 0.0 to 1.0
    description: str
    target_move_pct: float  # Expected % move if pattern confirms

class PatternRecognizer:
    """
    Analyzes price data to detect classic chart patterns.
    Uses pivot point detection and geometric analysis.
    """
    
    # Ticker aliases (fallback if DB cache miss)
    TICKER_ALIASES = {
        "RENDER": "RENDER-USD",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "XRP": "XRP-USD",
        "TCT": "0700.HK",
        "3XC": "1810.HK",
        "NUKL": "U3O8.DE",
        "JAZZ": "JAZZ",
    }
    
    def __init__(self):
        logger.info("PatternRecognizer initialized")
    
    def _get_resolved_ticker(self, ticker: str) -> str:
        """
        Resolve ticker using DB cache first, then local aliases.
        """
        ticker_u = ticker.upper()
        
        # 1. Check DB cache first
        try:
            from db_handler import DBHandler
            db = DBHandler()
            cached = db.get_ticker_cache(ticker_u)
            if cached:
                return cached.get("resolved_ticker", ticker_u)
        except:
            pass
        
        # 2. Fallback to local aliases
        return self.TICKER_ALIASES.get(ticker_u, ticker_u)
    
    def _get_price_data(self, ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch OHLC data for pattern analysis."""
        try:
            search_ticker = self._get_resolved_ticker(ticker)
            stock = yf.Ticker(search_ticker)
            df = stock.history(period=period)
            if df.empty:
                logger.warning(f"No data for {ticker} (resolved: {search_ticker})")
                return None
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _find_pivots(self, df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """
        Find local highs (resistance) and lows (support) pivot points.
        Returns (pivot_highs, pivot_lows) as lists of (index, price) tuples.
        """
        highs = []
        lows = []
        
        prices_high = df['High'].values
        prices_low = df['Low'].values
        
        for i in range(window, len(df) - window):
            # Check if this is a local high
            if prices_high[i] == max(prices_high[i-window:i+window+1]):
                highs.append((i, prices_high[i]))
            # Check if this is a local low
            if prices_low[i] == min(prices_low[i-window:i+window+1]):
                lows.append((i, prices_low[i]))
        
        return highs, lows
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame, pivots_high: List, pivots_low: List) -> Optional[DetectedPattern]:
        """
        Detect Head & Shoulders pattern (bearish reversal).
        Structure: Left Shoulder < Head > Right Shoulder, neckline connects lows.
        """
        if len(pivots_high) < 3 or len(pivots_low) < 2:
            return None
        
        # Get recent pivots (last 3 highs for H&S)
        recent_highs = pivots_high[-5:] if len(pivots_high) >= 5 else pivots_high
        
        for i in range(len(recent_highs) - 2):
            left_shoulder = recent_highs[i]
            head = recent_highs[i + 1]
            right_shoulder = recent_highs[i + 2]
            
            # Head must be higher than both shoulders
            if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                # Shoulders should be roughly equal (within 10%)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                if shoulder_diff < 0.15:
                    # Calculate confidence based on symmetry
                    confidence = 0.70 + (0.15 - shoulder_diff) * 2
                    confidence = min(confidence, 0.95)
                    
                    # Calculate target (head to neckline distance projected down)
                    neckline = min(left_shoulder[1], right_shoulder[1])
                    target_move = -((head[1] - neckline) / neckline * 100)
                    
                    return DetectedPattern(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        signal=PatternSignal.BEARISH,
                        confidence=round(confidence, 2),
                        description=f"H&S detected: Head at {head[1]:.2f}, shoulders ~{left_shoulder[1]:.2f}",
                        target_move_pct=round(target_move, 1)
                    )
        return None
    
    def _detect_inverse_head_and_shoulders(self, df: pd.DataFrame, pivots_high: List, pivots_low: List) -> Optional[DetectedPattern]:
        """
        Detect Inverse Head & Shoulders pattern (bullish reversal).
        """
        if len(pivots_low) < 3:
            return None
        
        recent_lows = pivots_low[-5:] if len(pivots_low) >= 5 else pivots_low
        
        for i in range(len(recent_lows) - 2):
            left_shoulder = recent_lows[i]
            head = recent_lows[i + 1]
            right_shoulder = recent_lows[i + 2]
            
            # Head must be lower than both shoulders
            if head[1] < left_shoulder[1] and head[1] < right_shoulder[1]:
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                if shoulder_diff < 0.15:
                    confidence = 0.70 + (0.15 - shoulder_diff) * 2
                    confidence = min(confidence, 0.95)
                    
                    neckline = max(left_shoulder[1], right_shoulder[1])
                    target_move = ((neckline - head[1]) / head[1] * 100)
                    
                    return DetectedPattern(
                        pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                        signal=PatternSignal.BULLISH,
                        confidence=round(confidence, 2),
                        description=f"Inverse H&S: Head at {head[1]:.2f}, shoulders ~{left_shoulder[1]:.2f}",
                        target_move_pct=round(target_move, 1)
                    )
        return None
    
    def _detect_double_top(self, df: pd.DataFrame, pivots_high: List) -> Optional[DetectedPattern]:
        """
        Detect Double Top pattern (bearish reversal).
        Two peaks at similar levels with a trough between them.
        """
        if len(pivots_high) < 2:
            return None
        
        recent_highs = pivots_high[-4:] if len(pivots_high) >= 4 else pivots_high
        
        for i in range(len(recent_highs) - 1):
            peak1 = recent_highs[i]
            peak2 = recent_highs[i + 1]
            
            # Peaks should be at similar levels (within 5%)
            peak_diff = abs(peak1[1] - peak2[1]) / peak1[1]
            if peak_diff < 0.05:
                # Need some separation between peaks
                if peak2[0] - peak1[0] > 10:  # At least 10 bars apart
                    confidence = 0.75 + (0.05 - peak_diff) * 4
                    confidence = min(confidence, 0.90)
                    
                    # Target is typically the height of the pattern
                    current_price = df['Close'].iloc[-1]
                    target_move = -((peak1[1] - current_price) / current_price * 100) * 0.8
                    
                    return DetectedPattern(
                        pattern_type=PatternType.DOUBLE_TOP,
                        signal=PatternSignal.BEARISH,
                        confidence=round(confidence, 2),
                        description=f"Double Top at ~{peak1[1]:.2f}",
                        target_move_pct=round(target_move, 1)
                    )
        return None
    
    def _detect_double_bottom(self, df: pd.DataFrame, pivots_low: List) -> Optional[DetectedPattern]:
        """
        Detect Double Bottom pattern (bullish reversal).
        """
        if len(pivots_low) < 2:
            return None
        
        recent_lows = pivots_low[-4:] if len(pivots_low) >= 4 else pivots_low
        
        for i in range(len(recent_lows) - 1):
            trough1 = recent_lows[i]
            trough2 = recent_lows[i + 1]
            
            trough_diff = abs(trough1[1] - trough2[1]) / trough1[1]
            if trough_diff < 0.05:
                if trough2[0] - trough1[0] > 10:
                    confidence = 0.75 + (0.05 - trough_diff) * 4
                    confidence = min(confidence, 0.90)
                    
                    current_price = df['Close'].iloc[-1]
                    target_move = ((current_price - trough1[1]) / trough1[1] * 100) * 0.8
                    
                    return DetectedPattern(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        signal=PatternSignal.BULLISH,
                        confidence=round(confidence, 2),
                        description=f"Double Bottom at ~{trough1[1]:.2f}",
                        target_move_pct=round(target_move, 1)
                    )
        return None
    
    def _detect_flag(self, df: pd.DataFrame) -> Optional[DetectedPattern]:
        """
        Detect Bull/Bear Flag patterns (continuation).
        Strong move followed by consolidation channel.
        """
        if len(df) < 30:
            return None
        
        # Look at last 30 bars
        recent = df.tail(30)
        earlier = df.tail(60).head(30) if len(df) >= 60 else df.head(30)
        
        # Calculate trend in earlier period
        earlier_return = (earlier['Close'].iloc[-1] - earlier['Close'].iloc[0]) / earlier['Close'].iloc[0] * 100
        
        # Calculate volatility in recent period (flag should be low volatility)
        recent_volatility = recent['Close'].std() / recent['Close'].mean() * 100
        earlier_volatility = earlier['Close'].std() / earlier['Close'].mean() * 100
        
        # Flag pattern: strong move followed by lower volatility consolidation
        if abs(earlier_return) > 15 and recent_volatility < earlier_volatility * 0.6:
            is_bullish = earlier_return > 0
            
            # Calculate slight counter-trend in flag
            recent_return = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0] * 100
            
            # For bull flag, recent should have slight downward drift or flat
            # For bear flag, recent should have slight upward drift or flat
            valid_flag = (is_bullish and recent_return < 5) or (not is_bullish and recent_return > -5)
            
            if valid_flag:
                confidence = 0.65 + (earlier_volatility - recent_volatility) / earlier_volatility * 0.25
                confidence = min(max(confidence, 0.50), 0.85)
                
                # Target is typically the pole height added to breakout
                target_move = earlier_return * 0.7  # Conservative target
                
                pattern_type = PatternType.BULL_FLAG if is_bullish else PatternType.BEAR_FLAG
                signal = PatternSignal.BULLISH if is_bullish else PatternSignal.BEARISH
                
                return DetectedPattern(
                    pattern_type=pattern_type,
                    signal=signal,
                    confidence=round(confidence, 2),
                    description=f"{'Bull' if is_bullish else 'Bear'} Flag forming after {earlier_return:.1f}% move",
                    target_move_pct=round(target_move, 1)
                )
        return None
    
    def _detect_wedge(self, df: pd.DataFrame, pivots_high: List, pivots_low: List) -> Optional[DetectedPattern]:
        """
        Detect Ascending/Descending Wedge patterns.
        Converging trendlines with both sloping in same direction.
        """
        if len(pivots_high) < 3 or len(pivots_low) < 3:
            return None
        
        # Get recent pivots
        recent_highs = pivots_high[-5:]
        recent_lows = pivots_low[-5:]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None
        
        # Calculate slopes
        high_slope = (recent_highs[-1][1] - recent_highs[0][1]) / max(1, (recent_highs[-1][0] - recent_highs[0][0]))
        low_slope = (recent_lows[-1][1] - recent_lows[0][1]) / max(1, (recent_lows[-1][0] - recent_lows[0][0]))
        
        # Wedge: both trendlines slope in same direction
        both_rising = high_slope > 0 and low_slope > 0
        both_falling = high_slope < 0 and low_slope < 0
        
        # Lines should be converging
        range_start = recent_highs[0][1] - recent_lows[0][1]
        range_end = recent_highs[-1][1] - recent_lows[-1][1]
        converging = range_end < range_start * 0.8
        
        if converging and (both_rising or both_falling):
            if both_rising:
                pattern_type = PatternType.ASCENDING_WEDGE
                signal = PatternSignal.BEARISH  # Ascending wedge is bearish
                description = "Ascending Wedge (bearish reversal pattern)"
                target_move = -15.0  # Typical breakdown target
            else:
                pattern_type = PatternType.DESCENDING_WEDGE
                signal = PatternSignal.BULLISH  # Descending wedge is bullish
                description = "Descending Wedge (bullish reversal pattern)"
                target_move = 15.0  # Typical breakout target
            
            # Confidence based on how well defined the wedge is
            convergence_ratio = range_end / range_start
            confidence = 0.60 + (0.8 - convergence_ratio) * 0.4
            confidence = min(max(confidence, 0.50), 0.85)
            
            return DetectedPattern(
                pattern_type=pattern_type,
                signal=signal,
                confidence=round(confidence, 2),
                description=description,
                target_move_pct=target_move
            )
        return None
    
    def detect_patterns(self, ticker: str, period: str = "6mo") -> List[DetectedPattern]:
        """
        Main method: Detect all patterns for a given ticker.
        Returns list of detected patterns sorted by confidence.
        """
        df = self._get_price_data(ticker, period)
        if df is None or len(df) < 30:
            return []
        
        patterns = []
        
        # Find pivot points
        pivots_high, pivots_low = self._find_pivots(df, window=5)
        
        # Run all pattern detectors
        detectors = [
            lambda: self._detect_head_and_shoulders(df, pivots_high, pivots_low),
            lambda: self._detect_inverse_head_and_shoulders(df, pivots_high, pivots_low),
            lambda: self._detect_double_top(df, pivots_high),
            lambda: self._detect_double_bottom(df, pivots_low),
            lambda: self._detect_flag(df),
            lambda: self._detect_wedge(df, pivots_high, pivots_low),
        ]
        
        for detector in detectors:
            try:
                result = detector()
                if result:
                    patterns.append(result)
            except Exception as e:
                logger.warning(f"Pattern detector error: {e}")
        
        # Sort by confidence descending
        patterns.sort(key=lambda x: -x.confidence)
        
        logger.info(f"Detected {len(patterns)} patterns for {ticker}")
        return patterns
    
    def get_pattern_summary(self, ticker: str) -> str:
        """
        Returns a human-readable summary for AI prompt injection.
        """
        patterns = self.detect_patterns(ticker)
        
        if not patterns:
            return f"[PATTERN ANALYSIS: No significant chart patterns detected for {ticker}]"
        
        lines = [f"[PATTERN ANALYSIS for {ticker}]"]
        for p in patterns[:3]:  # Top 3 patterns
            lines.append(f"  - {p.pattern_type.value}: {p.signal.value} ({int(p.confidence*100)}% conf)")
            lines.append(f"    └ {p.description}")
            lines.append(f"    └ Target Move: {p.target_move_pct:+.1f}%")
        
        return "\n".join(lines)
    
    def get_pattern_bias(self, ticker: str) -> Tuple[str, float]:
        """
        Returns overall pattern bias and confidence modifier.
        Used for signal confidence adjustment.
        
        Returns: (bias: "BULLISH"/"BEARISH"/"NEUTRAL", confidence_modifier: 0.9-1.1)
        """
        patterns = self.detect_patterns(ticker)
        
        if not patterns:
            return "NEUTRAL", 1.0
        
        # Weight by confidence
        bullish_score = sum(p.confidence for p in patterns if p.signal == PatternSignal.BULLISH)
        bearish_score = sum(p.confidence for p in patterns if p.signal == PatternSignal.BEARISH)
        
        if bullish_score > bearish_score + 0.2:
            modifier = 1.0 + min(bullish_score * 0.1, 0.15)
            return "BULLISH", round(modifier, 2)
        elif bearish_score > bullish_score + 0.2:
            modifier = 1.0 - min(bearish_score * 0.1, 0.15)
            return "BEARISH", round(modifier, 2)
        else:
            return "NEUTRAL", 1.0


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    pr = PatternRecognizer()
    
    # Test with Bitcoin
    print("\n=== Testing BTC-USD ===")
    print(pr.get_pattern_summary("BTC-USD"))
    
    # Test with RENDER
    print("\n=== Testing RENDER ===")
    print(pr.get_pattern_summary("RENDER"))
    
    # Test bias
    bias, mod = pr.get_pattern_bias("ETH")
    print(f"\nETH Bias: {bias}, Modifier: {mod}")
