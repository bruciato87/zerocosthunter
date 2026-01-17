"""
Correlation Engine - Cross-Asset Signal Propagation
====================================================
Propagates signals from one asset to correlated assets with discounted confidence.
"""

import logging
from typing import Dict, List, Tuple
from ticker_resolver import resolve_ticker

logger = logging.getLogger("CorrelationEngine")


class CorrelationEngine:
    """
    Manages asset correlations and propagates signals to correlated assets.
    """
    
    # Correlation matrix: ticker -> [(correlated_ticker, correlation_coefficient), ...]
    # Correlation coefficient ranges from 0 to 1 (1 = perfectly correlated)
    # 
    # Sources:
    # - Crypto: DeFiLlama December 2025 data, CryptoPotato analysis
    # - Stocks: MacroAxis, PortfoliosLab historical data
    #
    CORRELATION_MATRIX = {
        # =========================================================================
        # CRYPTO CORRELATIONS (December 2025 DeFiLlama data)
        # =========================================================================
        "BTC-USD": [("SOL-USD", 0.92), ("ETH-USD", 0.89), ("XRP-USD", 0.86), ("ADA-USD", 0.85), 
                    ("AVAX-USD", 0.82), ("DOT-USD", 0.80), ("LINK-USD", 0.78), ("RENDER-USD", 0.80), ("DOGE-USD", 0.75)],
        "ETH-USD": [("SOL-USD", 0.93), ("BTC-USD", 0.89), ("RENDER-USD", 0.85), ("LINK-USD", 0.82),
                    ("AVAX-USD", 0.80), ("DOT-USD", 0.78), ("ADA-USD", 0.75), ("XRP-USD", 0.75)],
        "SOL-USD": [("BTC-USD", 0.92), ("ETH-USD", 0.93), ("RENDER-USD", 0.85), ("AVAX-USD", 0.80)],
        "XRP-USD": [("BTC-USD", 0.86), ("ETH-USD", 0.75), ("SOL-USD", 0.70), ("ADA-USD", 0.70)],
        "XRP-EUR": [("BTC-USD", 0.86), ("ETH-USD", 0.75), ("SOL-USD", 0.70), ("ADA-USD", 0.70)],
        "RENDER-USD": [("ETH-USD", 0.85), ("SOL-USD", 0.85), ("BTC-USD", 0.80)],
        "ADA-USD": [("BTC-USD", 0.85), ("XRP-USD", 0.70), ("DOT-USD", 0.75)],
        "AVAX-USD": [("ETH-USD", 0.80), ("SOL-USD", 0.80), ("BTC-USD", 0.82)],
        "DOT-USD": [("BTC-USD", 0.80), ("ETH-USD", 0.78), ("ADA-USD", 0.75)],
        "LINK-USD": [("ETH-USD", 0.82), ("BTC-USD", 0.78)],
        "DOGE-USD": [("BTC-USD", 0.75), ("SHIB-USD", 0.70)],
        
        # =========================================================================
        # STOCK CORRELATIONS - TECHNOLOGY (MacroAxis/PortfoliosLab data)
        # =========================================================================
        # Semiconductors
        "NVDA": [("AMD", 0.66), ("AVGO", 0.60), ("TSM", 0.55), ("INTC", 0.50), ("QCOM", 0.55)],
        "AMD": [("NVDA", 0.66), ("INTC", 0.55), ("TSM", 0.50)],
        "INTC": [("AMD", 0.55), ("NVDA", 0.50)],
        "TSM": [("NVDA", 0.55), ("AMD", 0.50)],
        "AVGO": [("NVDA", 0.60), ("QCOM", 0.55)],
        
        # Big Tech / FAANG+
        "META": [("GOOGL", 0.63), ("SNAP", 0.50), ("PINS", 0.45)],
        "GOOGL": [("META", 0.63), ("MSFT", 0.55), ("AMZN", 0.50)],
        "AAPL": [("MSFT", 0.55), ("GOOGL", 0.50)],
        "MSFT": [("AAPL", 0.55), ("GOOGL", 0.55), ("CRM", 0.50)],
        "AMZN": [("GOOGL", 0.50), ("MSFT", 0.45)],
        
        # EV / Clean Energy
        "TSLA": [("RIVN", 0.55), ("NIO", 0.50), ("LCID", 0.50)],
        "RIVN": [("TSLA", 0.55), ("NIO", 0.50)],
        "NIO": [("TSLA", 0.50), ("RIVN", 0.50)],
        
        # AI / Software
        "PLTR": [("AI", 0.50), ("CRM", 0.45)],
        "CRM": [("MSFT", 0.50), ("PLTR", 0.45)],
        
        # =========================================================================
        # ETF CORRELATIONS
        # =========================================================================
        "EUNL.DE": [("SPY", 0.85), ("VTI", 0.80)],
        "SPY": [("EUNL.DE", 0.85), ("QQQ", 0.90), ("VTI", 0.95), ("IWM", 0.80)],
        "QQQ": [("SPY", 0.90), ("VTI", 0.85)],
        "VTI": [("SPY", 0.95), ("QQQ", 0.85)],
        "IWM": [("SPY", 0.80)],
        
        # Sector ETFs
        "XLK": [("QQQ", 0.90), ("SPY", 0.75)],  # Tech
        "XLF": [("SPY", 0.70)],  # Financials
        "XLE": [("OXY", 0.60), ("XOM", 0.65)],  # Energy
    }
    
    # Minimum confidence to trigger propagation
    MIN_CONFIDENCE_FOR_PROPAGATION = 0.80
    
    # Minimum correlation to generate a signal
    MIN_CORRELATION = 0.60
    
    # Cache for dynamic correlations (refreshed every 24h)
    _dynamic_cache = {}
    _cache_timestamp = None
    CACHE_TTL_HOURS = 24
    
    def __init__(self):
        logger.info("CorrelationEngine initialized (with dynamic correlation support)")
    
    def calculate_dynamic_correlation(self, ticker1: str, ticker2: str, period: str = "30d") -> float:
        """
        Calculate actual correlation coefficient from historical price data.
        
        Uses 30-day rolling correlation from yfinance.
        Returns the correlation coefficient (0-1 range, absolute value).
        """
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        
        try:
            # Use centralized ticker resolver
            t1 = resolve_ticker(ticker1)
            t2 = resolve_ticker(ticker2)
            
            # Download price data
            data1 = yf.download(t1, period=period, progress=False, auto_adjust=True)
            data2 = yf.download(t2, period=period, progress=False, auto_adjust=True)
            
            if data1.empty or data2.empty:
                logger.warning(f"No data for {t1} or {t2}, using static correlation")
                return self.get_static_correlation(ticker1, ticker2)
            
            # Handle MultiIndex columns
            if hasattr(data1.columns, 'levels'):
                close1 = data1['Close'].iloc[:, 0] if data1['Close'].ndim > 1 else data1['Close']
            else:
                close1 = data1['Close']
                
            if hasattr(data2.columns, 'levels'):
                close2 = data2['Close'].iloc[:, 0] if data2['Close'].ndim > 1 else data2['Close']
            else:
                close2 = data2['Close']
            
            # Align dates
            df = pd.DataFrame({'a': close1, 'b': close2}).dropna()
            
            if len(df) < 10:
                return self.get_static_correlation(ticker1, ticker2)
            
            # Calculate correlation
            correlation = df['a'].corr(df['b'])
            
            # Return absolute value (we care about magnitude, not direction)
            return abs(round(correlation, 2)) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            logger.warning(f"Dynamic correlation failed for {ticker1}-{ticker2}: {e}")
            return self.get_static_correlation(ticker1, ticker2)
    
    def get_static_correlation(self, ticker1: str, ticker2: str) -> float:
        """Get correlation from static matrix (fallback)."""
        correlations = self.CORRELATION_MATRIX.get(ticker1.upper(), [])
        for corr_ticker, corr_value in correlations:
            if corr_ticker.upper() == ticker2.upper():
                return corr_value
        return 0.0
    
    def get_correlated_assets(self, ticker: str, use_dynamic: bool = False) -> List[Tuple[str, float]]:
        """
        Get list of assets correlated with the given ticker.
        
        Returns:
            List of (ticker, correlation_coefficient) tuples
        """
        return self.CORRELATION_MATRIX.get(ticker.upper(), [])
    
    def propagate(self, source_ticker: str, sentiment: str, confidence: float, 
                  already_signaled: List[str] = None) -> List[Dict]:
        """
        Propagate a signal to correlated assets.
        
        Args:
            source_ticker: The ticker that received the original signal
            sentiment: BUY/SELL/HOLD/ACCUMULATE
            confidence: Original confidence score (0-1)
            already_signaled: List of tickers that already have signals (to avoid duplicates)
        
        Returns:
            List of propagated signal dictionaries
        """
        if already_signaled is None:
            already_signaled = []
        
        propagated_signals = []
        
        # Only propagate strong signals
        if confidence < self.MIN_CONFIDENCE_FOR_PROPAGATION:
            logger.debug(f"Confidence {confidence:.2f} < {self.MIN_CONFIDENCE_FOR_PROPAGATION}, no propagation")
            return propagated_signals
        
        # Only propagate actionable sentiments
        if sentiment not in ["BUY", "SELL", "ACCUMULATE"]:
            logger.debug(f"Sentiment {sentiment} not propagatable")
            return propagated_signals
        
        correlated = self.get_correlated_assets(source_ticker)
        
        for corr_ticker, correlation in correlated:
            # Skip if already signaled
            if corr_ticker in already_signaled:
                continue
            
            # Skip if source ticker is in already_signaled
            if source_ticker in already_signaled:
                continue
            
            # Skip weak correlations
            if correlation < self.MIN_CORRELATION:
                continue
            
            # Calculate discounted confidence
            propagated_confidence = confidence * correlation
            
            # Only generate signal if still above minimum threshold
            if propagated_confidence >= 0.60:  # Lower threshold for propagated signals
                signal = {
                    "ticker": corr_ticker,
                    "sentiment": sentiment,
                    "confidence": round(propagated_confidence, 2),
                    "source": f"Correlated with {source_ticker}",
                    "reasoning": f"Propagated from {source_ticker} ({sentiment} {confidence:.0%}) via {correlation:.0%} correlation",
                    "is_propagated": True
                }
                propagated_signals.append(signal)
                logger.info(f"Propagated: {source_ticker} → {corr_ticker} ({sentiment} {propagated_confidence:.2f})")
        
        return propagated_signals
    
    def get_correlation(self, ticker1: str, ticker2: str) -> float:
        """
        Get correlation coefficient between two tickers.
        
        Returns:
            Correlation coefficient (0-1), or 0 if not correlated
        """
        correlations = self.CORRELATION_MATRIX.get(ticker1.upper(), [])
        for corr_ticker, corr_value in correlations:
            if corr_ticker.upper() == ticker2.upper():
                return corr_value
        return 0.0
    
    def is_sector_correlated(self, ticker1: str, ticker2: str) -> bool:
        """
        Check if two tickers are in the same correlated group.
        """
        return self.get_correlation(ticker1, ticker2) > 0


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = CorrelationEngine()
    
    # Test propagation
    print("\n=== Test: BTC BUY 90% ===")
    signals = engine.propagate("BTC-USD", "BUY", 0.90)
    for s in signals:
        print(f"  → {s['ticker']}: {s['sentiment']} ({s['confidence']:.0%}) - {s['reasoning']}")
    
    print("\n=== Test: NVDA ACCUMULATE 85% ===")
    signals = engine.propagate("NVDA", "ACCUMULATE", 0.85)
    for s in signals:
        print(f"  → {s['ticker']}: {s['sentiment']} ({s['confidence']:.0%}) - {s['reasoning']}")
    
    print("\n=== Test: Low confidence (should not propagate) ===")
    signals = engine.propagate("BTC-USD", "BUY", 0.70)
    print(f"  Signals generated: {len(signals)} (expected: 0)")
