"""
Signal Intelligence Module - Enhanced Trading Logic
====================================================
Provides advanced signal filtering and enhancement based on:
1. Portfolio Correlation - Avoid overconcentration
2. DCA on Pullback - Optimal entry timing
3. Trailing Take-Profit - Lock in gains
4. Market Regime Detection - Adjust for macro conditions
5. Earnings Calendar - Event risk management
"""

import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from ticker_resolver import resolve_ticker, is_probable_ticker

logger = logging.getLogger("SignalIntelligence")


class SignalIntelligence:
    """
    Enhanced signal processing with multiple intelligence layers.
    Called before final signal generation to adjust confidence and actions.
    """
    
    def __init__(self, market_instance=None, advisor_instance=None):
        from db_handler import DBHandler
        from market_data import MarketData
        from advisor import Advisor
        
        self.db = DBHandler()
        self.market = market_instance if market_instance else MarketData()
        self.advisor = advisor_instance if advisor_instance else Advisor()
        
        # Cache for expensive lookups
        self._vix_cache = None
        self._sp500_trend_cache = None
        self._earnings_cache = {}
    
    # =========================================================================
    # 1. PORTFOLIO CORRELATION CHECK
    # =========================================================================
    
    def check_portfolio_correlation(self, ticker: str, sentiment: str, portfolio_context: List = None) -> Dict:
        """
        Check if adding this asset would overconcentrate the portfolio.
        Accepts optional portfolio_context (list of dicts) to avoid DB calls.
        """
        if sentiment not in ["BUY", "ACCUMULATE"]:
            return {"should_downgrade": False}
        
        try:
            # Use cached context if provided, otherwise fetch from DB (fallback)
            portfolio = portfolio_context if portfolio_context is not None else self.db.get_portfolio()
            if not portfolio:
                return {"should_downgrade": False}
            
            # Get sector of new asset
            new_sector = self.advisor.get_sector(ticker)
            
            # Calculate current sector allocations
            sector_values = {}
            total_value = 0.0
            
            for item in portfolio:
                qty = float(item.get('quantity', 0))
                avg = float(item.get('avg_price', 0))
                value = qty * avg  # Use cost basis as proxy
                total_value += value
                
                s = self.advisor.get_sector(item.get('ticker', ''))
                sector_values[s] = sector_values.get(s, 0.0) + value
            
            if total_value <= 0:
                return {"should_downgrade": False}
            
            # Current allocation of the target sector
            current_alloc = (sector_values.get(new_sector, 0) / total_value) * 100
            
            # Threshold: 50% max per sector (relaxed from 40%)
            if current_alloc >= 50:
                return {
                    "should_downgrade": True,
                    "reason": f"{new_sector} sector already at {current_alloc:.1f}% (max 50%)",
                    "sector": new_sector,
                    "current_allocation": current_alloc,
                    "action": "Downgrade BUY to HOLD"
                }
            
            # Warning zone: 40-50%
            if current_alloc >= 40:
                return {
                    "should_downgrade": False,
                    "warning": f"{new_sector} sector at {current_alloc:.1f}% - approaching limit",
                    "sector": new_sector,
                    "current_allocation": current_alloc
                }
            
            return {"should_downgrade": False, "sector": new_sector, "current_allocation": current_alloc}
            
        except Exception as e:
            logger.warning(f"Portfolio correlation check failed: {e}")
            return {"should_downgrade": False, "error": str(e)}
    
    # =========================================================================
    # 1.5 TECHNICAL CONFLUENCE CHECK (NEW - Predictive System L1)
    # =========================================================================
    
    def check_technical_confluence(self, ticker: str, sentiment: str) -> Dict:
        """
        Check if multiple technical indicators confirm the signal direction.
        
        Checks:
        1. RSI (oversold for BUY, overbought for SELL)
        2. Price vs SMA50 (above = bullish, below = bearish)
        3. Volume spike (>150% of 20d avg = strong conviction)
        
        Returns:
            {
                "alignment": 0-3 (how many indicators confirm),
                "multiplier": 0.85-1.15 (confidence adjustment),
                "indicators": {...detailed breakdown...}
            }
        """
        try:
            # Use centralized ticker resolver for self-learning cache
            yf_ticker = resolve_ticker(ticker)
            if not yf_ticker:
                 return {"alignment": 0, "multiplier": 1.0, "reason": "Ticker unresolvable/rejected"}
            
            data = yf.download(yf_ticker, period="60d", progress=False, auto_adjust=True)
            
            if data.empty or len(data) < 50:
                return {"alignment": 0, "multiplier": 1.0, "reason": "Insufficient data"}
            
            # Handle MultiIndex columns
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
                volume = data['Volume'].iloc[:, 0] if data['Volume'].ndim > 1 else data['Volume']
            else:
                close = data['Close']
                volume = data['Volume']
            
            current_price = float(close.iloc[-1])
            
            # 1. RSI Calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
            
            # 2. SMA50
            sma50 = float(close.rolling(window=50).mean().iloc[-1])
            price_above_sma = current_price > sma50
            
            # 3. Volume Spike
            avg_volume = float(volume.rolling(window=20).mean().iloc[-1])
            current_volume = float(volume.iloc[-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_spike = volume_ratio > 1.5
            
            # Determine alignment based on sentiment
            alignment = 0
            reasons = []
            
            if sentiment in ["BUY", "ACCUMULATE"]:
                # For BUY: RSI < 45 is good, price above SMA is bullish, volume spike confirms
                if current_rsi < 45:
                    alignment += 1
                    reasons.append(f"RSI {current_rsi:.0f} (not overbought)")
                if price_above_sma:
                    alignment += 1
                    reasons.append("Price > SMA50 (bullish)")
                if volume_spike:
                    alignment += 1
                    reasons.append(f"Volume spike {volume_ratio:.1f}x")
                    
            elif sentiment in ["SELL", "TRIM"]:
                # For SELL: RSI > 70 is overbought (standard), price below SMA is bearish
                if current_rsi > 70:
                    alignment += 1
                    reasons.append(f"RSI {current_rsi:.0f} (overbought)")
                if not price_above_sma:
                    alignment += 1
                    reasons.append("Price < SMA50 (bearish)")
                if volume_spike:
                    alignment += 1
                    reasons.append(f"Volume spike {volume_ratio:.1f}x")
            
            # Calculate multiplier
            if alignment >= 3:
                multiplier = 1.15  # Strong confluence
            elif alignment >= 2:
                multiplier = 1.05  # Moderate confluence
            elif alignment >= 1:
                multiplier = 1.0   # Neutral
            else:
                multiplier = 0.90  # No confluence, reduce confidence
            
            return {
                "alignment": alignment,
                "multiplier": multiplier,
                "indicators": {
                    "rsi": round(current_rsi, 1),
                    "sma50": round(sma50, 2),
                    "price": round(current_price, 2),
                    "price_vs_sma": "above" if price_above_sma else "below",
                    "volume_ratio": round(volume_ratio, 2),
                    "volume_spike": volume_spike
                },
                "reasons": reasons,
                "reason": " | ".join(reasons) if reasons else "No technical confluence"
            }
            
        except Exception as e:
            logger.warning(f"Technical confluence check failed for {ticker}: {e}")
            return {"alignment": 0, "multiplier": 1.0, "error": str(e)}
    
    # =========================================================================
    # L2: DIVERGENCE DETECTOR (NEW)
    # =========================================================================
    
    def check_divergence(self, ticker: str, lookback_days: int = 20) -> Dict:
        """
        Detect RSI divergence - early reversal signals.
        
        Bullish Divergence: Price makes lower low, RSI makes higher low
        Bearish Divergence: Price makes higher high, RSI makes lower high
        
        Returns:
            {
                "has_divergence": True/False,
                "type": "bullish" | "bearish" | None,
                "strength": 0.0-1.0,
                "confidence_boost": 1.0-1.20
            }
        """
        try:
            # Use centralized ticker resolver
            yf_ticker = resolve_ticker(ticker)
            if not yf_ticker:
                return {"has_divergence": False, "type": None, "strength": 0, "confidence_boost": 1.0}
            
            data = yf.download(yf_ticker, period=f"{lookback_days + 14}d", progress=False, auto_adjust=True)
            
            if data.empty or len(data) < lookback_days:
                return {"has_divergence": False, "type": None, "strength": 0, "confidence_boost": 1.0}
            
            # Handle MultiIndex
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                close = data['Close']
            
            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get recent data
            recent_close = close.iloc[-lookback_days:]
            recent_rsi = rsi.iloc[-lookback_days:]
            
            # Find local minima/maxima
            price_min_idx = recent_close.idxmin()
            price_max_idx = recent_close.idxmax()
            
            # Split into two halves for comparison
            half = lookback_days // 2
            first_half_close = close.iloc[-lookback_days:-half]
            second_half_close = close.iloc[-half:]
            first_half_rsi = rsi.iloc[-lookback_days:-half]
            second_half_rsi = rsi.iloc[-half:]
            
            has_divergence = False
            div_type = None
            strength = 0.0
            
            # Check Bullish Divergence: Price lower low, RSI higher low
            first_min_price = float(first_half_close.min())
            second_min_price = float(second_half_close.min())
            first_min_rsi = float(first_half_rsi.min())
            second_min_rsi = float(second_half_rsi.min())
            
            if second_min_price < first_min_price and second_min_rsi > first_min_rsi:
                has_divergence = True
                div_type = "bullish"
                # Strength based on how far apart the divergence is
                price_diff = (first_min_price - second_min_price) / first_min_price
                rsi_diff = (second_min_rsi - first_min_rsi) / 100
                strength = min(1.0, (price_diff + rsi_diff) * 5)
                logger.info(f"Bullish divergence detected for {ticker}: Price ↓{price_diff:.1%}, RSI ↑{rsi_diff:.1%}")
            
            # Check Bearish Divergence: Price higher high, RSI lower high
            first_max_price = float(first_half_close.max())
            second_max_price = float(second_half_close.max())
            first_max_rsi = float(first_half_rsi.max())
            second_max_rsi = float(second_half_rsi.max())
            
            if second_max_price > first_max_price and second_max_rsi < first_max_rsi:
                has_divergence = True
                div_type = "bearish"
                price_diff = (second_max_price - first_max_price) / first_max_price
                rsi_diff = (first_max_rsi - second_max_rsi) / 100
                strength = min(1.0, (price_diff + rsi_diff) * 5)
                logger.info(f"Bearish divergence detected for {ticker}: Price ↑{price_diff:.1%}, RSI ↓{rsi_diff:.1%}")
            
            # Confidence boost based on divergence
            if has_divergence:
                confidence_boost = 1.0 + (strength * 0.15)  # Max +15%
            else:
                confidence_boost = 1.0
            
            return {
                "has_divergence": has_divergence,
                "type": div_type,
                "strength": round(strength, 2),
                "confidence_boost": round(confidence_boost, 2),
                "lookback_days": lookback_days
            }
            
        except Exception as e:
            logger.warning(f"Divergence check failed for {ticker}: {e}")
            return {"has_divergence": False, "type": None, "strength": 0, "confidence_boost": 1.0, "error": str(e)}
    
    # =========================================================================
    # 2. DCA ON PULLBACK LOGIC
    # =========================================================================
    
    def check_dca_opportunity(self, ticker: str) -> Dict:
        """
        Check if current price is a good DCA entry (pullback from recent high).
        
        Returns:
            {
                "is_good_entry": True/False,
                "pullback_pct": -15.5,
                "rsi": 35.2,
                "reason": "Price down 15% from 30d high, RSI oversold"
            }
        """
        try:
            # Use centralized ticker resolver
            yf_ticker = resolve_ticker(ticker)
            if not yf_ticker:
                 return {"is_good_entry": False, "reason": "Ticker unresolvable/rejected"}
            
            data = yf.download(yf_ticker, period="30d", progress=False, auto_adjust=True)
            
            if data.empty or len(data) < 10:
                return {"is_good_entry": False, "reason": "Insufficient data"}
            
            # Handle MultiIndex columns
            if hasattr(data.columns, 'levels'):
                close = data['Close'].iloc[:, 0] if data['Close'].ndim > 1 else data['Close']
            else:
                close = data['Close']
            
            current_price = float(close.iloc[-1])
            high_30d = float(close.max())
            
            # Calculate pullback percentage
            pullback_pct = ((current_price - high_30d) / high_30d) * 100
            
            # Calculate RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
            
            # Good entry conditions:
            # 1. Price down at least 10% from 30d high
            # 2. RSI < 45 (not overbought)
            is_good_entry = pullback_pct <= -10 and current_rsi < 45
            
            # Excellent entry (for higher confidence boost)
            is_excellent = pullback_pct <= -20 and current_rsi < 35
            
            reason = []
            if pullback_pct <= -10:
                reason.append(f"Price down {abs(pullback_pct):.1f}% from 30d high")
            if current_rsi < 40:
                reason.append(f"RSI oversold ({current_rsi:.0f})")
            
            return {
                "is_good_entry": is_good_entry,
                "is_excellent_entry": is_excellent,
                "pullback_pct": pullback_pct,
                "rsi": current_rsi,
                "high_30d": high_30d,
                "current_price": current_price,
                "reason": " | ".join(reason) if reason else "No pullback detected"
            }
            
        except Exception as e:
            logger.warning(f"DCA check failed for {ticker}: {e}")
            return {"is_good_entry": False, "error": str(e)}
    
    # =========================================================================
    # 3. TRAILING TAKE-PROFIT
    # =========================================================================
    
    def check_take_profit(self, ticker: str, portfolio_context: List = None) -> Dict:
        """
        Check if a position should trim profits.
        Accepts optional portfolio_context to avoid DB calls.
        """
        try:
            # Use cached context if provided, otherwise fetch
            portfolio = portfolio_context if portfolio_context is not None else self.db.get_portfolio()
            if not portfolio:
                return {"should_take_profit": False}
            
            if not ticker:
                return {"should_take_profit": False}
            
            # Find the asset in portfolio
            asset_data = None
            total_value = 0.0
            
            for item in portfolio:
                qty = float(item.get('quantity', 0))
                price, _ = self.market.get_smart_price_eur(item.get('ticker', ''))
                if price <= 0:
                    price = float(item.get('avg_price', 0))
                value = qty * price
                total_value += value
                
                # Normalize ticker comparison
                item_ticker = item.get('ticker', '').upper().replace('-USD', '')
                check_ticker = ticker.upper().replace('-USD', '')
                
                if item_ticker == check_ticker:
                    asset_data = item
                    asset_data['current_value'] = value
            
            if not asset_data or total_value <= 0:
                return {"should_take_profit": False}
            
            qty = float(asset_data.get('quantity', 0))
            avg_price = float(asset_data.get('avg_price', 0))
            current_value = asset_data['current_value']
            
            # Calculate PnL
            cost_basis = qty * avg_price
            pnl_pct = ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0
            
            # Calculate allocation
            allocation_pct = (current_value / total_value) * 100
            
            # Get RSI
            dca_check = self.check_dca_opportunity(ticker)
            rsi = dca_check.get('rsi', 50)
            
            # Take profit conditions:
            # 1. PnL > 40%
            # 2. Allocation > 15% of portfolio
            # 3. RSI > 65 (approaching overbought)
            conditions_met = 0
            reasons = []
            
            if pnl_pct >= 40:
                conditions_met += 1
                reasons.append(f"Up {pnl_pct:.1f}%")
            
            if allocation_pct >= 15:
                conditions_met += 1
                reasons.append(f"{allocation_pct:.1f}% of portfolio")
            
            if rsi >= 65:
                conditions_met += 1
                reasons.append(f"RSI high ({rsi:.0f})")
            
            # Suggest take profit if at least 2 conditions met
            should_take = conditions_met >= 2 and pnl_pct >= 30
            
            return {
                "should_take_profit": should_take,
                "pnl_pct": pnl_pct,
                "allocation_pct": allocation_pct,
                "rsi": rsi,
                "conditions_met": conditions_met,
                "reason": " | ".join(reasons) if reasons else "No take-profit trigger",
                "suggested_trim": "20%" if should_take else None
            }
            
        except Exception as e:
            logger.warning(f"Take profit check failed for {ticker}: {e}")
            return {"should_take_profit": False, "error": str(e)}
    
    # =========================================================================
    # 4. MARKET REGIME DETECTION
    # =========================================================================
    
    def get_market_regime(self) -> Dict:
        """
        Detect current market regime based on VIX and S&P500 trend.
        
        Returns:
            {
                "regime": "RISK_ON" | "RISK_OFF" | "NEUTRAL",
                "vix": 18.5,
                "vix_level": "LOW" | "ELEVATED" | "HIGH" | "EXTREME",
                "sp500_trend": "BULLISH" | "BEARISH" | "NEUTRAL",
                "confidence_adjustment": 0.0 | -0.1 | -0.2
            }
        """
        try:
            # 0. Check internal session cache (Valid for 15 minutes)
            if self._vix_cache and self._sp500_trend_cache:
                 logger.debug("Using cached market regime data.")
                 return {
                    "regime": self._regime_cache if hasattr(self, '_regime_cache') else "NEUTRAL",
                    "vix": self._vix_cache,
                    "vix_level": self._vix_level_cache if hasattr(self, '_vix_level_cache') else "UNKNOWN",
                    "sp500_trend": self._sp500_trend_cache,
                    "confidence_adjustment": self._conf_adj_cache if hasattr(self, '_conf_adj_cache') else 0.0
                 }

            result = {
                "regime": "NEUTRAL",
                "vix": None,
                "vix_level": "UNKNOWN",
                "sp500_trend": "UNKNOWN",
                "confidence_adjustment": 0.0
            }
            
            # Get VIX
            try:
                vix_data = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
                if not vix_data.empty:
                    if hasattr(vix_data.columns, 'levels'):
                        vix = float(vix_data['Close'].iloc[-1, 0]) if vix_data['Close'].ndim > 1 else float(vix_data['Close'].iloc[-1])
                    else:
                        vix = float(vix_data['Close'].iloc[-1])
                    
                    result["vix"] = vix
                    
                    if vix < 15:
                        result["vix_level"] = "LOW"
                    elif vix < 20:
                        result["vix_level"] = "NORMAL"
                    elif vix < 30:
                        result["vix_level"] = "ELEVATED"
                    elif vix < 40:
                        result["vix_level"] = "HIGH"
                    else:
                        result["vix_level"] = "EXTREME"
            except Exception as e:
                logger.warning(f"VIX fetch failed: {e}")
            
            # Get S&P500 trend (price vs 200 SMA)
            try:
                sp_data = yf.download("^GSPC", period="1y", progress=False, auto_adjust=True)
                if not sp_data.empty and len(sp_data) > 200:
                    if hasattr(sp_data.columns, 'levels'):
                        close = sp_data['Close'].iloc[:, 0] if sp_data['Close'].ndim > 1 else sp_data['Close']
                    else:
                        close = sp_data['Close']
                    
                    current = float(close.iloc[-1])
                    sma200 = float(close.rolling(200).mean().iloc[-1])
                    
                    if current > sma200 * 1.02:  # 2% above
                        result["sp500_trend"] = "BULLISH"
                    elif current < sma200 * 0.98:  # 2% below
                        result["sp500_trend"] = "BEARISH"
                    else:
                        result["sp500_trend"] = "NEUTRAL"
            except Exception as e:
                logger.warning(f"S&P500 trend fetch failed: {e}")
            
            # Determine regime and confidence adjustment (RELAXED penalties)
            vix = result.get("vix", 20)
            
            if result["vix_level"] in ["HIGH", "EXTREME"]:
                result["regime"] = "RISK_OFF"
                result["confidence_adjustment"] = -0.05  # Reduced from -0.15
            elif result["vix_level"] == "ELEVATED" or result["sp500_trend"] == "BEARISH":
                result["regime"] = "CAUTIOUS"
                result["confidence_adjustment"] = -0.03  # Reduced from -0.10
            elif result["vix_level"] == "LOW" and result["sp500_trend"] == "BULLISH":
                result["regime"] = "RISK_ON"
                result["confidence_adjustment"] = 0.05

            # Cache results for the current session
            self._vix_cache = result.get("vix")
            self._sp500_trend_cache = result.get("sp500_trend")
            self._regime_cache = result["regime"]
            self._vix_level_cache = result["vix_level"]
            self._conf_adj_cache = result["confidence_adjustment"]
            
            return result
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            return {"regime": "NEUTRAL", "confidence_adjustment": 0.0}
    
    # =========================================================================
    # 5. EARNINGS CALENDAR
    # =========================================================================
    
    def check_earnings_risk(self, ticker: str) -> Dict:
        """
        Check if ticker has earnings coming up (high volatility event).
        
        Returns:
            {
                "has_upcoming_earnings": True/False,
                "earnings_date": "2026-01-15",
                "days_until": 5,
                "risk_level": "HIGH" | "MEDIUM" | "LOW" | "NONE"
            }
        """
        # Skip for crypto and ETFs/Indices (no earnings)
        if not ticker:
            return {"has_upcoming_earnings": False, "risk_level": "NONE", "reason": "No ticker provided"}

        ticker_u = ticker.upper().strip()
        if not is_probable_ticker(ticker_u):
            return {"has_upcoming_earnings": False, "risk_level": "NONE", "reason": "Filtered ticker noise"}

        if ticker_u.replace('-USD', '') in ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT', 'AVAX', 'LINK', 'RENDER', 'RNDR'] \
           or ticker_u in ['SPY', 'QQQ', 'DIA', 'IWM', 'EUNL.DE', 'GSPC', 'VIX']:
            return {"has_upcoming_earnings": False, "risk_level": "NONE", "reason": "Asset type has no earnings"}
        
        try:
            # Check cache first
            cache_key = f"{ticker_u}_{datetime.now().strftime('%Y-%m-%d')}"
            if cache_key in self._earnings_cache:
                return self._earnings_cache[cache_key]
            
            yf_ticker = resolve_ticker(ticker_u)
            if not yf_ticker:
                result = {"has_upcoming_earnings": False, "risk_level": "NONE", "reason": "Ticker unresolvable"}
                self._earnings_cache[cache_key] = result
                return result
            t = yf.Ticker(yf_ticker)
            
            # Get earnings dates
            try:
                earnings = t.calendar
                if earnings is None or earnings.empty:
                    result = {"has_upcoming_earnings": False, "risk_level": "NONE"}
                    self._earnings_cache[cache_key] = result
                    return result
                
                # earnings is typically a DataFrame with 'Earnings Date' row
                if 'Earnings Date' in earnings.index:
                    next_earnings = earnings.loc['Earnings Date', 0]
                    if isinstance(next_earnings, str):
                        next_earnings = datetime.strptime(next_earnings, '%Y-%m-%d')
                    
                    days_until = (next_earnings - datetime.now()).days
                    
                    if days_until < 0:
                        result = {"has_upcoming_earnings": False, "risk_level": "NONE"}
                    elif days_until <= 3:
                        result = {
                            "has_upcoming_earnings": True,
                            "earnings_date": next_earnings.strftime('%Y-%m-%d'),
                            "days_until": days_until,
                            "risk_level": "HIGH",
                            "action": "Downgrade BUY to HOLD"
                        }
                    elif days_until <= 7:
                        result = {
                            "has_upcoming_earnings": True,
                            "earnings_date": next_earnings.strftime('%Y-%m-%d'),
                            "days_until": days_until,
                            "risk_level": "MEDIUM",
                            "warning": "Earnings in 1 week - elevated volatility"
                        }
                    else:
                        result = {"has_upcoming_earnings": False, "risk_level": "LOW", "days_until": days_until}
                    
                    self._earnings_cache[cache_key] = result
                    return result
                    
            except Exception as e:
                logger.debug(f"No earnings data for {ticker}: {e}")
            
            result = {"has_upcoming_earnings": False, "risk_level": "NONE"}
            self._earnings_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Earnings check failed for {ticker}: {e}")
            return {"has_upcoming_earnings": False, "error": str(e)}
    
    # =========================================================================
    # MAIN INTELLIGENCE LAYER
    # =========================================================================
    
    def analyze_signal(self, ticker: str, sentiment: str, confidence: float, portfolio_context: List = None) -> Dict:
        """
        Run all intelligence checks on a potential signal.
        Accepts portfolio_context to avoid repeated DB calls.
        """
        result = {
            "original_sentiment": sentiment,
            "original_confidence": confidence,
            "adjusted_sentiment": sentiment,
            "adjusted_confidence": confidence,
            "checks": {},
            "warnings": [],
            "actions": []
        }
        
        # 1. Portfolio Correlation
        correlation = self.check_portfolio_correlation(ticker, sentiment, portfolio_context)
        result["checks"]["correlation"] = correlation
        if correlation.get("should_downgrade"):
            result["adjusted_sentiment"] = "HOLD"
            result["actions"].append(f"Downgraded to HOLD: {correlation['reason']}")
        elif correlation.get("warning"):
            result["warnings"].append(correlation["warning"])
        
        # 2. DCA Opportunity (for BUY/ACCUMULATE)
        if sentiment in ["BUY", "ACCUMULATE"]:
            dca = self.check_dca_opportunity(ticker)
            result["checks"]["dca"] = dca
            if dca.get("is_excellent_entry"):
                result["adjusted_confidence"] += 0.1
                result["actions"].append(f"Confidence boosted: Excellent DCA entry ({dca['reason']})")
            elif dca.get("is_good_entry"):
                result["adjusted_confidence"] += 0.05
                result["actions"].append(f"Good DCA entry: {dca['reason']}")
            elif dca.get("pullback_pct", 0) > -5:
                result["warnings"].append(f"No pullback from highs - consider waiting for dip")
        
        # 3. Take Profit Check (for owned assets)
        take_profit = self.check_take_profit(ticker, portfolio_context)
        result["checks"]["take_profit"] = take_profit
        if take_profit.get("should_take_profit") and sentiment in ["HOLD", "ACCUMULATE"]:
            result["adjusted_sentiment"] = "SELL"
            result["actions"].append(f"Suggest TRIM {take_profit['suggested_trim']}: {take_profit['reason']}")
        
        # 4. Market Regime
        regime = self.get_market_regime()
        result["checks"]["market_regime"] = regime
        result["adjusted_confidence"] += regime.get("confidence_adjustment", 0)
        if regime.get("regime") == "RISK_OFF":
            result["warnings"].append(f"RISK_OFF regime (VIX={regime.get('vix', '?')}). Extra caution advised.")
            # Don't downgrade to HOLD - just warn (user can decide)
        
        # 5. Earnings Calendar (for stocks)
        earnings = self.check_earnings_risk(ticker)
        result["checks"]["earnings"] = earnings
        if earnings.get("risk_level") == "HIGH" and sentiment == "BUY":
            result["adjusted_sentiment"] = "HOLD"
            result["actions"].append(f"Earnings in {earnings.get('days_until', '?')} days - high volatility risk")
        elif earnings.get("risk_level") == "MEDIUM":
            result["warnings"].append(f"Earnings on {earnings.get('earnings_date')} - elevated volatility")
        
        # Clamp confidence
        result["adjusted_confidence"] = max(0.0, min(1.0, result["adjusted_confidence"]))
        
        return result
    
    def generate_context_for_ai(self, ticker: str, portfolio_context: List = None) -> str:
        """
        Generate a context string to inject into AI prompt for smarter decisions.
        """
        lines = [f"[SIGNAL INTELLIGENCE for {ticker}]"]
        
        # Correlation
        corr = self.check_portfolio_correlation(ticker, "BUY", portfolio_context)
        if corr.get("current_allocation"):
            lines.append(f"- Sector ({corr.get('sector', '?')}): {corr['current_allocation']:.1f}% of portfolio")
            if corr.get("should_downgrade"):
                lines.append(f"  ⚠️ OVERWEIGHT: Do not add more to this sector!")
        
        # DCA
        dca = self.check_dca_opportunity(ticker)
        if dca.get("pullback_pct"):
            lines.append(f"- Pullback: {dca['pullback_pct']:.1f}% from 30d high, RSI: {dca.get('rsi', '?'):.0f}")
            if dca.get("is_good_entry"):
                lines.append("  ✅ GOOD DCA ENTRY: Price has pulled back")
        
        # Take Profit
        tp = self.check_take_profit(ticker, portfolio_context)
        if tp.get("pnl_pct") and tp.get("pnl_pct") > 20:
            lines.append(f"- Current PnL: +{tp['pnl_pct']:.1f}%, Allocation: {tp.get('allocation_pct', 0):.1f}%")
            if tp.get("should_take_profit"):
                lines.append(f"  💰 CONSIDER TRIM: High gains + high allocation")
        
        # Market Regime
        regime = self.get_market_regime()
        lines.append(f"- Market: {regime.get('regime', 'NEUTRAL')} (VIX: {regime.get('vix', '?')}, S&P: {regime.get('sp500_trend', '?')})")
        
        # Earnings
        earn = self.check_earnings_risk(ticker)
        if earn.get("has_upcoming_earnings"):
            lines.append(f"- ⚠️ EARNINGS in {earn.get('days_until', '?')} days ({earn.get('earnings_date', '?')})")
        
        return "\n".join(lines)


