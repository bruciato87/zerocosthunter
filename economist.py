import logging
import datetime
from datetime import timedelta
from typing import Dict, Optional, Tuple
import yfinance as yf
from zoneinfo import ZoneInfo

logger = logging.getLogger("Economist")

class Economist:
    CRYPTO_BASE_TICKERS = {
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", "LINK", "AVAX",
        "SHIB", "MATIC", "LTC", "XLM", "HBAR", "UNI", "RENDER", "RNDR", "TRX",
    }
    EU_TICKER_SUFFIXES = (
        ".DE", ".MI", ".PA", ".AS", ".BR", ".LS", ".MC", ".SW", ".ST",
        ".OL", ".HE", ".CO", ".VI", ".IR", ".F", ".FRA", ".L",
    )

    def __init__(self):
        # Italy timezone
        self.ITALY_TZ = ZoneInfo("Europe/Rome")
        
        # 2026 FOMC Meeting Schedule (Critical Dates)
        self.FED_MEETINGS_2026 = [
            datetime.date(2026, 1, 28),
            datetime.date(2026, 3, 18),
            datetime.date(2026, 4, 29),
            datetime.date(2026, 6, 17),
            datetime.date(2026, 7, 29),
            datetime.date(2026, 9, 16),
            datetime.date(2026, 10, 28),
            datetime.date(2026, 12, 9)
        ]
        
        # US Market Holidays 2026 (NYSE closed)
        self.US_HOLIDAYS_2026 = [
            datetime.date(2026, 1, 1),   # New Year's Day
            datetime.date(2026, 1, 19),  # MLK Day
            datetime.date(2026, 2, 16),  # Presidents Day
            datetime.date(2026, 4, 3),   # Good Friday
            datetime.date(2026, 5, 25),  # Memorial Day
            datetime.date(2026, 7, 3),   # Independence Day (observed)
            datetime.date(2026, 9, 7),   # Labor Day
            datetime.date(2026, 11, 26), # Thanksgiving
            datetime.date(2026, 12, 25), # Christmas
        ]

    def _safe_last_close(self, ticker: str, period: str, context: str):
        """Return last close for ticker or None, logging failures without breaking flow."""
        try:
            hist = yf.Ticker(ticker).history(period=period)
            if hist.empty:
                return None
            return hist["Close"].iloc[-1]
        except Exception as e:
            logger.debug(f"{context} fetch failed for {ticker}: {e}")
            return None
    
    def get_market_status(self):
        """
        Returns current market status for Italy timezone.
        US Markets: 15:30-22:00 CET (9:30-16:00 EST)
        EU Markets: 9:00-17:30 CET
        Crypto: 24/7
        """
        now = datetime.datetime.now(self.ITALY_TZ)
        today = now.date()
        current_hour = now.hour
        current_minute = now.minute
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        status = {
            "crypto": "OPEN (24/7)",
            "us_stocks": "CLOSED",
            "eu_stocks": "CLOSED",
            "is_weekend": weekday >= 5,
            "is_us_holiday": today in self.US_HOLIDAYS_2026,
            "current_time_italy": now.strftime("%H:%M CET")
        }
        
        # Weekend check
        if weekday >= 5:
            status["us_stocks"] = "🔴 CLOSED (Weekend)"
            status["eu_stocks"] = "🔴 CLOSED (Weekend)"
            return status
        
        # US Holiday check
        if today in self.US_HOLIDAYS_2026:
            status["us_stocks"] = "🔴 CLOSED (Holiday)"
        
        # EU Market Hours (9:00-17:30 CET)
        if 9 <= current_hour < 17 or (current_hour == 17 and current_minute < 30):
            status["eu_stocks"] = "🟢 OPEN"
        elif current_hour < 9:
            status["eu_stocks"] = f"🟡 Opens at 9:00 CET"
        else:
            status["eu_stocks"] = "🔴 CLOSED"
        
        # US Market Hours (15:30-22:00 CET)
        if today not in self.US_HOLIDAYS_2026:
            if (current_hour == 15 and current_minute >= 30) or (16 <= current_hour < 22):
                status["us_stocks"] = "🟢 OPEN"
            elif current_hour < 15 or (current_hour == 15 and current_minute < 30):
                status["us_stocks"] = f"🟡 Opens at 15:30 CET"
            else:
                status["us_stocks"] = "🔴 CLOSED"
        
        return status

    def classify_market_for_ticker(
        self,
        ticker: str,
        resolved_ticker: Optional[str] = None,
        is_crypto: Optional[bool] = None,
        currency: Optional[str] = None,
    ) -> str:
        """
        Classify ticker venue bucket for market-hours gating.
        Returns one of: CRYPTO, EU, US.
        """
        def _base(sym: Optional[str]) -> str:
            s = (sym or "").upper().strip()
            for suffix in ("-USD", "-EUR", "-GBP", "-USDT"):
                if s.endswith(suffix):
                    return s[: -len(suffix)]
            return s

        if is_crypto is True:
            return "CRYPTO"

        symbols = [resolved_ticker, ticker]
        bases = {_base(s) for s in symbols if s}
        if any(b in self.CRYPTO_BASE_TICKERS for b in bases):
            return "CRYPTO"

        resolved_u = (resolved_ticker or "").upper().strip()
        ticker_u = (ticker or "").upper().strip()
        if any(resolved_u.endswith(suf) or ticker_u.endswith(suf) for suf in self.EU_TICKER_SUFFIXES):
            return "EU"

        currency_u = (currency or "").upper().strip()
        if currency_u in {"EUR", "CHF", "GBP", "SEK", "NOK", "DKK"}:
            return "EU"

        return "US"

    def get_trading_status_for_ticker(
        self,
        ticker: str,
        market_status: Optional[Dict] = None,
        resolved_ticker: Optional[str] = None,
        is_crypto: Optional[bool] = None,
        currency: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        """
        Return (is_open, market_bucket, market_status_label) for a ticker.
        """
        status = market_status or self.get_market_status()
        market_bucket = self.classify_market_for_ticker(
            ticker=ticker,
            resolved_ticker=resolved_ticker,
            is_crypto=is_crypto,
            currency=currency,
        )

        if market_bucket == "CRYPTO":
            label = str(status.get("crypto", "OPEN (24/7)"))
            return True, market_bucket, label

        if market_bucket == "EU":
            label = str(status.get("eu_stocks", "UNKNOWN"))
            return ("🟢 OPEN" in label), market_bucket, label

        label = str(status.get("us_stocks", "UNKNOWN"))
        return ("🟢 OPEN" in label), market_bucket, label

    def is_market_open_for_ticker(
        self,
        ticker: str,
        market_status: Optional[Dict] = None,
        resolved_ticker: Optional[str] = None,
        is_crypto: Optional[bool] = None,
        currency: Optional[str] = None,
    ) -> bool:
        is_open, _, _ = self.get_trading_status_for_ticker(
            ticker=ticker,
            market_status=market_status,
            resolved_ticker=resolved_ticker,
            is_crypto=is_crypto,
            currency=currency,
        )
        return is_open
        
    def check_risk_level(self):
        """
        Determines current Macro Risk Level (Enhanced Level 10).
        HIGH: Within 24h of FED Meeting OR VIX > 30 OR S&P Death Cross OR DXY Spike.
        MEDIUM: VIX > 20 OR Yield Spike.
        LOW: Normal conditions.
        """
        today = datetime.date.today()
        risk = "LOW"
        reason = []

        # 1. Check FED Calendar (Volatility Trap)
        # Warn if today is the meeting day or the day BEFORE
        for meeting in self.FED_MEETINGS_2026:
            delta = (meeting - today).days
            if 0 <= delta <= 1:
                risk = "HIGH"
                reason.append(f"FED Mtg in {delta}d ({meeting})")
                break
        
        # 2. Check VIX (Fear Index)
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                vix_val = hist['Close'].iloc[-1]
                if vix_val > 30:
                    risk = "HIGH"
                    reason.append(f"VIX Extreme ({vix_val:.1f})")
                elif vix_val > 20 and risk != "HIGH":
                    risk = "MEDIUM"
                    reason.append(f"VIX Elevated ({vix_val:.1f})")
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")

        # 3. Check 10Y Treasury Yield (Trends)
        # Rapid spikes in yields often hurt tech stocks
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if not hist.empty:
                current_yield = hist['Close'].iloc[-1]
                start_yield = hist['Close'].iloc[0]
                change = ((current_yield - start_yield) / start_yield) * 100
                if change > 5.0: # +5% spike in yields in 5 days
                    if risk != "HIGH": risk = "MEDIUM"
                    reason.append(f"Yield Spike (+{change:.1f}%)")
        except Exception as e:
            logger.warning(f"10Y yield trend fetch failed: {e}")

        # 4. S&P 500 Trend (MA50 vs MA200) - NEW Level 10
        try:
            spy = yf.Ticker("^GSPC")  # S&P 500 Index
            hist = spy.history(period="220d")
            if len(hist) >= 200:
                ma50 = hist['Close'].iloc[-50:].mean()
                ma200 = hist['Close'].iloc[-200:].mean()
                current_price = hist['Close'].iloc[-1]
                
                # Death Cross: MA50 < MA200 (Bearish)
                if ma50 < ma200:
                    if risk != "HIGH":
                        risk = "MEDIUM"
                    reason.append(f"S&P Death Cross (MA50<MA200)")
                    
                # Golden Cross: MA50 > MA200 (Bullish) - could reduce risk if all else is OK
                elif ma50 > ma200 * 1.02:  # MA50 > MA200 by at least 2%
                    if risk == "MEDIUM" and len(reason) == 1:
                        # Only downgrade if medium was from yield spike alone
                        pass  # Keep medium as precaution
                    reason.append(f"S&P Golden Cross (Bullish)")
                
                # Check trend direction (7-day momentum)
                week_ago_price = hist['Close'].iloc[-7]
                momentum = ((current_price - week_ago_price) / week_ago_price) * 100
                if momentum < -5:  # -5% weekly drop
                    if risk != "HIGH":
                        risk = "MEDIUM"
                    reason.append(f"S&P Correction ({momentum:.1f}% week)")
                    
        except Exception as e:
            logger.warning(f"S&P 500 trend fetch failed: {e}")

        # 5. DXY (Dollar Index) - NEW Level 10
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="5d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                start = hist['Close'].iloc[0]
                change = ((current - start) / start) * 100
                
                # Strong dollar surge (>2% in 5 days) is bearish for risk assets
                if change > 2.0:
                    if risk != "HIGH":
                        risk = "MEDIUM"
                    reason.append(f"DXY Spike (+{change:.1f}%)")
                elif change < -2.0:
                    # Dollar weakness is bullish for risk assets
                    reason.append(f"DXY Weak ({change:.1f}%) → Risk ON")
        except Exception as e:
            logger.warning(f"DXY fetch failed: {e}")

        return risk, ", ".join(reason)

    def _get_fear_greed_snapshot(self):
        """
        Fetch Fear & Greed with backward compatibility across Insider versions.
        Returns: {"value": int, "label": str} or None.
        """
        try:
            from insider import Insider
            ins = Insider()
            fg_data = None

            if hasattr(ins, "get_fear_greed"):
                fg_data = ins.get_fear_greed()
            elif hasattr(ins, "get_crypto_fear_greed"):
                fg_data = ins.get_crypto_fear_greed()

            if not fg_data:
                return None

            value = fg_data.get("value", 50)
            label = fg_data.get("label") or fg_data.get("classification") or "Neutral"
            try:
                value = int(value)
            except Exception:
                value = 50

            return {"value": value, "label": label}
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return None

    def get_macro_summary(self):
        """
        Returns a string context for the AI Brain.
        Enhanced with DXY and Fear & Greed.
        """
        risk, reason = self.check_risk_level()
        
        # Fetch Data for Context
        vix_context = "N/A"
        yield_context = "N/A"
        dxy_context = "N/A"
        dxy_trend = ""
        fg_context = "N/A"
        
        # VIX
        vix = self._safe_last_close("^VIX", "1d", "VIX")
        if vix is not None:
            vix_context = f"{vix:.2f}"
        
        # 10Y Yield
        tnx = self._safe_last_close("^TNX", "1d", "10Y Yield")
        if tnx is not None:
            yield_context = f"{tnx:.2f}%"
        
        # DXY (Dollar Index) - NEW
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="5d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                start = hist['Close'].iloc[0]
                change = ((current - start) / start) * 100
                dxy_context = f"{current:.2f}"
                if change > 1.0:
                    dxy_trend = "RISING (Bearish for Crypto/Gold)"
                elif change < -1.0:
                    dxy_trend = "FALLING (Bullish for Crypto/Gold)"
                else:
                    dxy_trend = "➡️ STABLE"
        except Exception as e:
            logger.warning(f"DXY fetch failed: {e}")
        
        fg_data = self._get_fear_greed_snapshot()
        if fg_data:
            fg_context = f"{fg_data['value']}/100 ({fg_data['label']})"
        
        # Get Market Status
        market_status = self.get_market_status()
        
        summary = f"""
        [MACRO STRATEGIST CONTEXT]
        
        🕐 {market_status['current_time_italy']} (Italy)
        🇺🇸 US Stocks: {market_status['us_stocks']}
        🇪🇺 EU Stocks: {market_status['eu_stocks']}
        ₿ Crypto: {market_status['crypto']}
        
        📊 Risk Level: {risk}
        📋 Reason: {reason if reason else "Normal Market Conditions"}
        
        VIX (Fear): {vix_context}
        10Y Yield: {yield_context}
        DXY (Dollar): {dxy_context} {dxy_trend}
        Fear & Greed: {fg_context}
        
        FED 2026 Next Mtg: {self._get_next_meeting()}
        
        STRATEGY: {'CAUTION: Do NOT Buy volatile assets.' if risk == 'HIGH' else 'MODERATE: Be selective.' if risk == 'MEDIUM' else 'GREEN LIGHT: Macro environment stable.'}
        
        **MARKET RULES:**
        - If US/EU CLOSED: Only Crypto signals are actionable now
        - If Weekend: Stock signals are for next week's review
        - If DXY RISING: Avoid Crypto/Gold, favor USD assets
        - If Fear & Greed < 25: "Buy the Dip" opportunity
        """
        return summary

    def get_dashboard_stats(self):
        """
        Returns structured data for the Web Dashboard.
        Enhanced with DXY and Fear & Greed.
        """
        risk, reason = self.check_risk_level()
        stats = {
            "risk_level": risk,
            "risk_reason": reason if reason else "Normal Market Conditions",
            "vix": "N/A",
            "tnx_yield": "N/A",
            "dxy": "N/A",
            "dxy_trend": "STABLE",
            "fear_greed": "N/A",
            "next_meeting": self._get_next_meeting()
        }
        
        vix = self._safe_last_close("^VIX", "1d", "Dashboard VIX")
        if vix is not None:
            stats["vix"] = round(vix, 2)
        
        tnx = self._safe_last_close("^TNX", "1d", "Dashboard 10Y Yield")
        if tnx is not None:
            stats["tnx_yield"] = f"{tnx:.2f}%"
        
        # DXY
        try:
            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="5d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                start = hist['Close'].iloc[0]
                change = ((current - start) / start) * 100
                stats["dxy"] = round(current, 2)
                if change > 1.0:
                    stats["dxy_trend"] = "RISING"
                elif change < -1.0:
                    stats["dxy_trend"] = "FALLING"
        except Exception as e:
            logger.debug(f"Dashboard DXY fetch failed: {e}")
        
        fg_data = self._get_fear_greed_snapshot()
        if fg_data:
            stats["fear_greed"] = fg_data["value"]
        
        return stats

    def _get_next_meeting(self):
        today = datetime.date.today()
        for m in self.FED_MEETINGS_2026:
            if m >= today:
                return m.strftime("%Y-%m-%d")
        return "No more meetings in 2026"

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    eco = Economist()
    print(eco.get_macro_summary())
