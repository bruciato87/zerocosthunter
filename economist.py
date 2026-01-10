import logging
import datetime
from datetime import timedelta
import yfinance as yf

logger = logging.getLogger("Economist")

class Economist:
    def __init__(self):
        # 2026 FOMC Meeting Schedule (Critical Dates)
        # Dates are the LAST DAY of the meeting (when Rate Decision is announced)
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
        
    def check_risk_level(self):
        """
        Determines current Macro Risk Level.
        HIGH: Within 24h of FED Meeting OR VIX > 30.
        MEDIUM: VIX > 20.
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
        except: pass

        return risk, ", ".join(reason)

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
        try:
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            vix_context = f"{vix:.2f}"
        except: pass
        
        # 10Y Yield
        try:
            tnx = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]
            yield_context = f"{tnx:.2f}%"
        except: pass
        
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
                    dxy_trend = "📈 RISING (Bearish for Crypto/Gold)"
                elif change < -1.0:
                    dxy_trend = "📉 FALLING (Bullish for Crypto/Gold)"
                else:
                    dxy_trend = "➡️ STABLE"
        except Exception as e:
            logger.warning(f"DXY fetch failed: {e}")
        
        # Fear & Greed from Insider (if available)
        try:
            from insider import Insider
            ins = Insider()
            fg_data = ins.get_fear_greed()
            if fg_data:
                fg_val = fg_data.get('value', 50)
                fg_label = fg_data.get('label', 'Neutral')
                fg_context = f"{fg_val}/100 ({fg_label})"
        except: 
            fg_context = "N/A"
        
        summary = f"""
        [MACRO STRATEGIST CONTEXT]
        
        📊 Risk Level: {risk}
        📋 Reason: {reason if reason else "Normal Market Conditions"}
        
        🔥 VIX (Fear): {vix_context}
        💵 10Y Yield: {yield_context}
        💲 DXY (Dollar): {dxy_context} {dxy_trend}
        😱 Fear & Greed: {fg_context}
        
        📅 FED 2026 Next Mtg: {self._get_next_meeting()}
        
        🎯 STRATEGY: {'⚠️ CAUTION: Do NOT Buy volatile assets.' if risk == 'HIGH' else '🟡 MODERATE: Be selective.' if risk == 'MEDIUM' else '✅ GREEN LIGHT: Macro environment stable.'}
        
        **MACRO RULES:**
        - If DXY RISING: Avoid Crypto/Gold, favor USD assets
        - If DXY FALLING: Crypto/Gold may outperform
        - If Fear & Greed < 25: "Buy the Dip" opportunity
        - If Fear & Greed > 75: Take profits, avoid FOMO
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
        
        try:
            vix = yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
            stats["vix"] = round(vix, 2)
        except: pass
        
        try:
            tnx = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]
            stats["tnx_yield"] = f"{tnx:.2f}%"
        except: pass
        
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
        except: pass
        
        # Fear & Greed
        try:
            from insider import Insider
            ins = Insider()
            fg_data = ins.get_fear_greed()
            if fg_data:
                stats["fear_greed"] = fg_data.get('value', 50)
        except: pass
        
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
