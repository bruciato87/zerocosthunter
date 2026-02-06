"""
Position Watchdog - Intelligent Portfolio Position Monitoring & Exit Signals
=============================================================================
AI-Driven exit signal generation using:
1. ATR-based dynamic thresholds (volatility-adjusted)
2. ML predictions for trend exhaustion  
3. Market Regime awareness (Bull/Bear/Sideways)
4. Technical indicators (RSI, MACD, Momentum)
5. Tax-aware profit calculations (Italian 26% capital gains)
6. Trade Republic fee consideration (€1 per trade)

NO STATIC THRESHOLDS - All decisions are context-aware and dynamic.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("PositionWatchdog")

# Italian Tax Law Constants
CAPITAL_GAINS_TAX = 0.26  # 26% on profits
TRADE_REPUBLIC_FEE = 1.0  # €1 per trade

# Conservative Mode Settings
MIN_WORTHWHILE_NET_PROFIT = 10.0  # €10 minimum net profit to suggest exit
MIN_CONFIDENCE = 0.75  # 75% confidence threshold
MIN_TECH_SCORE_FOR_EXIT = -30  # Technical score must be bearish
REQUIRE_MULTI_CONFIRM = True  # Require both ML and Tech to confirm


@dataclass
class ExitSignal:
    """Represents an exit signal for a position."""
    ticker: str
    action: str  # "SELL", "TRIM", "HOLD"
    reason: str
    urgency: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    current_price: float
    entry_price: float
    quantity: float
    pnl_percent: float
    gross_profit: float
    tax_amount: float
    net_profit: float
    dynamic_stop_loss: float
    confidence: float
    technical_score: int  # -100 to +100 (bearish to bullish)
    suggested_action: str
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None


class PositionWatchdog:
    """
    Intelligent Position Monitor with Dynamic Exit Logic.
    
    Uses ML + Volatility + Regime + Technicals to determine optimal exit points.
    Considers taxes and fees to ensure profitable exits.
    """
    
    CRYPTO_TICKERS = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'DOT', 'LINK', 
                      'AVAX', 'SHIB', 'MATIC', 'LTC', 'XLM', 'HBAR', 'UNI'}
    
    def __init__(self, db_handler=None, market_data=None, ml_predictor=None, regime_classifier=None):
        self.db = db_handler
        self.market = market_data
        self.ml = ml_predictor
        self.regime = regime_classifier
        self._high_water_marks = {}
        self.settings = {}
        
    def _lazy_load(self):
        """Lazy load dependencies."""
        if self.db is None:
            from db_handler import DBHandler
            self.db = DBHandler()
        
        # Load settings
        if not self.settings:
            try:
                self.settings = self.db.get_settings()
            except:
                self.settings = {"min_confidence": 0.75, "risk_profile": "BALANCED"}

        if self.market is None:
            from market_data import MarketData
            self.market = MarketData()
        if self.ml is None:
            from ml_predictor import MLPredictor
            self.ml = MLPredictor()
        if self.regime is None:
            from market_regime import MarketRegimeClassifier
            self.regime = MarketRegimeClassifier()
        if not hasattr(self, 'hunter') or self.hunter is None:
            from hunter import NewsHunter
            self.hunter = NewsHunter()
    
    def _is_crypto(self, ticker: str) -> bool:
        base = ticker.upper().replace('-USD', '').replace('-EUR', '')
        return base in self.CRYPTO_TICKERS
    
    def _calculate_tax_and_fees(self, quantity: float, entry_price: float, current_price: float) -> Dict:
        """
        Calculate net profit after Italian taxes and Trade Republic fees.
        """
        gross_profit = (current_price - entry_price) * quantity
        
        # Only pay tax on profits
        if gross_profit > 0:
            tax = gross_profit * CAPITAL_GAINS_TAX
        else:
            tax = 0
        
        # Trade Republic charges €1 per trade
        fees = TRADE_REPUBLIC_FEE
        
        net_profit = gross_profit - tax - fees
        
        return {
            'gross_profit': round(gross_profit, 2),
            'tax_amount': round(tax, 2),
            'fees': fees,
            'net_profit': round(net_profit, 2),
            'is_worthwhile': net_profit >= MIN_WORTHWHILE_NET_PROFIT
        }
    
    def _get_technical_score(self, ticker: str) -> Tuple[int, str]:
        """
        Get a technical score from -100 (very bearish) to +100 (very bullish).
        Based on RSI, momentum, and price trends.
        """
        try:
            tech_data = self.market.get_technical_summary(ticker, return_dict=True)
            if not tech_data or not isinstance(tech_data, dict):
                return (0, "Dati tecnici non disponibili")
            
            score = 0
            reasons = []
            
            # RSI Analysis
            rsi = tech_data.get('rsi', 50)
            if rsi > 70:
                score -= 30
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < 30:
                score += 30
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 60:
                score -= 10
            elif rsi < 40:
                score += 10
            
            # Momentum
            momentum = tech_data.get('momentum', 0)
            if momentum < -5:
                score -= 25
                reasons.append(f"Momentum negativo ({momentum:.1f}%)")
            elif momentum > 5:
                score += 25
                reasons.append(f"Momentum positivo ({momentum:.1f}%)")
            
            # Price vs SMA
            price = tech_data.get('price', 0)
            sma20 = tech_data.get('sma20', price)
            if price > 0 and sma20 > 0:
                if price < sma20 * 0.95:
                    score -= 20
                    reasons.append("Sotto SMA20")
                elif price > sma20 * 1.05:
                    score += 20
                    reasons.append("Sopra SMA20")
            
            # Trend direction (short-term)
            trend = tech_data.get('trend', 'NEUTRAL')
            if trend == 'DOWNTREND':
                score -= 25
                reasons.append("Trend ribassista")
            elif trend == 'UPTREND':
                score += 25
                reasons.append("Trend rialzista")
            
            # Clamp to -100, +100
            score = max(-100, min(100, score))
            
            return (score, ", ".join(reasons) if reasons else "Neutrale")
            
        except Exception as e:
            logger.warning(f"Technical score failed for {ticker}: {e}")
            return (0, "Errore analisi tecnica")
    
    
    def _is_economical(self, ticker: str, amount_eur: float, tax_info: Dict) -> Tuple[bool, str]:
        """Check if a trade is economically viable considering fees and taxes."""
        fee = 1.0  # Trade Republic fixed fee
        total_cost = fee + tax_info.get('tax_amount', 0)
        
        # Rule 1: Minimum amount check to keep fee < 2%
        if amount_eur < 50:
             return False, f"Controvalore troppo basso (€{amount_eur:.2f}). Fee incide troppo."
             
        # Rule 2: Cost-to-Proceed ratio. Costs should not eat > 30% of the proceeds.
        # This is more relevant for sells in profit.
        if amount_eur > 0:
            cost_ratio = total_cost / amount_eur
            if cost_ratio > 0.3:
                return False, f"Costi (€{total_cost:.2f}) incidono {cost_ratio:.0%} sul ricavato."
        
        return True, ""

    def _calculate_dynamic_thresholds(self, ticker: str, entry_price: float) -> Dict:
        """Calculate dynamic exit thresholds based on ATR + Regime."""
        try:
            atr_data = self.market.calculate_atr(ticker)
            atr = atr_data.get('atr', 0)
            volatility = atr_data.get('volatility', 'MEDIUM')
            
            # Base multipliers adjusted by Risk Profile
            profile = self.settings.get('risk_profile', 'BALANCED').upper()
            
            if profile == 'CONSERVATIVE':
                # Tighter stops, lower targets
                if volatility == 'HIGH': sl_mult, tp_mult = 1.5, 3.0
                elif volatility == 'LOW': sl_mult, tp_mult = 1.0, 2.0
                else: sl_mult, tp_mult = 1.2, 2.5
            elif profile == 'AGGRESSIVE':
                # Wider stops, higher targets (Diamond Hands)
                if volatility == 'HIGH': sl_mult, tp_mult = 3.5, 6.0
                elif volatility == 'LOW': sl_mult, tp_mult = 2.0, 4.0
                else: sl_mult, tp_mult = 3.0, 5.0
            else: # BALANCED (Default)
                if volatility == 'HIGH': sl_mult, tp_mult = 2.5, 4.0
                elif volatility == 'LOW': sl_mult, tp_mult = 1.5, 2.5
                else: sl_mult, tp_mult = 2.0, 3.0
            
            # Regime adjustment
            regime_data = self.regime.classify() if self.regime else {'regime': 'NEUTRAL'}
            regime = regime_data.get('regime', 'NEUTRAL')
            
            if regime == 'BEAR':
                sl_mult *= 0.8
                tp_mult *= 0.7
            elif regime == 'BULL':
                sl_mult *= 1.2
                tp_mult *= 1.5
            
            # Crypto adjustment
            if self._is_crypto(ticker):
                sl_mult *= 1.3
                tp_mult *= 1.4
            
            return {
                'stop_loss_price': entry_price - (atr * sl_mult),
                'take_profit_price': entry_price + (atr * tp_mult),
                'atr': atr,
                'volatility': volatility,
                'regime': regime
            }
        except Exception as e:
            logger.warning(f"Dynamic threshold calc failed: {e}")
            is_crypto = self._is_crypto(ticker)
            return {
                'stop_loss_price': entry_price * (0.85 if is_crypto else 0.90),
                'take_profit_price': entry_price * (1.20 if is_crypto else 1.15),
                'atr': 0,
                'volatility': 'UNKNOWN',
                'regime': 'NEUTRAL'
            }
    
    def _get_ml_exit_signal(self, ticker: str, pnl_pct: float) -> Tuple[str, float, str]:
        """Get ML recommendation: EXIT, HOLD, ACCUMULATE."""
        try:
            pred = self.ml.predict(ticker)
            if not pred:
                return ("HOLD", 0.5, "ML non disponibile")
            
            direction = pred.direction
            conf = pred.confidence
            
            if pnl_pct > 0:
                if direction == 'DOWN' and conf > 0.6:
                    return ("EXIT", conf, f"ML prevede ribasso ({conf:.0%})")
                elif direction == 'UP' and conf > 0.7:
                    return ("HOLD", conf, f"ML ancora bullish ({conf:.0%})")
            else:
                if direction == 'DOWN' and conf > 0.6:
                    return ("EXIT", conf, f"ML conferma trend negativo")
                elif direction == 'UP' and conf > 0.6:
                    return ("HOLD", conf, f"ML vede recupero ({conf:.0%})")
            
            return ("HOLD", 0.5, "ML neutrale")
        except:
            return ("HOLD", 0.5, "ML error")
    
    async def scan_portfolio(self) -> List[ExitSignal]:
        """Intelligent scan of all positions."""
        self._lazy_load()
        signals = []
        
        try:
            portfolio = self.db.get_portfolio()
            if not portfolio:
                logger.info("PositionWatchdog: Portfolio vuoto.")
                return signals
            
            logger.info(f"PositionWatchdog: Scanning {len(portfolio)} posizioni...")
            
            for pos in portfolio:
                try:
                    signal = await self._analyze_position(pos)
                    if signal and signal.action != "HOLD":
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error analyzing {pos.get('ticker')}: {e}")
            
            # Sort by urgency
            urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            signals.sort(key=lambda s: (urgency_order.get(s.urgency, 4), -s.confidence))
            
            return signals
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return signals
    
    def _analyze_position_sync(self, pos: Dict) -> Optional[ExitSignal]:
        """Synchronous wrapper for _analyze_position."""
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # If we are in an existing loop, we can't use run_until_complete
                    # We'll use a nested loop approach or similar if nest_asyncio is present
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(self._analyze_position(pos))
            except RuntimeError:
                # No running loop, create a new one
                return asyncio.run(self._analyze_position(pos))
        except Exception as e:
            logger.error(f"Sync analysis failed for {pos.get('ticker')}: {e}")
            return None

    async def _analyze_position(self, pos: Dict) -> Optional[ExitSignal]:
        """Analyze single position with all factors."""
        self._lazy_load()
        ticker = pos.get('ticker', '')
        quantity = float(pos.get('quantity', 0))
        entry_price = float(pos.get('avg_price', 0))
        
        if quantity <= 0 or entry_price <= 0:
            return None
        
        # [PHASE 3] Get current targets from DB record
        target_price = pos.get('target_price')
        stop_loss_price = pos.get('stop_loss_price')
        target_type = pos.get('target_type', 'AUTO')
        
        # Current price
        current_price, _ = self.market.get_smart_price_eur(ticker)
        if current_price <= 0:
            return None
        
        # P&L
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Tax/Fee calculation
        tax_info = self._calculate_tax_and_fees(quantity, entry_price, current_price)
        
        # Dynamic thresholds
        thresholds = self._calculate_dynamic_thresholds(ticker, entry_price)
        
        # Technical score (-100 to +100)
        tech_score, tech_reason = self._get_technical_score(ticker)
        
        # ML signal
        ml_action, ml_conf, ml_reason = self._get_ml_exit_signal(ticker, pnl_pct)
        
        # [PHASE 2] News Sentiment Check
        news_data = self.hunter.check_owned_asset_news(ticker)
        has_bad_news = news_data.get('is_negative', False)
        news_summary = news_data.get('summary', '')
        
        # High water mark for trailing
        hwm = self._high_water_marks.get(ticker, current_price)
        if current_price > hwm:
            self._high_water_marks[ticker] = current_price
            hwm = current_price
            
        # --- [PHASE 3] AUTOMATIC TARGET SETTING ---
        if not target_price or not stop_loss_price:
            logger.info(f"Phase 3: Setting automatic targets for {ticker}...")
            # Use dynamic thresholds as initial targets
            auto_tp = round(thresholds['take_profit_price'], 2)
            auto_sl = round(thresholds['stop_loss_price'], 2)
            
            # Save to DB
            self.db.update_portfolio_targets(
                ticker=ticker, 
                target_price=auto_tp, 
                stop_loss_price=auto_sl,
                target_type='AUTO'
            )
            
            # Update local variables for this analysis
            target_price = auto_tp
            stop_loss_price = auto_sl
            
            # Add a "TARGET SET" signal for the report
            # (We reuse ExitSignal but with a special action or just prefix reason)
            # Actually, better to just let it run and maybe add a flag or note to the signal.
            target_set_note = f"🎯 TARGET AUTO SET: Profit @ €{auto_tp}, Stop @ €{auto_sl}\n"
        else:
            target_set_note = ""
        
        # --- EXIT DECISION LOGIC ---
        
        # 0. CORE ASSET PROTECTION (Phase 9)
        is_core = pos.get('is_core', False)
        if is_core and pnl_pct > -15:
             # Skip technical SELL/TRIM for core assets unless it's a critical crash
             return None

        # 1. STOP LOSS (Trigger unless already in deep capitulation or oversold)
        if current_price <= thresholds['stop_loss_price']:
            # [STRATEGY] If already deeply underwater (>30%) and RSI is oversold (<35), 
            # selling might be a mistake (selling the bottom). Suggest HOLD instead.
            is_oversold = False
            try:
                tech_data = self.market.get_technical_summary(ticker, return_dict=True)
                if isinstance(tech_data, dict) and tech_data.get('rsi', 50) < 35:
                    is_oversold = True
            except:
                pass

            if pnl_pct < -15 and is_oversold:
                # [ADAPTIVE] Incorporate Correlation Context
                corr_reason = ""
                try:
                    from correlation_engine import CorrelationEngine
                    ce = CorrelationEngine()
                    correlated = ce.get_correlated_assets(ticker)
                    if correlated:
                        # Check if high correlation with BTC/ETH during a drop
                        corr_reason = " | 🔗 Rischio Sistemico (Asset correlati)"
                except:
                    pass

                return ExitSignal(
                    ticker=ticker, action="HOLD",
                    reason=f"⚠️ OVERSOLD PROTECTION: {pnl_pct:+.1f}% | 📉 Ipervenduto (RSI < 35){corr_reason}",
                    urgency="MEDIUM", current_price=current_price, entry_price=entry_price,
                    quantity=quantity, pnl_percent=pnl_pct,
                    gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                    net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                    confidence=0.8, technical_score=tech_score,
                    suggested_action=f"Evita vendita compulsiva su ipervenduto. Attendi rimbalzo tecnico prima di liquidare {ticker}."
                )

            # [REFINEMENT] MILD STOP LOSS (De-risking)
            # If loss is small (< 5%) and ML is not bearish, suggest TRIM 50%
            if pnl_pct > -5 and ml_action != "EXIT":
                current_value = current_price * quantity
                trim_value = current_value * 0.5
                is_ok, _ = self._is_economical(ticker, trim_value, tax_info)
                if is_ok:
                    return ExitSignal(
                        ticker=ticker, action="TRIM",
                        reason=f"🛡️ DE-RISKING: {pnl_pct:+.1f}% | Soglia ATR toccata | ML: {ml_action}",
                        urgency="MEDIUM", current_price=current_price, entry_price=entry_price,
                        quantity=quantity, pnl_percent=pnl_pct,
                        gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                        net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                        confidence=0.7, technical_score=tech_score,
                        suggested_action=f"Vendi 50% per ridurre esposizione su {ticker}."
                    )

            # Standard Stop Loss (Hard Exit)
            # Check if it's economical even for a stop loss, unless it's CRITICAL
            current_value = current_price * quantity
            is_ok, econ_reason = self._is_economical(ticker, current_value, tax_info)
            if not is_ok and pnl_pct > -20: # If loss is not too bad and it's not economical, HOLD
                 return ExitSignal(
                    ticker=ticker, action="HOLD",
                    reason=f"🛑 STOP LOSS DELAYED: {pnl_pct:+.1f}% | {econ_reason}",
                    urgency="LOW", current_price=current_price, entry_price=entry_price,
                    quantity=quantity, pnl_percent=pnl_pct,
                    gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                    net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                    confidence=0.8, technical_score=tech_score,
                    suggested_action=f"Controvalore troppo basso per stop-loss economico su {ticker}."
                )

            return ExitSignal(
                ticker=ticker, action="SELL",
                reason=f"🛑 STOP LOSS: {pnl_pct:+.1f}% | Tecnici: {tech_reason}" + (f" | ⚠️ News: {news_summary}" if has_bad_news else ""),
                urgency="CRITICAL", current_price=current_price, entry_price=entry_price,
                quantity=quantity, pnl_percent=pnl_pct,
                gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                confidence=0.9, technical_score=tech_score,
                suggested_action=f"Vendi {ticker} per limitare ulteriori perdite."
            )
        
        # 2. TAKE PROFIT / SCALING OUT
        target_hit_pct = 0
        if target_price and entry_price < target_price:
            target_hit_pct = (current_price - entry_price) / (target_price - entry_price)
        
        current_value = current_price * quantity
        
        if current_price >= thresholds['take_profit_price'] or target_hit_pct >= 0.8:
            # Full Exit condition
            if current_price >= thresholds['take_profit_price'] and (ml_action == "EXIT" or tech_score < -20):
                # [FIX] Check if economical
                is_ok, econ_reason = self._is_economical(ticker, current_value, tax_info)
                if not is_ok or tax_info['net_profit'] < 0:
                     return ExitSignal(
                        ticker=ticker, action="HOLD",
                        reason=f"💰 TAKE PROFIT DELAYED: +{pnl_pct:.1f}% | " + (econ_reason if not is_ok else "Net profit negativo"),
                        urgency="LOW", current_price=current_price, entry_price=entry_price,
                        quantity=quantity, pnl_percent=pnl_pct,
                        gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                        net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                        confidence=ml_conf, technical_score=tech_score,
                        suggested_action=f"Controvalore o profitto troppo basso per chiusura su {ticker}."
                    )

                return ExitSignal(
                    ticker=ticker, action="SELL",
                    reason=f"💰 TAKE PROFIT: +{pnl_pct:.1f}% | Netto: €{tax_info['net_profit']:.2f} | {ml_reason}",
                    urgency="HIGH", current_price=current_price, entry_price=entry_price,
                    quantity=quantity, pnl_percent=pnl_pct,
                    gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                    net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                    confidence=ml_conf, technical_score=tech_score,
                    suggested_action=f"Prendi profitto totale. Tasse: €{tax_info['tax_amount']:.2f}, Netto: €{tax_info['net_profit']:.2f}"
                )
            
            # [REFINEMENT] SCALING OUT (Partial Take Profit)
            # If hit 100% of target but ML is still positive
            if target_hit_pct >= 1.0:
                trim_value = current_value * 0.5
                is_ok, _ = self._is_economical(ticker, trim_value, tax_info)
                if is_ok:
                    return ExitSignal(
                        ticker=ticker, action="TRIM",
                        reason=f"🎯 TARGET HIT: +{pnl_pct:.1f}% | ML ancora bullish | Scaling out",
                        urgency="MEDIUM", current_price=current_price, entry_price=entry_price,
                        quantity=quantity, pnl_percent=pnl_pct,
                        gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                        net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                        confidence=0.8, technical_score=tech_score,
                        suggested_action=f"Vendi 50% per bloccare profitto e lascia correre il resto."
                    )
            # If hit 80% of target distance
            elif target_hit_pct >= 0.8:
                trim_value = current_value * 0.25
                is_ok, _ = self._is_economical(ticker, trim_value, tax_info)
                if is_ok:
                    return ExitSignal(
                        ticker=ticker, action="TRIM",
                        reason=f"🏁 NEAR TARGET: +{pnl_pct:.1f}% | 80% del percorso fatto | De-risking",
                        urgency="LOW", current_price=current_price, entry_price=entry_price,
                        quantity=quantity, pnl_percent=pnl_pct,
                        gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                        net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                        confidence=0.7, technical_score=tech_score,
                        suggested_action=f"Vendi 25% per mettere in sicurezza una parte del profitto."
                    )

        # 3. TECHNICAL BEARISH (Weakness Trim)
        if tech_score <= -30 and pnl_pct > 2:
            trim_value = current_value * 0.25
            is_ok, _ = self._is_economical(ticker, trim_value, tax_info)
            if is_ok:
                return ExitSignal(
                    ticker=ticker, action="TRIM",
                    reason=f"📉 DEBOLEZZA TECNICA: Score {tech_score} | {tech_reason}",
                    urgency="MEDIUM", current_price=current_price, entry_price=entry_price,
                    quantity=quantity, pnl_percent=pnl_pct,
                    gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                    net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                    confidence=0.6, technical_score=tech_score,
                    suggested_action=f"Vendi 25% su debolezza tecnica."
                )
        
        # 4. ML STRONG EXIT or BAD NEWS
        if (ml_action == "EXIT" and ml_conf > 0.75) or (has_bad_news and pnl_pct > 0):
            # [FIX] Check if economical
            current_value = current_price * quantity
            is_ok, econ_reason = self._is_economical(ticker, current_value, tax_info)
            if not is_ok or (pnl_pct > 0 and tax_info['net_profit'] < 1): # If in profit, expect at least €1 net
                 return None # Skip this signal if not worthwhile

            reason_str = f"🤖 ML EXIT: {ml_reason}" if ml_action == "EXIT" else f"📉 NEWS EXIT: {news_summary}"
            return ExitSignal(
                ticker=ticker, action="SELL",
                reason=f"{reason_str} | Tech: {tech_score:+d}",
                urgency="MEDIUM", current_price=current_price, entry_price=entry_price,
                quantity=quantity, pnl_percent=pnl_pct,
                gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                confidence=ml_conf, technical_score=tech_score,
                suggested_action=f"ML raccomanda exit. Netto: €{tax_info['net_profit']:.2f}",
                target_price=target_price, stop_loss_price=stop_loss_price
            )
        
        # 5. NEW: Report automatic target setting even if no exit is triggered
        if target_set_note:
            return ExitSignal(
                ticker=ticker, action="HOLD",
                reason=target_set_note,
                urgency="LOW", current_price=current_price, entry_price=entry_price,
                quantity=quantity, pnl_percent=pnl_pct,
                gross_profit=tax_info['gross_profit'], tax_amount=tax_info['tax_amount'],
                net_profit=tax_info['net_profit'], dynamic_stop_loss=thresholds['stop_loss_price'],
                confidence=1.0, technical_score=tech_score,
                suggested_action="Target impostati automaticamente.",
                target_price=target_price, stop_loss_price=stop_loss_price
            )
        
        return None
    
    def format_telegram_report(self, signals: List[ExitSignal]) -> str:
        """Format report for Telegram."""
        if not signals:
            return "✅ **Position Watchdog AI**\nTutte le posizioni monitorate. Nessun exit signal.\n_Analisi: ATR + ML + Tecnici + Tasse_"
        
        lines = ["🔔 **POSITION WATCHDOG AI**\n_Exit signals con calcolo netto tasse (26%) e fee (€1)_\n"]
        
        for s in signals:
            emoji = "🔴" if s.urgency in ["CRITICAL", "HIGH"] else ("🟡" if s.urgency == "MEDIUM" else "ℹ️")
            lines.append(f"{emoji} **{s.ticker}** → {s.action}")
            
            if s.target_price and s.stop_loss_price:
                lines.append(f"   🎯 Target: Profit €{s.target_price} | Stop €{s.stop_loss_price}")
            
            if s.action != "HOLD":
                lines.append(f"   P&L: {s.pnl_percent:+.1f}% | Lordo: €{s.gross_profit:.2f}")
                lines.append(f"   Tasse: €{s.tax_amount:.2f} | **Netto: €{s.net_profit:.2f}**")
            
            lines.append(f"   {s.reason}")
            lines.append(f"   👉 _{s.suggested_action}_\n")
        
        return "\n".join(lines)
    
    async def run_daily(self, target_chat_id: str = None):
        """Daily job."""
        from telegram_bot import TelegramNotifier
        self._lazy_load()
        
        chat_id = target_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not chat_id:
            logger.error("No TELEGRAM_CHAT_ID")
            return
        
        logger.info("PositionWatchdog: Starting scan...")
        signals = await self.scan_portfolio()
        
        notifier = TelegramNotifier()
        await notifier.send_message(chat_id=chat_id, message=self.format_telegram_report(signals))
        
        logger.info(f"PositionWatchdog: {len(signals)} signals sent.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    watchdog = PositionWatchdog()
    asyncio.run(watchdog.run_daily())
