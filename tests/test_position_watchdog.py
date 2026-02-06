"""
Tests for Position Watchdog module - Conservative Mode.
"""

import pytest
from unittest.mock import MagicMock, patch
from position_watchdog import PositionWatchdog, ExitSignal, MIN_WORTHWHILE_NET_PROFIT, MIN_CONFIDENCE


class TestTaxCalculations:
    """Test Italian tax and Trade Republic fee calculations."""
    
    def setup_method(self):
        self.watchdog = PositionWatchdog(
            db_handler=MagicMock(),
            market_data=MagicMock(),
            ml_predictor=MagicMock(),
            regime_classifier=MagicMock()
        )
        self.watchdog.hunter = MagicMock()
    
    def test_tax_on_profit(self):
        """26% tax on profits."""
        result = self.watchdog._calculate_tax_and_fees(
            quantity=10, entry_price=100, current_price=120
        )
        
        # Gross: (120-100)*10 = €200
        # Tax: 200 * 0.26 = €52
        # Fee: €1
        # Net: 200 - 52 - 1 = €147
        assert result['gross_profit'] == 200.0
        assert result['tax_amount'] == 52.0
        assert result['net_profit'] == 147.0
        assert result['is_worthwhile'] == True  # > €10
    
    def test_no_tax_on_loss(self):
        """No tax when losing money."""
        result = self.watchdog._calculate_tax_and_fees(
            quantity=10, entry_price=100, current_price=90
        )
        
        # Gross: (90-100)*10 = -€100
        # Tax: 0 (no tax on losses)
        # Fee: €1
        # Net: -100 - 0 - 1 = -€101
        assert result['gross_profit'] == -100.0
        assert result['tax_amount'] == 0.0
        assert result['net_profit'] == -101.0
        assert result['is_worthwhile'] == False
    
    def test_small_profit_not_worthwhile(self):
        """Profits under €10 net are not worthwhile."""
        result = self.watchdog._calculate_tax_and_fees(
            quantity=1, entry_price=100, current_price=110
        )
        
        # Gross: €10
        # Tax: 10 * 0.26 = €2.60
        # Fee: €1
        # Net: 10 - 2.60 - 1 = €6.40
        assert result['net_profit'] == 6.4
        assert result['is_worthwhile'] == False  # < €10 min threshold


class TestConservativeMode:
    """Test conservative exit logic."""
    
    def setup_method(self):
        self.mock_db = MagicMock()
        self.mock_market = MagicMock()
        self.mock_ml = MagicMock()
        self.mock_regime = MagicMock()
        
        self.watchdog = PositionWatchdog(
            db_handler=self.mock_db,
            market_data=self.mock_market,
            ml_predictor=self.mock_ml,
            regime_classifier=self.mock_regime
        )
        
        # Default regime
        self.mock_regime.classify.return_value = {'regime': 'NEUTRAL'}
        
        self.mock_hunter = MagicMock()
        self.watchdog.hunter = self.mock_hunter
        self.mock_hunter.check_owned_asset_news.return_value = {'is_negative': False}
    
    @pytest.mark.asyncio
    async def test_stop_loss_always_triggers(self):
        """Stop loss should trigger even with losses."""
        position = {
            'ticker': 'AAPL',
            'quantity': 10,
            'avg_price': 100.0
        }
        
        # Price dropped below dynamic stop loss
        self.mock_market.get_smart_price_eur.return_value = (85.0, 'AAPL')
        self.mock_market.calculate_atr.return_value = {'atr': 5.0, 'volatility': 'MEDIUM'}
        self.mock_market.get_technical_summary.return_value = {'rsi': 25, 'momentum': -8}
        
        ml_pred = MagicMock()
        ml_pred.direction = 'DOWN'
        ml_pred.confidence = 0.8
        self.mock_ml.predict.return_value = ml_pred
        
        signal = await self.watchdog._analyze_position(position)
        
        assert signal is not None
        assert signal.action == "SELL"
        assert signal.urgency == "CRITICAL"
        assert "STOP LOSS" in signal.reason
    
    @pytest.mark.asyncio
    async def test_take_profit_requires_confirmation(self):
        """Take profit needs ML + Tech confirmation in conservative mode."""
        position = {
            'ticker': 'AAPL',
            'quantity': 100,
            'avg_price': 100.0
        }
        
        # Good profit
        self.mock_market.get_smart_price_eur.return_value = (130.0, 'AAPL')
        self.mock_market.calculate_atr.return_value = {'atr': 5.0, 'volatility': 'MEDIUM'}
        
        # Bearish technicals
        self.mock_market.get_technical_summary.return_value = {
            'rsi': 75, 'momentum': -3, 'trend': 'DOWNTREND'
        }
        
        # ML confirms exit
        ml_pred = MagicMock()
        ml_pred.direction = 'DOWN'
        ml_pred.confidence = 0.8
        self.mock_ml.predict.return_value = ml_pred
        
        signal = await self.watchdog._analyze_position(position)
        
        assert signal is not None
        assert signal.action == "SELL"
        assert "TAKE PROFIT" in signal.reason
        assert signal.net_profit > MIN_WORTHWHILE_NET_PROFIT


class TestDynamicThresholds:
    """Test ATR-based dynamic threshold calculations."""
    
    def setup_method(self):
        self.mock_market = MagicMock()
        self.mock_regime = MagicMock()
        self.mock_regime.classify.return_value = {'regime': 'NEUTRAL'}
        
        self.watchdog = PositionWatchdog(
            db_handler=MagicMock(),
            market_data=self.mock_market,
            ml_predictor=MagicMock(),
            regime_classifier=self.mock_regime
        )
    
    def test_high_volatility_wider_thresholds(self):
        """High volatility = wider stop loss."""
        self.mock_market.calculate_atr.return_value = {'atr': 10.0, 'volatility': 'HIGH'}
        
        thresholds = self.watchdog._calculate_dynamic_thresholds('AAPL', 100.0)
        
        # High vol: sl_mult = 2.5, tp_mult = 4.0
        # SL: 100 - (10 * 2.5) = 75
        # TP: 100 + (10 * 4.0) = 140
        assert thresholds['stop_loss_price'] == 75.0
        assert thresholds['take_profit_price'] == 140.0
    
    def test_bear_market_tighter_thresholds(self):
        """Bear market = tighter thresholds."""
        self.mock_market.calculate_atr.return_value = {'atr': 10.0, 'volatility': 'MEDIUM'}
        self.mock_regime.classify.return_value = {'regime': 'BEAR'}
        
        thresholds = self.watchdog._calculate_dynamic_thresholds('AAPL', 100.0)
        
        # Medium vol: 2.0, 3.0 | Bear: *0.8, *0.7
        # SL: 100 - (10 * 2.0 * 0.8) = 84
        # TP: 100 + (10 * 3.0 * 0.7) = 121
        assert thresholds['stop_loss_price'] == 84.0
        assert thresholds['take_profit_price'] == 121.0
    
    def test_crypto_gets_wider_thresholds(self):
        """Crypto assets get wider thresholds."""
        self.mock_market.calculate_atr.return_value = {'atr': 100.0, 'volatility': 'MEDIUM'}
        
        thresholds = self.watchdog._calculate_dynamic_thresholds('BTC-USD', 1000.0)
        
        # Medium: 2.0, 3.0 | Crypto: *1.3, *1.4
        # SL: 1000 - (100 * 2.0 * 1.3) = 740
        # TP: 1000 + (100 * 3.0 * 1.4) = 1420
        assert thresholds['stop_loss_price'] == 740.0
        assert thresholds['take_profit_price'] == 1420.0


class TestTelegramReport:
    """Test Telegram report formatting."""
    
    def test_empty_report(self):
        watchdog = PositionWatchdog()
        report = watchdog.format_telegram_report([])
        
        assert "Nessun exit signal" in report
        assert "Tasse" in report
    
    def test_report_with_signals(self):
        watchdog = PositionWatchdog()
        signals = [
            ExitSignal(
                ticker="AAPL", action="SELL",
                reason="STOP LOSS", urgency="CRITICAL",
                current_price=85.0, entry_price=100.0,
                quantity=10, pnl_percent=-15.0,
                gross_profit=-150.0, tax_amount=0.0,
                net_profit=-151.0, dynamic_stop_loss=85.0,
                confidence=0.9, technical_score=-50,
                suggested_action="Vendi AAPL"
            )
        ]
        
        report = watchdog.format_telegram_report(signals)
        
        assert "AAPL" in report
        assert "SELL" in report
        assert "🔴" in report  # CRITICAL urgency
        assert "Netto: €-151.00" in report
        assert "SELL" in report


class TestPhase2News:
    """Test news sentiment integration for exits."""
    
    def setup_method(self):
        self.mock_db = MagicMock()
        self.mock_market = MagicMock()
        self.mock_ml = MagicMock()
        self.mock_hunter = MagicMock()
        
        self.watchdog = PositionWatchdog(
            db_handler=self.mock_db,
            market_data=self.mock_market,
            ml_predictor=self.mock_ml
        )
        self.watchdog.hunter = self.mock_hunter
        self.mock_market.calculate_atr.return_value = {'atr': 5.0, 'volatility': 'MEDIUM'}
    
    @pytest.mark.asyncio
    async def test_exit_on_bad_news(self):
        """Negative news should trigger a SELL signal even if price is OK."""
        position = {
            'ticker': 'AAPL',
            'quantity': 10,
            'avg_price': 100.0,
            'target_price': 150.0,
            'stop_loss_price': 80.0
        }
        
        self.mock_market.get_smart_price_eur.return_value = (105.0, 'AAPL')
        self.mock_hunter.check_owned_asset_news.return_value = {
            'is_negative': True,
            'summary': 'AAPL reporting major hack'
        }
        self.mock_ml.predict.return_value = None # No ML signal
        self.mock_market.get_technical_summary.return_value = {'rsi': 50}
        
        signal = await self.watchdog._analyze_position(position)
        
        assert signal is not None
        assert signal.action == "SELL"
        assert "NEWS EXIT" in signal.reason
        assert "major hack" in signal.reason


class TestPhase3Targets:
    """Test automatic target setting logic."""
    
    def setup_method(self):
        self.mock_db = MagicMock()
        self.mock_market = MagicMock()
        self.mock_ml = MagicMock()
        self.mock_hunter = MagicMock()
        self.mock_regime = MagicMock()
        
        self.watchdog = PositionWatchdog(
            db_handler=self.mock_db,
            market_data=self.mock_market,
            ml_predictor=self.mock_ml,
            regime_classifier=self.mock_regime
        )
        self.watchdog.hunter = self.mock_hunter
        self.mock_regime.classify.return_value = {'regime': 'NEUTRAL'}
    
    @pytest.mark.asyncio
    async def test_auto_target_setting(self):
        """Missing targets should be auto-set based on dynamic thresholds."""
        position = {
            'ticker': 'TSLA',
            'quantity': 5,
            'avg_price': 200.0,
            'target_price': None, # Missing
            'stop_loss_price': None # Missing
        }
        
        self.mock_market.get_smart_price_eur.return_value = (210.0, 'TSLA')
        self.mock_market.calculate_atr.return_value = {'atr': 10.0, 'volatility': 'MEDIUM'}
        self.mock_hunter.check_owned_asset_news.return_value = {'is_negative': False}
        self.mock_ml.predict.return_value = None
        self.mock_market.get_technical_summary.return_value = {'rsi': 50}
        
        # Call analysis
        signal = await self.watchdog._analyze_position(position)
        
        # Verify DB update was called
        self.mock_db.update_portfolio_targets.assert_called_once()
        args, kwargs = self.mock_db.update_portfolio_targets.call_args
        assert kwargs['ticker'] == 'TSLA'
        assert kwargs['target_type'] == 'AUTO'
        assert kwargs['target_price'] > 210.0 # Based on ATR 10 * 3
        
        # Should return a "HOLD" signal because target was just set
        assert signal is not None
        assert signal.action == "HOLD"
        assert "TARGET AUTO SET" in signal.reason


class TestPhase9Customization:
    """Test Phase 9: Risk Profiles and Core Assets."""
    
    def setup_method(self):
        self.mock_db = MagicMock()
        self.mock_market = MagicMock()
        self.mock_ml = MagicMock()
        self.mock_hunter = MagicMock()
        self.mock_regime = MagicMock()
        
        self.watchdog = PositionWatchdog(
            db_handler=self.mock_db,
            market_data=self.mock_market,
            ml_predictor=self.mock_ml,
            regime_classifier=self.mock_regime
        )
        self.watchdog.hunter = self.mock_hunter
        self.mock_regime.classify.return_value = {'regime': 'NEUTRAL'}
        self.mock_market.calculate_atr.return_value = {'atr': 10.0, 'volatility': 'MEDIUM'}
        self.mock_hunter.check_owned_asset_news.return_value = {'is_negative': False}

    def test_risk_profile_conservative(self):
        """CONSERVATIVE profile should have tighter thresholds."""
        self.watchdog.settings = {'risk_profile': 'CONSERVATIVE'}
        thresholds = self.watchdog._calculate_dynamic_thresholds('AAPL', 100.0)
        
        # Conservative + Medium Vol: sl_mult = 1.2, tp_mult = 2.5
        # SL: 100 - (10 * 1.2) = 88.0
        assert thresholds['stop_loss_price'] == 88.0

    def test_risk_profile_aggressive(self):
        """AGGRESSIVE profile should have wider thresholds."""
        self.watchdog.settings = {'risk_profile': 'AGGRESSIVE'}
        thresholds = self.watchdog._calculate_dynamic_thresholds('AAPL', 100.0)
        
        # Aggressive + Medium Vol: sl_mult = 3.0, tp_mult = 5.0
        # SL: 100 - (10 * 3.0) = 70.0
        assert thresholds['stop_loss_price'] == 70.0

    @pytest.mark.asyncio
    async def test_oversold_protection_threshold_new(self):
        """Test the new -15% RSI protection threshold."""
        position = {
            'ticker': 'ETH-USD',
            'quantity': 10,
            'avg_price': 100.0
        }
        
        # Price is -20% (80.0), which is below Stop Loss but above the old -30%
        self.mock_market.get_smart_price_eur.return_value = (80.0, 'ETH-USD')
        self.mock_market.calculate_atr.return_value = {'atr': 5.0, 'volatility': 'MEDIUM'}
        # RSI is oversold
        self.mock_market.get_technical_summary.return_value = {'rsi': 25}
        
        # Should now return HOLD due to "OVERSOLD PROTECTION"
        signal = await self.watchdog._analyze_position(position)
        
        assert signal is not None
        assert signal.action == "HOLD"
        assert "OVERSOLD PROTECTION" in signal.reason
        assert "RSI < 35" in signal.reason

    @pytest.mark.asyncio
    async def test_core_asset_protection(self):
        """Core assets should ignore technical SELL signals."""
        position = {
            'ticker': 'BTC-USD',
            'quantity': 1,
            'avg_price': 50000.0,
            'is_core': True
        }
        
        # PnL % = -10% (Should be protected anyway by core logic)
        self.mock_market.get_smart_price_eur.return_value = (46000.0, 'BTC-USD') 
        self.mock_market.calculate_atr.return_value = {'atr': 1000.0, 'volatility': 'MEDIUM'}
        self.watchdog.settings = {'risk_profile': 'BALANCED'}
        
        signal = await self.watchdog._analyze_position(position)
        assert signal is None # Protected!

        # Test 2: PnL -20% and NOT oversold (Should trigger even if core)
        self.mock_market.get_smart_price_eur.return_value = (40000.0, 'BTC-USD')
        self.mock_market.get_technical_summary.return_value = {'rsi': 50} 
        signal = await self.watchdog._analyze_position(position)
        assert signal is not None
        assert signal.action == "SELL"
