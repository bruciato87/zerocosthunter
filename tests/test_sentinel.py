import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sentinel import Sentinel

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def sentinel(mock_db):
    with patch('sentinel.PaperTrader'):
        return Sentinel(db_handler=mock_db)

@pytest.mark.asyncio
async def test_check_price_alerts_above(sentinel, mock_db):
    """Test triggered ABOVE alert."""
    mock_db.get_active_alerts.return_value = [
        {'id': 1, 'ticker': 'BTC-USD', 'condition': 'ABOVE', 'price_threshold': 50000, 'chat_id': 123}
    ]
    
    market_data = MagicMock()
    market_data.get_smart_price_eur_async = AsyncMock(return_value=(60000, "BTC-USD"))
    
    notifications = await sentinel.check_alerts(market_data)
    
    assert len(notifications) >= 1
    assert "BTC-USD" in notifications[0]["text"]
    assert "sopra €50000" in notifications[0]["text"]
    mock_db.deactivate_alert.assert_called_once()

@pytest.mark.asyncio
async def test_check_price_alerts_below(sentinel, mock_db):
    """Test triggered BELOW alert."""
    mock_db.get_active_alerts.return_value = [
        {'id': 2, 'ticker': 'AAPL', 'condition': 'BELOW', 'price_threshold': 150, 'chat_id': 123}
    ]
    
    market_data = MagicMock()
    market_data.get_smart_price_eur_async = AsyncMock(return_value=(140, "AAPL"))
    
    notifications = await sentinel.check_alerts(market_data)
    
    assert len(notifications) >= 1
    assert "sotto €150" in notifications[0]["text"]

@pytest.mark.asyncio
async def test_check_portfolio_volatility_breaker(sentinel, mock_db):
    """Test volatility breaker notification."""
    mock_db.get_portfolio.return_value = [
        {'ticker': 'SOL-USD', 'chat_id': 123, 'avg_price': 100}
    ]
    
    market_data = MagicMock()
    # Mocking price and -6% change
    market_data.get_smart_price_eur_async = AsyncMock(return_value=(100, "SOL-USD", -6.0))
    
    notifications = await sentinel.check_alerts(market_data)
    
    # One from price alerts (empty), one from risk check
    assert any("Volatility Breaker" in n["text"] for n in notifications)

def test_get_strategic_forecast(sentinel, mock_db):
    """Test strategic forecast logic."""
    mock_db.get_portfolio.return_value = [
        {'ticker': 'BTC-USD', 'quantity': 1, 'current_price': 50000, 'avg_price': 40000},
        {'ticker': 'ETH-USD', 'quantity': 10, 'current_price': 3000, 'avg_price': 2500}
    ]
    
    market_data = MagicMock()
    market_data.calculate_correlation_matrix.return_value = {
        'high_correlation_pairs': [("BTC-USD", "ETH-USD", 0.95)],
        'diversification_score': 30
    }
    
    with patch('strategy_manager.StrategyManager.get_market_regime') as mock_regime:
        mock_regime.return_value = {
            'regime': 'BULL',
            'description': '🐂 BULLISH',
            'targets': {'Crypto': 40.0, 'Technology': 30.0},
            'recommendation': 'aggressive'
        }
        
        forecast = sentinel.get_strategic_forecast(market_data)
        
        assert "regime" in forecast
        assert forecast["regime"] == "🐂 BULLISH"
        # Since BTC is $50k and total portfolio is $50k, Crypto is 100%
        # Target is 40%, so we expect an OVERWEIGHT gap for Crypto
        assert any(g["sector"] == "Crypto" and g["type"] == "OVERWEIGHT" for g in forecast["gaps"])
        assert any("correlati" in c for c in forecast["correlation_warnings"])
