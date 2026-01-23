
import pytest
from market_data import MarketData

def test_get_current_price_success(mock_market_deps):
    """Test price fetch logic with mocked yfinance."""
    md = MarketData()
    # Mock requests to ensure CoinGecko returns nothing, forcing YFinance fallback
    # The default mock_requests returns empty json, so get_crypto_data_coingecko returns None
    
    price = md.get_market_price("BTC-USD")
    assert price == 100.0

def test_smart_price_eur(mock_market_deps):
    """Test EUR conversion logic."""
    md = MarketData()
    price, found_ticker = md.get_smart_price_eur("BTC-USD")
    assert isinstance(price, float)
    assert found_ticker == "BTC-USD" 

def test_get_current_price_failure(mocker, mock_requests):
    """Test failure handling when yfinance raises exception."""
    # Ensure CoinGecko fails
    mock_requests.return_value.json.return_value = {}
    
    mock_ticker = mocker.patch("yfinance.Ticker")
    instance = mock_ticker.return_value
    
    # Critical: MarketData checks fast_info first. Ensure it fails or returns None.
    # By default specific attributes of MagicMock are MagicMocks.
    # We set fast_info to raise AttributeError to simulate it missing or failing
    del instance.fast_info 
    
    # Also fail history
    instance.history.side_effect = Exception("API Error")
    # Also fail info fallback
    instance.info = {}
    
    md = MarketData()
    price = md.get_market_price("INVALID")
    assert price is None
