
import pytest
from unittest.mock import MagicMock
import os
import sys

# Ensure the root directory is in the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Disable yfinance session cache to avoid ResourceWarning in tests
try:
    import yfinance as yf
    import requests
    # Ensure no persistent cache is used during tests
    yf.set_tz_cache_location(None)
except Exception:
    pass

@pytest.fixture
def mock_db(mocker):
    """Mock the DBHandler to avoid database calls."""
    mock = mocker.patch("db_handler.DBHandler")
    instance = mock.return_value
    # Setup common return values
    instance.get_portfolio.return_value = []
    instance.get_ticker_cache.return_value = None
    return instance

@pytest.fixture
def mock_yfinance(mocker):
    """Mock yfinance to avoid external API calls."""
    mock_ticker = mocker.patch("yfinance.Ticker")
    instance = mock_ticker.return_value
    
    # Mock history dataframe
    import pandas as pd
    df = pd.DataFrame({
        "Close": [100.0],
        "High": [105.0],
        "Low": [95.0],
        "Volume": [1000]
    })
    instance.history.return_value = df
    instance.info = {"currentPrice": 100.0, "currency": "USD"}
    
    # Mock fast_info behavior
    # MarketData calls t.fast_info.get('last_price')
    instance.fast_info.get.return_value = 100.0
    
    return instance

@pytest.fixture
def mock_requests(mocker):
    """Mock requests to avoid external API calls (CoinGecko)."""
    mock = mocker.patch("requests.get")
    mock.return_value.json.return_value = {} # Default empty response
    mock.return_value.status_code = 200
    return mock

@pytest.fixture
def mock_market_deps(mock_yfinance, mock_requests, mock_db, mocker):
    """Setup MarketData dependencies."""
    return mock_yfinance, mock_requests, mock_db

@pytest.fixture
def mock_brain(mocker):
    """Mock the AI Brain to avoid LLM costs/latency."""
    mock = mocker.patch("brain.Brain")
    instance = mock.return_value
    # Default behavior: Return a generic string
    instance._generate_with_fallback.return_value = "AI Analysis Result"
    instance._call_gemini_with_tiered_fallback.return_value = "Gemini Fallback Result"
    return instance

@pytest.fixture
def mock_env(monkeypatch):
    """Override environment variables for testing."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake_key")
