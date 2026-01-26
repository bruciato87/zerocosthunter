import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pulse_hunter import PulseHunter

@pytest.fixture
def mock_market():
    return MagicMock()

@pytest.fixture
def pulse_hunter(mock_market):
    with patch('pulse_hunter.DBHandler'):
        return PulseHunter(market_instance=mock_market)

def test_detect_anomalies_volume_spike(pulse_hunter, mock_market):
    """Test detection of significant volume spikes."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30)
    # Zig-zag prices to stay around RSI 50
    prices = [100.0, 102.0] * 15 
    data = {
        'Close': prices,
        'High': [200.0] * 30, # Far from breakout
        'Low': [50.0] * 30,
        'Volume': [1000] * 29 + [10000]  # 10x spike
    }
    df = pd.DataFrame(data, index=dates)
    mock_market.get_historical_data.return_value = df
    
    anomaly = pulse_hunter.detect_anomalies("TEST")
    
    assert anomaly is not None
    assert "Volume anomalo" in anomaly["findings"][0]
    assert anomaly["metrics"]["vol_ratio"] == 10.0
    assert anomaly["confidence_modifier"] >= 0.2

def test_detect_anomalies_rsi_oversold(pulse_hunter, mock_market):
    """Test detection of oversold conditions."""
    # Create 30 days of sharply falling prices to trigger RSI < 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30)
    prices = np.linspace(100, 50, 30)
    data = {
        'Close': prices,
        'High': prices + 5,
        'Low': prices - 5,
        'Volume': [1000] * 30
    }
    df = pd.DataFrame(data, index=dates)
    mock_market.get_historical_data.return_value = df
    
    anomaly = pulse_hunter.detect_anomalies("TEST")
    
    assert anomaly is not None
    assert any("Oversold" in f for f in anomaly["findings"])
    assert anomaly["metrics"]["rsi"] < 35

def test_detect_anomalies_breakout(pulse_hunter, mock_market):
    """Test detection of breakout potential near 20d high."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30)
    # Price is 99, 20d high is 100
    prices = [90] * 10 + [100] + [99] * 19 
    data = {
        'Close': prices,
        'High': prices,
        'Low': [80] * 30,
        'Volume': [1000] * 30
    }
    df = pd.DataFrame(data, index=dates)
    mock_market.get_historical_data.return_value = df
    
    anomaly = pulse_hunter.detect_anomalies("TEST")
    
    assert anomaly is not None
    assert any("Breakout" in f for f in anomaly["findings"])

def test_scan_aggregates_results(pulse_hunter, mock_market):
    """Test that scan method combines multiple findings."""
    pulse_hunter._get_watchlist = MagicMock(return_value=["BTC", "ETH"])
    
    # Mock detect_anomalies to return finding for BTC only
    pulse_hunter.detect_anomalies = MagicMock(side_effect=[{"ticker": "BTC", "findings": ["A"]}, None])
    
    results = pulse_hunter.scan()
    assert len(results) == 1
    assert results[0]["ticker"] == "BTC"
