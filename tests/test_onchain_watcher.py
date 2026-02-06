import pytest
from onchain_watcher import OnChainWatcher
from unittest.mock import MagicMock, patch

@pytest.fixture
def watcher():
    return OnChainWatcher()

def test_dexscreener_parsing(watcher):
    """Test parsing of DexScreener API response."""
    mock_data = {
        "pairs": [
            {
                "pairAddress": "0x123",
                "baseToken": {"symbol": "SOL"},
                "priceUsd": "100.5",
                "liquidity": {"usd": 500000},
                "volume": {"h24": 1000000},
                "priceChange": {"h24": 15.5},
                "chainId": "solana"
            }
        ]
    }
    
    with patch.object(watcher.session, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response
        
        data = watcher.get_token_data("SOL")
        assert data is not None
        assert data['price_usd'] == 100.5
        assert data['chain'] == "solana"
        assert data['liquidity_usd'] == 500000

def test_onchain_context_formatting(watcher):
    """Test the string formatting of on-chain context."""
    with patch.object(watcher, 'get_token_data') as mock_data:
        mock_data.return_value = {
            "liquidity_usd": 200000,
            "volume_24h": 60000,
            "price_change_24h": 25.0,
            "chain": "solana"
        }
        
        context = watcher.get_onchain_context("SOL")
        assert "HEALTHY" in context
        assert "RALLYING" in context
        assert "25.0%" in context
