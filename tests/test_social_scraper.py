import pytest
from social_scraper import SocialScraper
from unittest.mock import MagicMock, patch

@pytest.fixture
def scraper():
    return SocialScraper()

def test_reddit_ticker_extraction(scraper):
    """Test that tickers are correctly extracted from Reddit JSON format."""
    mock_data = {
        "data": {
            "children": [
                {"data": {"title": "Buy NVDA it is mooning", "selftext": "Thinking about AAPL too"}},
                {"data": {"title": "SOL vs ETH debate", "selftext": "I like SOL better"}}
            ]
        }
    }
    
    with patch.object(scraper.session, 'get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response
        
        trending = scraper.get_reddit_trending()
        
        assert "NVDA" in trending
        assert "AAPL" in trending
        assert "SOL" in trending
        assert "ETH" in trending
        assert "THE" not in trending # Blacklist check

def test_social_context_formatting(scraper):
    """Test the string formatting of social context."""
    with patch.object(scraper, 'get_reddit_trending') as mock_trending:
        mock_trending.return_value = {"BTC": 10, "ETH": 3}
        
        context = scraper.get_social_context("BTC")
        assert "HIGH HYPE" in context
        assert "10 mentions" in context
        
        context_eth = scraper.get_social_context("ETH")
        assert "MODERATE INTEREST" in context_eth
