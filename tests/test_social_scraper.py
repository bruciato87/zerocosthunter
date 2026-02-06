import pytest
from unittest.mock import MagicMock, patch
from social_scraper import SocialScraper

@pytest.fixture
def scraper():
    return SocialScraper()

def test_detect_velocity_surging(scraper):
    """Test detection of a sentiment surge."""
    with patch('db_handler.DBHandler') as mock_db_class:
        mock_db = mock_db_class.return_value
        # History: past 3 runs were low (e.g., 1 mention each)
        mock_db.get_social_history.return_value = [
            {'mentions': 10, 'created_at': '2026-01-26T12:00:00'}, # Current (index 0)
            {'mentions': 1, 'created_at': '2026-01-26T11:00:00'},
            {'mentions': 1, 'created_at': '2026-01-26T10:00:00'},
            {'mentions': 1, 'created_at': '2026-01-26T09:00:00'}
        ]
        
        # Current count is 10, avg past is 1. Growth = 900%
        result = scraper.detect_velocity("BTC", 10)
        
        assert result is not None
        assert result["status"] == "🚀 SURGING"
        assert result["growth"] > 150

def test_detect_velocity_stable(scraper):
    """Test stable sentiment detection."""
    with patch('db_handler.DBHandler') as mock_db_class:
        mock_db = mock_db_class.return_value
        # History: past 3 runs were similar (e.g., 5 mentions each)
        mock_db.get_social_history.return_value = [
            {'mentions': 6, 'created_at': '2026-01-26T12:00:00'}, # Current
            {'mentions': 5, 'created_at': '2026-01-26T11:00:00'},
            {'mentions': 5, 'created_at': '2026-01-26T10:00:00'},
            {'mentions': 5, 'created_at': '2026-01-26T09:00:00'}
        ]
        
        # Current count 6, avg past 5. Growth = 20%
        result = scraper.detect_velocity("ETH", 6)
        
        assert result is not None
        assert result["status"] == "STABLE"

def test_get_social_context_integration(scraper):
    """Test if context string includes velocity label."""
    with patch.object(scraper, 'get_reddit_trending', return_value={"SOL": 10}):
        with patch.object(scraper, 'detect_velocity', return_value={"status": "🚀 SURGING", "growth": 900}):
            context = scraper.get_social_context("SOL")
            assert "SOL" in context
            assert "🚀 SURGING" in context
            assert "HIGH HYPE" in context
