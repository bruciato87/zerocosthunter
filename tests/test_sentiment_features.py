"""
Tests for Phase 2: Sentiment Features in ML Predictor
=====================================================
Verifies sentiment integration and ML feature extraction.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSentimentAggregator:
    """Tests for get_numeric_scores() in sentiment_aggregator.py"""
    
    def test_get_numeric_scores_returns_expected_keys(self):
        """Verify get_numeric_scores returns all required keys."""
        from sentiment_aggregator import SentimentAggregator
        
        with patch.object(SentimentAggregator, 'get_fear_greed_stock', return_value=55):
            with patch.object(SentimentAggregator, 'get_fear_greed_crypto', return_value=60):
                with patch.object(SentimentAggregator, 'get_whale_sentiment', return_value="NEUTRAL"):
                    with patch.object(SentimentAggregator, 'get_vix_level', return_value=20.0):
                        agg = SentimentAggregator()
                        scores = agg.get_numeric_scores(is_crypto=False)
        
        assert 'fear_greed_score' in scores
        assert 'whale_activity_score' in scores
        assert 'vix_normalized' in scores
    
    def test_get_numeric_scores_stock_uses_stock_fg(self):
        """For stocks, fear_greed_score should use stock F&G index."""
        from sentiment_aggregator import SentimentAggregator
        
        with patch.object(SentimentAggregator, 'get_fear_greed_stock', return_value=45):
            with patch.object(SentimentAggregator, 'get_fear_greed_crypto', return_value=70):
                with patch.object(SentimentAggregator, 'get_whale_sentiment', return_value="NEUTRAL"):
                    with patch.object(SentimentAggregator, 'get_vix_level', return_value=20.0):
                        agg = SentimentAggregator()
                        scores = agg.get_numeric_scores(is_crypto=False)
        
        assert scores['fear_greed_score'] == 45  # Should use stock value
    
    def test_get_numeric_scores_crypto_uses_crypto_fg(self):
        """For crypto, fear_greed_score should use crypto F&G index."""
        from sentiment_aggregator import SentimentAggregator
        
        with patch.object(SentimentAggregator, 'get_fear_greed_stock', return_value=45):
            with patch.object(SentimentAggregator, 'get_fear_greed_crypto', return_value=70):
                with patch.object(SentimentAggregator, 'get_whale_sentiment', return_value="NEUTRAL"):
                    with patch.object(SentimentAggregator, 'get_vix_level', return_value=20.0):
                        agg = SentimentAggregator()
                        scores = agg.get_numeric_scores(is_crypto=True)
        
        assert scores['fear_greed_score'] == 70  # Should use crypto value
    
    def test_whale_activity_mapping(self):
        """Verify whale status correctly maps to numeric score."""
        from sentiment_aggregator import SentimentAggregator
        
        test_cases = [
            ("BULLISH", 80),
            ("ACCUMULATION", 50),
            ("NEUTRAL", 0),
            ("DISTRIBUTION", -50),
            ("BEARISH", -80),
        ]
        
        for whale_status, expected_score in test_cases:
            with patch.object(SentimentAggregator, 'get_fear_greed_stock', return_value=50):
                with patch.object(SentimentAggregator, 'get_fear_greed_crypto', return_value=50):
                    with patch.object(SentimentAggregator, 'get_whale_sentiment', return_value=whale_status):
                        with patch.object(SentimentAggregator, 'get_vix_level', return_value=20.0):
                            agg = SentimentAggregator()
                            scores = agg.get_numeric_scores()
            
            assert scores['whale_activity_score'] == expected_score, f"Failed for {whale_status}"


class TestSocialScraper:
    """Tests for get_hype_score() in social_scraper.py"""
    
    def test_get_hype_score_returns_float(self):
        """Verify get_hype_score returns a float."""
        from social_scraper import SocialScraper
        
        with patch.object(SocialScraper, 'get_reddit_trending', return_value={"NVDA": 5}):
            with patch.object(SocialScraper, 'detect_velocity', return_value={"status": "STABLE", "growth": 0}):
                scraper = SocialScraper()
                score = scraper.get_hype_score("NVDA")
        
        assert isinstance(score, float)
        assert 0 <= score <= 10
    
    def test_hype_score_levels(self):
        """Test hype score ranges for different mention counts."""
        from social_scraper import SocialScraper
        
        test_cases = [
            (0, 0.0),    # No mentions = 0
            (1, 2.0),    # Low mentions
            (3, 4.0),    # Moderate
            (5, 6.0),    # High
            (10, 8.0),   # Very high
        ]
        
        for count, expected_base in test_cases:
            with patch.object(SocialScraper, 'get_reddit_trending', return_value={"TEST": count}):
                with patch.object(SocialScraper, 'detect_velocity', return_value={"status": "STABLE", "growth": 0}):
                    scraper = SocialScraper()
                    score = scraper.get_hype_score("TEST")
            
            assert score == expected_base, f"Failed for count={count}: got {score}, expected {expected_base}"
    
    def test_hype_score_velocity_boost(self):
        """Test that SURGING velocity boosts the hype score."""
        from social_scraper import SocialScraper
        
        with patch.object(SocialScraper, 'get_reddit_trending', return_value={"TEST": 5}):
            # Stable velocity
            with patch.object(SocialScraper, 'detect_velocity', return_value={"status": "STABLE", "growth": 0}):
                scraper = SocialScraper()
                stable_score = scraper.get_hype_score("TEST")
            
            # Surging velocity
            with patch.object(SocialScraper, 'detect_velocity', return_value={"status": "🚀 SURGING", "growth": 200}):
                scraper = SocialScraper()
                surging_score = scraper.get_hype_score("TEST")
        
        assert surging_score > stable_score


class TestMLPredictorSentimentFeatures:
    """Tests for sentiment features in MLPredictor."""
    
    def test_feature_columns_include_sentiment(self):
        """Verify FEATURE_COLUMNS includes all sentiment features."""
        from ml_predictor import MLPredictor
        
        required_features = [
            'fear_greed_score',
            'social_hype_score',
            'whale_activity_score',
        ]
        
        for feature in required_features:
            assert feature in MLPredictor.FEATURE_COLUMNS, f"Missing feature: {feature}"
    
    def test_feature_count(self):
        """Verify total feature count is as expected (19 original + 3 sentiment = 22)."""
        from ml_predictor import MLPredictor
        
        # 15 base + 4 time-aware (Phase 1) + 3 sentiment (Phase 2) = 22
        assert len(MLPredictor.FEATURE_COLUMNS) == 22


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
