"""
Tests for Phase 2: Dynamic Consensus Weights
=============================================
Verifies dynamic weight adjustment based on system performance.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDynamicConsensusWeights:
    """Tests for get_dynamic_weights() in consensus_engine.py"""
    
    def test_default_weights_sum_to_one(self):
        """Verify default weights sum to 1.0."""
        from consensus_engine import ConsensusEngine
        
        engine = ConsensusEngine()
        total = sum(engine.default_weights.values())
        
        assert abs(total - 1.0) < 0.001
    
    def test_dynamic_weights_defaults_with_insufficient_data(self):
        """With fewer than 10 trades, should use default weights."""
        from consensus_engine import ConsensusEngine
        
        with patch('db_handler.DBHandler') as mock_db:
            mock_db.return_value.get_audit_stats.return_value = {
                "win_rate": 60.0,
                "closed": 5,  # Less than 10
                "wins": 3,
                "losses": 2
            }
            
            engine = ConsensusEngine()
            weights = engine.get_dynamic_weights(force_refresh=True)
        
        assert weights == engine.default_weights
    
    def test_high_performance_boosts_ml_weight(self):
        """With >65% win rate, ML weight should increase."""
        from consensus_engine import ConsensusEngine
        
        with patch('db_handler.DBHandler') as mock_db:
            mock_db.return_value.get_audit_stats.return_value = {
                "win_rate": 70.0,  # High win rate
                "closed": 20,
                "wins": 14,
                "losses": 6
            }
            
            engine = ConsensusEngine()
            weights = engine.get_dynamic_weights(force_refresh=True)
        
        assert weights["ml"] > engine.default_weights["ml"]  # ML boosted
        assert weights["ml"] == pytest.approx(0.40, abs=0.01)
    
    def test_low_performance_boosts_council_weight(self):
        """With <45% win rate, Council weight should increase."""
        from consensus_engine import ConsensusEngine
        
        with patch('db_handler.DBHandler') as mock_db:
            mock_db.return_value.get_audit_stats.return_value = {
                "win_rate": 40.0,  # Low win rate
                "closed": 20,
                "wins": 8,
                "losses": 12
            }
            
            engine = ConsensusEngine()
            weights = engine.get_dynamic_weights(force_refresh=True)
        
        assert weights["council"] > engine.default_weights["council"]  # Council boosted
        assert weights["ml"] < engine.default_weights["ml"]  # ML reduced
    
    def test_dynamic_weights_sum_to_one(self):
        """Dynamic weights should always sum to 1.0 after normalization."""
        from consensus_engine import ConsensusEngine
        
        test_stats = [
            {"win_rate": 70.0, "closed": 20},  # High
            {"win_rate": 40.0, "closed": 20},  # Low
            {"win_rate": 55.0, "closed": 20},  # Medium
        ]
        
        for stats in test_stats:
            with patch('db_handler.DBHandler') as mock_db:
                mock_db.return_value.get_audit_stats.return_value = {
                    **stats,
                    "wins": int(stats["win_rate"] * stats["closed"] / 100),
                    "losses": int((100 - stats["win_rate"]) * stats["closed"] / 100)
                }
                
                engine = ConsensusEngine()
                weights = engine.get_dynamic_weights(force_refresh=True)
            
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"Weights don't sum to 1.0 for stats {stats}: {weights}"
    
    def test_calculate_weighted_action_includes_weights_used(self):
        """Response should include weights_used field."""
        from consensus_engine import ConsensusEngine
        
        with patch('db_handler.DBHandler') as mock_db:
            mock_db.return_value.get_audit_stats.return_value = {
                "win_rate": 55.0,
                "closed": 15,
                "wins": 8,
                "losses": 7
            }
            
            engine = ConsensusEngine()
            prediction = {
                "ticker": "TEST",
                "sentiment": "BUY",
                "critic_score": 75,
                "council_summary": "BUY recommended"
            }
            
            result = engine.calculate_weighted_action(prediction)
        
        assert "weights_used" in result
        assert isinstance(result["weights_used"], dict)
        assert "ml" in result["weights_used"]
        assert "council" in result["weights_used"]
    
    def test_weights_caching(self):
        """Weights should be cached for 5 minutes."""
        from consensus_engine import ConsensusEngine
        from datetime import datetime, timedelta
        
        with patch('db_handler.DBHandler') as mock_db:
            mock_db.return_value.get_audit_stats.return_value = {
                "win_rate": 55.0,
                "closed": 15,
                "wins": 8,
                "losses": 7
            }
            
            engine = ConsensusEngine()
            
            # First call - should hit DB
            weights1 = engine.get_dynamic_weights(force_refresh=True)
            call_count_after_first = mock_db.return_value.get_audit_stats.call_count
            
            # Second call - should use cache
            weights2 = engine.get_dynamic_weights()
            call_count_after_second = mock_db.return_value.get_audit_stats.call_count
            
            # Should be same call count (cached)
            assert call_count_after_first == call_count_after_second
            assert weights1 == weights2


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
