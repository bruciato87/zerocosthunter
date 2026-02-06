"""
Tests for Critic ML-First Architecture and AI Fallback Resilience.

These tests verify:
1. ML-First local validation for high-confidence signals
2. AI escalation only for low-confidence edge cases
3. Graceful fallback when all AI drivers fail
"""

import pytest
from unittest.mock import MagicMock, patch
from critic import Critic
import json


class TestCriticMLFirst:
    """Tests for the ML-First Critic architecture."""
    
    def test_high_confidence_approved_locally(self):
        """High confidence signals (>70%) should be approved locally without AI."""
        critic = Critic()
        
        signal = {"ticker": "BTC", "direction": "BUY", "confidence": 0.90, "reasoning": "Moon"}
        result = critic._ml_critique(signal)
        
        assert result is not None
        assert result.verdict == "APPROVE"
        assert result.score == 90  # 70 + (0.9 - 0.7) * 100
        assert "[ML Critic]" in result.reasoning
    
    def test_medium_confidence_approved_with_caution(self):
        """Medium confidence signals (40-70%) should be approved with caution."""
        critic = Critic()
        
        signal = {"ticker": "ETH", "direction": "SELL", "confidence": 0.55, "reasoning": "Bearish"}
        result = critic._ml_critique(signal)
        
        assert result is not None
        assert result.verdict == "APPROVE"
        assert result.score == 66  # 50 + 0.55 * 30
        assert "Moderate" in result.reasoning
    
    def test_low_confidence_escalates_to_ai(self):
        """Low confidence signals (<40%) should escalate to AI."""
        critic = Critic()
        
        signal = {"ticker": "DOGE", "direction": "BUY", "confidence": 0.30, "reasoning": "Maybe"}
        result = critic._ml_critique(signal)
        
        assert result is None  # Should return None to trigger AI escalation
    
    def test_hold_signals_auto_approved(self):
        """HOLD signals should always be auto-approved regardless of confidence."""
        critic = Critic()
        
        signal = {"ticker": "XRP", "direction": "HOLD", "confidence": 0.10, "reasoning": "No signal"}
        result = critic._ml_critique(signal)
        
        assert result is not None
        assert result.verdict == "APPROVE"
        assert result.score == 60
        assert "HOLD signals auto-approved" in result.reasoning


class TestCriticAIFallback:
    """Tests for AI fallback when ML-First escalates."""
    
    @patch('brain.Brain')
    def test_ai_fallback_for_low_confidence(self, mock_brain_class):
        """Low confidence signals should escalate to AI and get a verdict."""
        mock_brain_instance = MagicMock()
        mock_brain_class.return_value = mock_brain_instance
        
        # Setup AI response
        ai_response = json.dumps({
            "verdict": "REJECT",
            "score": 35,
            "concerns": ["High risk pattern"],
            "reasoning": "Pattern recognition suggests high risk."
        })
        mock_brain_instance._generate_with_fallback.return_value = ai_response
        
        critic = Critic(brain_instance=mock_brain_instance)
        
        # Low confidence signal that triggers AI escalation
        signal = {"ticker": "SHIB", "direction": "BUY", "confidence": 0.25, "reasoning": "Pump"}
        result = critic.critique_signal(signal, "Context")
        
        # AI should have been called and returned REJECT
        assert result.verdict == "REJECT"
        assert result.score == 35
        mock_brain_instance._generate_with_fallback.assert_called_once()
    
    @patch('brain.Brain')
    def test_ai_total_failure_returns_safe_hold(self, mock_brain_class):
        """When all AI fails for low-confidence signals, return safe HOLD."""
        mock_brain_instance = MagicMock()
        mock_brain_class.return_value = mock_brain_instance
        
        # Both Main AND Last Resort fail
        mock_brain_instance._generate_with_fallback.side_effect = Exception("Main Fail")
        mock_brain_instance._call_gemini_fallback.side_effect = Exception("Last Resort Fail")
        mock_brain_instance.gemini_client = True
        
        critic = Critic(brain_instance=mock_brain_instance)
        
        # Very low confidence signal that needs AI but AI fails
        signal = {"ticker": "MEME", "direction": "BUY", "confidence": 0.15, "reasoning": "YOLO"}
        result = critic.critique_signal(signal, "Context")
        
        # Should return HOLD as safe fallback
        assert result.verdict == "HOLD"
        assert "Expert Broker unavailable" in result.reasoning
