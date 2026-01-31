
import pytest
from unittest.mock import MagicMock, patch
from critic import Critic
import json

@patch('brain.Brain')
def test_critic_last_resort_retry(mock_brain_class):
    # Setup Mock Brain
    mock_brain_instance = MagicMock()
    mock_brain_class.return_value = mock_brain_instance
    
    # 1. Simulate Main Brain Failure (e.g. OpenRouter 429)
    # The first call to _generate_with_fallback usually handles the main logic
    mock_brain_instance._generate_with_fallback.side_effect = Exception("OpenRouter 429 Resource Exhausted")
    
    # 2. Setup Gemini Client for Last Resort check
    mock_brain_instance.gemini_client = True
    
    # 3. Simulate Successful Last Resort Call
    last_resort_response = json.dumps({
        "verdict": "APPROVE",
        "score": 75,
        "concerns": ["Minor volatility"],
        "reasoning": "Emergency check approved."
    })
    mock_brain_instance._call_gemini_fallback.return_value = last_resort_response
    
    # Initialize Critic with mocked brain
    critic = Critic(brain_instance=mock_brain_instance)
    
    # Run Critique
    signal = {"ticker": "BTC", "direction": "BUY", "confidence": 0.9, "reasoning": "Moon"}
    context = "Price $100k, Trend Up."
    
    result = critic.critique_signal(signal, context)
    
    # Verifications
    assert result.verdict == "APPROVE"
    assert result.score == 75
    assert "[Emergency Recovered]" in result.reasoning
    
    # Verify simplifed prompt call
    mock_brain_instance._call_gemini_fallback.assert_called_once()
    call_args = mock_brain_instance._call_gemini_fallback.call_args
    assert "You are a Risk Manager" in call_args[0][0] # Prompt check
    assert call_args[1]['model'] == "gemini-2.0-flash" # Model check

@patch('brain.Brain')
def test_critic_total_failure_fallback(mock_brain_class):
    # Setup Mock Brain for TOTAL failure
    mock_brain_instance = MagicMock()
    mock_brain_class.return_value = mock_brain_instance
    
    # Both Main AND Last Resort fail
    mock_brain_instance._generate_with_fallback.side_effect = Exception("Main Fail")
    mock_brain_instance._call_gemini_fallback.side_effect = Exception("Last Resort Fail")
    mock_brain_instance.gemini_client = True
    
    critic = Critic(brain_instance=mock_brain_instance)
    
    result = critic.critique_signal({"ticker": "ETH"}, "Context")
    
    # Should hit the final static fallback
    assert result.verdict == "HOLD"
    assert "Expert Broker unavailable" in result.reasoning
