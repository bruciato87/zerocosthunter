
import pytest
from critic import Critic
from unittest.mock import MagicMock

@pytest.fixture
def mock_critic_deps(mocker, mock_env):
    """Mock dependencies for Critic."""
    # Mock Brain fallback
    mock_brain_class = mocker.patch("brain.Brain")
    mock_brain_instance = mock_brain_class.return_value
    
    # Helper to simulate responses
    def set_response(text):
        mock_brain_instance._generate_with_fallback.return_value = text
            
    return mock_brain_instance, set_response

def test_verify_unowned_asset_removal(mock_critic_deps, mocker):
    """Regression Test: Ensure Critic removes HOLD for unowned assets."""
    mock_brain, set_response = mock_critic_deps
    
    critic = Critic()
    strategy_input = """
    🟢 ACCUMULATE BTC-USD
    🟡 HOLD DOGE
    """
    
    # We simulate that the LLM returns the cleaned list in JSON
    json_output = '{"revised_strategy": "🟢 ACCUMULATE BTC-USD", "was_modified": true, "broker_reasoning": "Asset DOGE not in portfolio."}'
    set_response(json_output)
    
    held_assets = ["BTC-USD"]
    
    result = critic.critique_rebalance_strategy(strategy_input, "SIDEWAYS", 10000, held_assets)
    
    # Verify Brain was called with correct priority
    args, kwargs = mock_brain._generate_with_fallback.call_args
    prompt_sent = args[0]
    
    assert kwargs.get('prefer_direct') is True
    assert "CURRENT PORTFOLIO ASSETS: BTC-USD" in prompt_sent
    assert "DOGE" in prompt_sent
    assert "🟢 ACCUMULATE BTC-USD" in result
    assert "Expert Broker Review" in result

def test_veto_buy_logic(mock_critic_deps, mocker):
    """Test that Critic prompt includes VETO instructions."""
    mock_brain, set_response = mock_critic_deps
    
    critic = Critic()
    strategy_input = "🟢 BUY PEPE"
    held_assets = ["BTC-USD"]
    
    # Simulate LLM returning modified strategy in JSON
    json_output = '{"revised_strategy": "🚫 AVOID PEPE - Risk too high", "was_modified": true, "broker_reasoning": "Bearish regime risk."}'
    set_response(json_output)
    
    result = critic.critique_rebalance_strategy(strategy_input, "BEARISH", 10000, held_assets)
    
    # Verify prompt contains specific Bearish instructions
    args, kwargs = mock_brain._generate_with_fallback.call_args
    prompt_sent = args[0]
    
    assert kwargs.get('prefer_direct') is True
    assert "BEARISH" in prompt_sent
    assert "Expert Broker Review" in result
    assert "🚫 AVOID PEPE - Risk too high" in result
