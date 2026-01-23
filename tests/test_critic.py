
import pytest
from critic import Critic
from unittest.mock import MagicMock

@pytest.fixture
def mock_critic_deps(mocker, mock_env):
    """Mock dependencies for Critic."""
    # Mock GenAI Client - use a more direct approach
    mock_client = MagicMock()
    mocker.patch("google.genai.Client", return_value=mock_client)
    
    # Mock Brain fallback
    mock_brain_class = mocker.patch("brain.Brain")
    mock_brain_instance = mock_brain_class.return_value
    
    # Helper to simulate responses
    def set_response(text):
        # Set for both to be safe
        mock_resp = MagicMock()
        mock_resp.text = text
        mock_client.models.generate_content.return_value = mock_resp
        mock_client.models.generate_content.side_effect = None
        
        mock_brain_instance._generate_with_fallback.return_value = text
            
    return mock_client, mock_brain_instance, set_response

def test_verify_unowned_asset_removal(mock_critic_deps, mocker):
    """Regression Test: Ensure Critic removes HOLD for unowned assets."""
    mock_client, mock_brain, set_response = mock_critic_deps
    
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
    
    # Verify Gemini was called
    args, kwargs = mock_client.models.generate_content.call_args
    prompt_sent = kwargs.get('contents', args[1] if len(args) > 1 else "")
    
    # CRITICAL: Verify the prompt contains the held assets list!
    assert "CURRENT PORTFOLIO ASSETS: BTC-USD" in str(prompt_sent)
    assert "DOGE" in str(prompt_sent)
    
    # Critic wraps modified strategy in a header. Check for inclusion.
    assert "🟢 ACCUMULATE BTC-USD" in result
    assert "Expert Broker Review" in result

def test_veto_buy_logic(mock_critic_deps, mocker):
    """Test that Critic prompt includes VETO instructions."""
    mock_client, mock_brain, set_response = mock_critic_deps
    
    critic = Critic()
    strategy_input = "🟢 BUY PEPE"
    held_assets = ["BTC-USD"]
    
    # Simulate LLM returning modified strategy in JSON
    json_output = '{"revised_strategy": "🚫 AVOID PEPE - Risk too high", "was_modified": true, "broker_reasoning": "Bearish regime risk."}'
    set_response(json_output)
    
    result = critic.critique_rebalance_strategy(strategy_input, "BEARISH", 10000, held_assets)
    
    # Verify prompt contains specific Bearish instructions
    args, kwargs = mock_client.models.generate_content.call_args
    prompt_sent = kwargs.get('contents', args[1] if len(args) > 1 else "")
    
    assert "BEARISH" in str(prompt_sent)
    assert "Expert Broker Review" in result
    assert "🚫 AVOID PEPE - Risk too high" in result
