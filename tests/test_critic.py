
import pytest
from critic import Critic
from unittest.mock import MagicMock

@pytest.fixture
def mock_critic_brain(mocker):
    """Mock the Brain used by Critic."""
    mock_brain_class = mocker.patch("brain.Brain")
    mock_brain_instance = mock_brain_class.return_value
    # Helper to simulate LLM responses
    def set_response(text):
        mock_brain_instance._generate_with_fallback.return_value = text
    return mock_brain_instance, set_response

def test_verify_unowned_asset_removal(mock_critic_brain):
    """Regression Test: Ensure Critic removes HOLD for unowned assets."""
    mock_brain, set_brain_response = mock_critic_brain
    
    # Scene: AI suggests HOLD DOGE, but we only own BTC.
    # The Critic Prompt should instruct the LLM to remove it.
    # WE MUST SIMULATE THE LLM OBEYING THE PROMPT.
    # Since we use a Mock LLM, we must manually return the "Correct" output 
    # that a good LLM would produce given the prompt.
    # This seemingly tests nothing (garbage in, garbage out), BUT:
    # It ensures the CODE passes the correct `held_assets` list to the method 
    # and that the method logic handles the response replacement correctly.
    
    # However, to test the PROMPT LOGIC itself, we'd need a real LLM.
    # Here we verify the Python wrapper logic around the prompt.
    
    critic = Critic()
    strategy_input = """
    🟢 ACCUMULATE BTC-USD
    🟡 HOLD DOGE
    """
    
    # We simulate that the LLM, properly prompted, returns the cleaned list
    expected_output = "🟢 ACCUMULATE BTC-USD"
    set_brain_response(expected_output)
    
    held_assets = ["BTC-USD"]
    
    result = critic.critique_rebalance_strategy(strategy_input, "SIDEWAYS", 10000, held_assets)
    
    # Verify Brain was called
    args, _ = mock_brain._generate_with_fallback.call_args
    prompt_sent = args[0]
    
    # CRITICAL: Verify the prompt contains the held assets list!
    assert "CURRENT PORTFOLIO ASSETS: BTC-USD" in prompt_sent
    # CRITICAL: Verify prompt contains the instruction to delete unowned HOLDs
    assert "DELETE THE LINE" in prompt_sent
    
    # Critic wraps modified strategy in a header. Check for inclusion.
    assert expected_output in result
    assert "Risk Manager Update" in result

def test_veto_buy_logic(mock_critic_brain):
    """Test that Critic prompt includes VETO instructions."""
    mock_brain, set_brain_response = mock_critic_brain
    
    critic = Critic()
    strategy_input = "🟢 BUY PEPE"
    held_assets = ["BTC-USD"]
    
    # Simulate LLM returning modified strategy
    set_brain_response("🚫 AVOID PEPE - Risk too high")
    
    result = critic.critique_rebalance_strategy(strategy_input, "BEARISH", 10000, held_assets)
    
    # Verify prompt contains specific Bearish instructions
    args, _ = mock_brain._generate_with_fallback.call_args
    prompt_sent = args[0]
    
    assert "Risk too high for Bearish Regime" in prompt_sent or "VETO" in prompt_sent
    assert result == "\n\n👮‍♂️ **Risk Manager Update (Critic)**:\n🚫 AVOID PEPE - Risk too high" 
