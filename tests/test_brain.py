
import pytest
from brain import Brain
from unittest.mock import MagicMock

@pytest.fixture
def mock_brain_deps(mocker, mock_env):
    """Mock dependencies for Brain class."""
    # Mock OpenRouter
    mocker.patch("requests.post")
    
    # Mock Gemini Client
    mocker.patch("google.genai.Client")
    
    # Mock Critic to prevent initialization
    mocker.patch("critic.Critic")
    
    # Mock DBHandler to prevent logging calls
    mocker.patch("db_handler.DBHandler")

def test_openrouter_success(mock_brain_deps, mocker, monkeypatch):
    """Test standard OpenRouter success path."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_router_key")
    
    # Setup Mock Response
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenRouter Result"}}]
    }
    mock_post.return_value = mock_response
    
    brain = Brain()
    # Force use of OpenRouter
    brain.app_mode = "PROD" 
    
    result = brain._generate_with_fallback("Test Prompt")
    
    assert result == "OpenRouter Result"
    assert brain.last_run_details["provider"] == "OpenRouter"

def test_fallback_to_gemini(mock_brain_deps, mocker, monkeypatch):
    """Test OpenRouter failure triggering Gemini fallback."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_router_key")
    
    # 1. Fail OpenRouter
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.status_code = 500  # Simulate Server Error
    mock_post.return_value = mock_response
    
    # 2. Setup Gemini Success
    brain = Brain()
    brain.app_mode = "PROD"
    
    # Mock the client instance attached to brain
    mock_gemini_response = MagicMock()
    mock_gemini_response.text = "Gemini Result"
    
    # Brain.__init__ creates self.gemini_client. 
    # We need to ensure that client's models.generate_content returns success
    brain.gemini_client.models.generate_content.return_value = mock_gemini_response
    
    result = brain._generate_with_fallback("Test Prompt")
    
    assert result == "Gemini Result"
    # Provider should be updated to Google Direct
    assert "Google Direct" in brain.last_run_details["provider"]

def test_gemini_retry_logic(mock_brain_deps, mocker):
    """Test Gemini retries on 429 then succeeds."""
    brain = Brain()
    
    # Mock Gemini Client
    mock_gen_content = brain.gemini_client.models.generate_content
    
    # Side effect: Raise 429 twice, then succeed
    mock_gen_content.side_effect = [
        Exception("429 Resource Exhausted"),
        Exception("429 Resource Exhausted"),
        MagicMock(text="Gemini Retry Success")
    ]
    
    # We verify _call_gemini_with_retries directly
    # Need to patch time.sleep to speed up test
    mocker.patch("time.sleep")
    
    result = brain._call_gemini_with_retries("Test Prompt", json_mode=False)
    
    assert result == "Gemini Retry Success"
    assert mock_gen_content.call_count == 3
