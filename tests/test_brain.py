
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

def test_gemini_tiered_fallback(mock_brain_deps, mocker):
    """Test Gemini falls back to next tier on 429."""
    brain = Brain()
    
    # Mock Gemini Client
    mock_gen_content = brain.gemini_client.models.generate_content
    
    # Side effect: Raise 429 twice (for gemini-3-flash and gemini-2.5-flash), 
    # then succeed (for gemini-2.5-flash-lite)
    mock_gen_content.side_effect = [
        Exception("429 Resource Exhausted"),
        Exception("429 Resource Exhausted"),
        MagicMock(text="Gemini Tier Success")
    ]
    
    # We verify _call_gemini_with_tiered_fallback directly
    result = brain._call_gemini_with_tiered_fallback("Test Prompt", json_mode=False)
    
    assert result == "Gemini Tier Success"
    assert mock_gen_content.call_count == 3
    # Check that it called with different models (optional but good)
    calls = mock_gen_content.call_args_list
    assert calls[0].kwargs['model'] == "gemini-2.0-flash"
    assert calls[1].kwargs['model'] == "gemini-2.0-flash-lite"
    assert calls[2].kwargs['model'] == "gemini-flash-latest"

def test_parse_trade_republic_pdf(mock_brain_deps, mocker):
    """Test PDF parsing logic with mock PDF and AI response."""
    import json
    brain = Brain()
    
    # 1. Mock file opening (validation check)
    mocker.patch("builtins.open", mocker.mock_open(read_data=b"%PDF-1.7"))

    # 2. Mock PdfReader
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Trade Republic Order Confirmation: Bought 10 AAPL @ 150.25 EUR"
    mock_pdf.pages = [mock_page]
    mocker.patch("pypdf.PdfReader", return_value=mock_pdf)
    
    # 3. Mock AI Generation
    mock_ai_response = {
        "ticker": "AAPL",
        "action": "BUY",
        "quantity": 10.0,
        "price": 150.25,
        "commission": 1.0,
        "tax": 0.0,
        "net_total": 1503.50,
        "asset_name": "Apple Inc"
    }
    mocker.patch.object(brain, "_generate_with_fallback", return_value=json.dumps(mock_ai_response))
    
    # 3. Execute
    result = brain.parse_trade_republic_pdf("mock.pdf")
    
    # 4. Verify
    assert result["ticker"] == "AAPL"
    assert result["quantity"] == 10.0
    assert result["action"] == "BUY"
    assert brain._generate_with_fallback.called
