
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

def test_quota_guard_logic(mock_brain_deps, mocker):
    """Verify background tasks bypass Gemini Direct."""
    brain = Brain()
    brain.gemini_api_key = "fake"
    brain.openrouter_api_key = "fake"
    
    # Mock OpenRouter success
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "OR"}}, {"usage": {}}]}
    
    # Mock Gemini Direct
    mock_direct = mocker.patch.object(brain, "_call_gemini_with_tiered_fallback")
    
    # Background task with prefer_direct=True should still use OpenRouter
    brain._generate_with_fallback("Prompt", task_type="council_debate", prefer_direct=True)
    assert not mock_direct.called
    
    # Manual task should use Gemini Direct
    brain._generate_with_fallback("Prompt", task_type="analyze", prefer_direct=True)
    assert mock_direct.called

def test_new_sentiments_support(mock_brain_deps, mocker):
    """Verify system handles new 'WATCH' and 'AVOID' sentiments if returned by AI."""
    brain = Brain()
    
    # Mock OpenRouter response directly
    mock_post = mocker.patch("requests.post")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '[{"ticker": "AAPL", "sentiment": "WATCH", "reasoning": "Test", "confidence": 0.8, "risk_score": 4, "target_price": "€200", "upside_percentage": 10.0, "stop_loss": 180.0, "take_profit": 220.0}]'}}]
    }
    mock_post.return_value = mock_response
    
    # Test the logic
    response_text = brain._generate_with_fallback("Prompt", json_mode=True)
    import json
    analysis_results = json.loads(response_text)
    
    assert analysis_results[0]["sentiment"] == "WATCH"
    assert analysis_results[0]["ticker"] == "AAPL"
