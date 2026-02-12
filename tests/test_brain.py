
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
    
    # Mock OpenRouter Discovery
    mocker.patch("brain.Brain._get_best_free_model", return_value="meta-llama/llama-3.3-70b-instruct:free")
    
    # Mock Gemini Discovery
    mocker.patch("brain.Brain._get_best_gemini_model", return_value="gemini-2.0-flash-exp")

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
    # 1. Mock Gemini to fail/unavailable to reach OpenRouter
    mocker.patch.object(brain, "_get_best_gemini_model", return_value=None)
    
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

def test_priority_logic(mock_brain_deps, mocker):
    """Verify system prioritizes Gemini then OpenRouter."""
    brain = Brain()
    brain.gemini_api_key = "fake"
    brain.openrouter_api_key = "fake"
    
    # Mock Gemini success
    mock_gemini = mocker.patch.object(brain, "_call_gemini_fallback", return_value="GEMINI_WIN")
    
    # Mock OpenRouter (should not be called if Gemini succeeds)
    mock_or = mocker.patch.object(brain, "_call_openrouter", return_value="OR_WIN")
    
    result = brain._generate_with_fallback("Prompt", task_type="hunt")
    assert result == "GEMINI_WIN"
    assert mock_gemini.called
    assert not mock_or.called
    
    # Now simulate Gemini failure
    mock_gemini.side_effect = Exception("429")
    result = brain._generate_with_fallback("Prompt", task_type="hunt")
    assert result == "OR_WIN"

def test_fast_fail_simple_task_on_rate_limit(mock_brain_deps, mocker):
    """Simple/PDF tasks should not enter long fallback loops when providers return 429."""
    brain = Brain()
    brain.gemini_api_key = "fake"
    brain.openrouter_api_key = "fake"

    mocker.patch.object(
        brain,
        "_get_best_gemini_model",
        side_effect=["gemini-2.0-flash", "gemini-2.0-flash-lite", None],
    )
    mock_gemini = mocker.patch.object(
        brain,
        "_call_gemini_fallback",
        side_effect=Exception("429 Resource Exhausted"),
    )
    mock_openrouter = mocker.patch.object(
        brain,
        "_call_openrouter",
        side_effect=Exception("429 Provider rate limit"),
    )
    mock_tiered = mocker.patch.object(
        brain,
        "_call_gemini_with_tiered_fallback",
        return_value="SHOULD_NOT_BE_USED",
    )

    with pytest.raises(Exception, match="rate limit"):
        brain._generate_with_fallback("Prompt", task_type="simple")

    assert mock_gemini.call_count <= 2
    assert mock_openrouter.call_count == 1
    mock_tiered.assert_not_called()

def test_new_sentiments_support(mock_brain_deps, mocker, monkeypatch):
    """Verify system handles new 'WATCH' and 'AVOID' sentiments if returned by AI."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake_router_key")
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
    # Mock Gemini discovery to return None to force OpenRouter
    mocker.patch.object(brain, "_get_best_gemini_model", return_value=None)
    
    response_text = brain._generate_with_fallback("Prompt", json_mode=True)
    import json
    analysis_results = json.loads(response_text)
    
    assert analysis_results[0]["sentiment"] == "WATCH"
    assert analysis_results[0]["ticker"] == "AAPL"

# ============= NEW TESTS: Self-Adapting Discovery =============

def test_gemini_discovery_multi_pattern(mock_brain_deps, mocker):
    """Test Gemini discovery handles SDK API changes gracefully."""
    brain = Brain()
    
    # Mock models.list to return objects with different attribute patterns
    class MockModel1:
        name = "models/gemini-2.5-flash"
        supported_generation_methods = ["generateContent"]
    
    class MockModel2:
        model_name = "models/gemini-2.0-pro"  # Different attribute name
        # No supported_generation_methods - should still be discovered
    
    class MockModelEmbedding:
        name = "models/text-embedding-004"  # Should be filtered out

    class MockModelTTS:
        name = "models/gemini-2.5-flash-preview-tts"  # Should be filtered out
        supported_generation_methods = ["generateContent"]
        supported_response_modalities = ["AUDIO"]
        
    mock_list = mocker.patch.object(brain.gemini_client.models, "list")
    mock_list.return_value = [MockModel1(), MockModel2(), MockModelEmbedding(), MockModelTTS()]
    
    # Clear cache to force discovery
    brain._cached_best_gemini = None
    brain._cache_timestamp = 0
    
    result = brain._get_best_gemini_model()
    
    # Should discover gemini models and pick best by preference
    assert "gemini" in result
    assert "embedding" not in result
    assert "tts" not in result
    assert "audio" not in result

def test_gemini_discovery_skips_audio_only_models(mocker, monkeypatch):
    """Ensure audio-only Gemini models are excluded from text generation discovery."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key")
    mocker.patch("brain.Critic")
    mocker.patch("brain.Council")
    mock_client = MagicMock()
    mocker.patch("brain.genai.Client", return_value=mock_client)

    brain = Brain()

    class AudioOnlyModel:
        name = "models/gemini-2.5-pro-preview-tts"
        supported_generation_methods = ["generateContent"]
        supported_response_modalities = ["AUDIO"]

    class TextModel:
        name = "models/gemini-2.0-flash"
        supported_generation_methods = ["generateContent"]
        supported_response_modalities = ["TEXT"]

    brain.gemini_client.models.list.return_value = [AudioOnlyModel(), TextModel()]

    brain._cached_best_gemini = None
    brain._cache_timestamp = 0

    result = brain._get_best_gemini_model()
    assert "tts" not in result
    assert result == "gemini-2.0-flash"

def test_openrouter_discovery_multi_pattern(mock_brain_deps, mocker):
    """Test OpenRouter discovery handles API changes gracefully."""
    brain = Brain()
    brain.openrouter_api_key = "fake_key"
    
    # Clear cache
    brain._cached_best_model = None
    brain._cached_scored_candidates = None
    brain._cache_timestamp = 0
    
    # Mock response with different structure patterns
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "id": "meta-llama/llama-3.3-70b:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 65536
            },
            {
                "model": "google/gemini-test:free",  # Different key name
                "pricing": {"input": 0, "output": 0},  # Different pricing keys
                "context_window": 32000  # Different context key
            },
            {
                "id": "paid-model/test",
                "pricing": {"prompt": "0.01", "completion": "0.01"},
                "context_length": 128000
            }
        ]
    }
    mock_get.return_value = mock_response
    
    result = brain._get_best_free_model(task_type="hunt", min_context_needed=32000)
    
    # Should find free models
    assert ":free" in result or result in ["meta-llama/llama-3.3-70b:free", "google/gemini-test:free"]
    # Should not select paid model
    assert "paid-model" not in result

def test_openrouter_fallback_on_api_failure(mock_brain_deps, mocker):
    """Test OpenRouter falls back to static list on API failure."""
    brain = Brain()
    brain.openrouter_api_key = "fake_key"
    
    # Clear cache
    brain._cached_best_model = None
    brain._cache_timestamp = 0
    
    # Mock API failure
    mock_get = mocker.patch("requests.get")
    mock_get.side_effect = Exception("Connection timeout")
    
    result = brain._get_best_free_model(task_type="hunt")
    
    # Should return a static fallback model
    assert result is not None
    # Model should be from the static fallback list
    assert any(prov in result for prov in ["google/", "meta-llama/", "mistralai/"])

def test_usage_summary_deduplicates_task_and_formats_models(mock_brain_deps):
    brain = Brain()
    brain.usage_history = []

    brain._record_usage("analyze_news", "google/gemini-2.0-flash", "Google Direct")
    brain._record_usage("critic_validation", "meta-llama/llama-3.3-70b-instruct", "OpenRouter")
    brain._record_usage("analyze_news", "google/gemini-2.0-pro", "Google Direct")

    summary = brain.get_usage_summary()
    assert "Hunter (Gemini Pro)" in summary
    assert "Critic (Llama 3)" in summary
    assert "Gemini Flash" not in summary
