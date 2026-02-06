import pytest
from unittest.mock import MagicMock, patch
from brain import Brain
import asyncio

@pytest.mark.asyncio
async def test_brain_prioritizes_gemini_then_openrouter():
    """
    Verifies that the Brain tries discovered Gemini first, then OpenRouter.
    """
    # Mock DBHandler to avoid settings fetch in Brain.__init__
    with patch('db_handler.DBHandler') as mock_db:
        mock_db.return_value.get_settings.return_value = {"app_mode": "PROD"}
        
        brain = Brain()
        brain.gemini_api_key = "test_key"
        brain.gemini_client = MagicMock()
        brain.openrouter_api_key = "or_key"
        
        # Mock Gemini discovery to return a model
        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.0-flash-exp"
        mock_model.supported_generation_methods = ["generateContent"]
        brain.gemini_client.models.list.return_value = [mock_model]
        
        # Mock Gemini call to fail (429) to trigger OpenRouter
        with patch.object(brain, '_call_gemini_fallback', side_effect=Exception("429 Resource Exhausted")) as mock_gemini_call, \
             patch.object(brain, '_call_openrouter', return_value="OpenRouter Success") as mock_openrouter:
            
            result = brain._generate_with_fallback("test prompt", task_type="hunt")
            
            assert result == "OpenRouter Success"
            mock_gemini_call.assert_called()
            mock_openrouter.assert_called_once()
            print("\n✅ AI Resilience Verified: Prioritizes Gemini, then falls back to OpenRouter on 429.")

if __name__ == "__main__":
    asyncio.run(test_brain_prioritizes_gemini_then_openrouter())
