
import pytest
from unittest.mock import MagicMock
from brain import Brain

def test_usage_tracking_and_summary():
    brain = Brain()
    
    # 1. Simulate usage recording
    brain._record_usage("hunt", "google/gemini-2.0-flash-exp:free", "OpenRouter")
    brain._record_usage("critic_eval", "meta-llama/llama-3.3-70b-instruct:free", "OpenRouter")
    brain._record_usage("council_debate", "deepseek/deepseek-chat:free", "OpenRouter")
    
    # 2. Get Summary
    summary = brain.get_usage_summary()
    print(f"Generated Summary: {summary}")
    
    # 3. Assertions
    assert "Gemini" in summary
    assert "Llama 3" in summary
    assert "DeepSeek" in summary
    assert "Hunter" in summary
    assert "Critic" in summary
    assert "Council" in summary
    
    # Check format
    assert "🤖 AI:" in summary
    assert "|" in summary

def test_usage_deduplication():
    brain = Brain()
    
    # Simulate retries (should keep latest)
    brain._record_usage("hunt", "google/gemini-flash-1.5", "Google Direct")
    brain._record_usage("hunt", "google/gemini-2.0-pro-exp-02-05:free", "OpenRouter") # Winner
    
    summary = brain.get_usage_summary()
    print(f"Dedupe Summary: {summary}")
    
    # Should show the last one used for "hunt"
    assert "Gemini Pro" in summary
    assert "Gemini Flash" not in summary
