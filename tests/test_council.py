import pytest
import asyncio
import json
from council import Council
from unittest.mock import MagicMock, patch, AsyncMock

@pytest.fixture
def brain_mock():
    mock = MagicMock()
    # Mock _generate_with_fallback to return persona JSONs
    return mock

@pytest.fixture
def council(brain_mock):
    return Council(brain_instance=brain_mock)

@pytest.mark.asyncio
async def test_council_consensus_buy(council, brain_mock):
    """Test that council reaches a BUY consensus when 2/3 agree."""
    # Personas will respond in sequence
    responses = [
        json.dumps({"sentiment": "BUY", "confidence": 0.8, "argument": "Growth is key"}),
        json.dumps({"sentiment": "HOLD", "confidence": 0.5, "argument": "Too risky"}),
        json.dumps({"sentiment": "BUY", "confidence": 0.7, "argument": "Technicals look good"})
    ]
    brain_mock._generate_with_fallback.side_effect = responses
    
    initial_signal = {"ticker": "AAPL", "sentiment": "BUY", "confidence": 0.8, "reasoning": "Bullish news"}
    verdict = await council.get_consensus("AAPL", initial_signal)
    
    assert verdict["sentiment"] == "BUY"
    assert verdict["consensus_score"] == 2
    assert "THE_BULL: BUY" in verdict["council_full_debate"]
    assert "THE_BULL: BUY" in verdict["council_summary"]

@pytest.mark.asyncio
async def test_report_consensus(brain_mock):
    """Test Council critiquing a report."""
    council = Council(brain_instance=brain_mock)
    
    brain_mock._generate_with_fallback.return_value = '{"verdict": "APPROVE", "critique": "Solid analysis"}'
    
    initial_report = "This is a bullish report on AAPL."
    result = await council.get_report_consensus("AAPL", initial_report, "Market is bullish")
    
    assert "ADVERSARIAL COUNCIL REVIEW" in result
    assert "THE_BULL" in result
    assert "Solid analysis" in result

@pytest.mark.asyncio
async def test_strategy_consensus(brain_mock):
    """Test Council debating a rebalance strategy."""
    council = Council(brain_instance=brain_mock)
    
    brain_mock._generate_with_fallback.return_value = '{"verdict": "BULLISH", "critique": "Move looks good"}'
    
    initial_strategy = "BUY BTC"
    result = await council.get_strategy_consensus("Value: 10k", initial_strategy)
    
    assert "COUNCIL STRATEGY CONSENSUS" in result
    assert "THE_QUANT" in result
    assert "Move looks good" in result

@pytest.mark.asyncio
async def test_council_disagreement(council, brain_mock):
    """Test council behavior when everyone disagrees (2/3 majority fails implies tie-break or common sense)."""
    # Actually with 3 agents, counts.most_common(1) will always return something even if it's 1 vote.
    responses = [
        json.dumps({"sentiment": "BUY", "confidence": 0.8, "argument": "Bullish"}),
        json.dumps({"sentiment": "SELL", "confidence": 0.8, "argument": "Bearish"}),
        json.dumps({"sentiment": "HOLD", "confidence": 0.5, "argument": "Neutral"})
    ]
    brain_mock._generate_with_fallback.side_effect = responses
    
    initial_signal = {"ticker": "BTC", "sentiment": "BUY", "confidence": 0.7, "reasoning": "Some news"}
    verdict = await council.get_consensus("BTC", initial_signal)
    
    # most_common(1) will pick the first one which is BUY in this sequence
    assert verdict["consensus_score"] == 1
    assert "COUNCIL DEBATE" in verdict["council_full_debate"]
