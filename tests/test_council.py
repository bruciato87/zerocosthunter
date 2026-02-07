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
    # Combined persona response in a single call
    unified_response = json.dumps({
        "THE_BULL": {"sentiment": "BUY", "confidence": 0.8, "argument": "Growth is key"},
        "THE_BEAR": {"sentiment": "HOLD", "confidence": 0.5, "argument": "Too risky"},
        "THE_QUANT": {"sentiment": "BUY", "confidence": 0.7, "argument": "Technicals look good"}
    })
    brain_mock._generate_with_fallback.return_value = unified_response
    
    initial_signal = {"ticker": "AAPL", "sentiment": "BUY", "confidence": 0.8, "reasoning": "Bullish news"}
    verdict = await council.get_consensus("AAPL", initial_signal)
    
    assert verdict["sentiment"] == "BUY"
    assert verdict["consensus_score"] == 2
    assert "**THE_BULL**: BUY" in verdict["council_full_debate"]
    assert "MAJORITY VERDICT: BUY (2/3)" in verdict["council_summary"]


@pytest.mark.asyncio
async def test_council_owned_asset_maps_buy_to_accumulate(council, brain_mock):
    unified_response = json.dumps({
        "THE_BULL": {"sentiment": "BUY", "confidence": 0.8, "argument": "Trend forte"},
        "THE_BEAR": {"sentiment": "HOLD", "confidence": 0.5, "argument": "Rischio medio"},
        "THE_QUANT": {"sentiment": "BUY", "confidence": 0.7, "argument": "Confluenza positiva"}
    })
    brain_mock._generate_with_fallback.return_value = unified_response

    initial_signal = {"ticker": "BTC-USD", "sentiment": "BUY", "confidence": 0.82, "reasoning": "Setup bullish"}
    portfolio_context = [{"ticker": "BTC-USD", "quantity": 0.2, "avg_price": 50000}]
    verdict = await council.get_consensus("BTC-USD", initial_signal, portfolio_context=portfolio_context)

    assert verdict["sentiment"] == "ACCUMULATE"
    assert "ACCUMULATE" in verdict["council_summary"]
    assert "Ownership Context" in verdict["council_full_debate"]

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
    """Test council behavior when everyone disagrees."""
    unified_response = json.dumps({
        "THE_BULL": {"sentiment": "BUY", "confidence": 0.8, "argument": "Bullish"},
        "THE_BEAR": {"sentiment": "SELL", "confidence": 0.8, "argument": "Bearish"},
        "THE_QUANT": {"sentiment": "HOLD", "confidence": 0.5, "argument": "Neutral"}
    })
    brain_mock._generate_with_fallback.return_value = unified_response
    
    initial_signal = {"ticker": "BTC", "sentiment": "BUY", "confidence": 0.7, "reasoning": "Some news"}
    verdict = await council.get_consensus("BTC", initial_signal)
    
    # most_common(1) will pick the first one which is BUY in this sequence
    assert verdict["consensus_score"] == 1
    assert "COUNCIL DEBATE" in verdict["council_full_debate"]

@pytest.mark.asyncio
async def test_council_preserves_metadata(council, brain_mock):
    """Test that council does not drop Critic and other metadata."""
    unified_response = json.dumps({
        "THE_BULL": {"sentiment": "BUY", "confidence": 0.8, "argument": "G"},
        "THE_BEAR": {"sentiment": "BUY", "confidence": 0.8, "argument": "G"},
        "THE_QUANT": {"sentiment": "BUY", "confidence": 0.8, "argument": "G"}
    })
    brain_mock._generate_with_fallback.return_value = unified_response
    
    initial_signal = {
        "ticker": "METADATA_TEST", 
        "sentiment": "BUY", 
        "critic_reasoning": "IMPORTANT_CRITIC_NOTE",
        "stop_loss": 100.0
    }
    verdict = await council.get_consensus("METADATA_TEST", initial_signal)
    
    assert verdict["critic_reasoning"] == "IMPORTANT_CRITIC_NOTE"
    assert verdict["stop_loss"] == 100.0
    assert verdict["ticker"] == "METADATA_TEST"
