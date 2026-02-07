import pytest
from consensus_engine import ConsensusEngine

def test_calculate_weighted_action_strong_buy():
    engine = ConsensusEngine()
    prediction = {
        "ticker": "AAPL",
        "sentiment": "BUY", # Hunter
        "critic_score": 90,   # Critic (+80)
        "council_summary": "UNANIMOUS VERDICT: BUY (3/3)", # Council (+70)
        "council_full_debate": "ML Prediction: UP" # ML (+80)
    }
    
    result = engine.calculate_weighted_action(prediction, is_owned=False)
    
    assert "STRONG BUY" in result["final_action"]
    assert result["final_score"] > 60
    assert result["is_disputed"] is False

def test_calculate_weighted_action_strong_accumulate_owned():
    """Verify that a STRONG BUY signal for an owned asset becomes STRONG ACCUMULATE."""
    engine = ConsensusEngine()
    prediction = {
        "ticker": "AAPL",
        "sentiment": "BUY", # Hunter
        "critic_score": 90,   # Critic (+80)
        "council_summary": "UNANIMOUS VERDICT: BUY (3/3)", # Council (+70)
        "council_full_debate": "ML Prediction: UP" # ML (+80)
    }
    
    result = engine.calculate_weighted_action(prediction, is_owned=True)
    
    assert "STRONG ACCUMULATE" in result["final_action"]

def test_calculate_weighted_action_accumulate_owned():
    """Verify that a BUY signal for an owned asset becomes ACCUMULATE."""
    engine = ConsensusEngine()
    prediction = {
        "ticker": "GOOGL",
        "sentiment": "BUY", # Hunter (+70)
        "critic_score": 60,   # Critic (+20)
        "council_summary": "MAJORITY VERDICT: BUY (2/3)", # Council (+70)
        "council_full_debate": "ML Prediction: FLAT" # ML (0)
    }
    # Score approx: (70*0.1) + (20*0.2) + (70*0.4) + (0*0.3) = 7 + 4 + 28 + 0 = 39
    
    result = engine.calculate_weighted_action(prediction, is_owned=True)
    
    assert "ACCUMULATE" in result["final_action"]
    assert "STRONG" not in result["final_action"]


def test_council_accumulate_summary_is_scored_as_bullish():
    engine = ConsensusEngine()
    prediction = {
        "ticker": "BTC-USD",
        "sentiment": "ACCUMULATE",
        "critic_score": 60,
        "council_summary": "MAJORITY VERDICT: ACCUMULATE (2/3) | OWNED_ASSET",
        "council_full_debate": "ML Prediction: FLAT"
    }

    result = engine.calculate_weighted_action(prediction, is_owned=True)
    assert result["components"]["council"] > 0
    assert "ACCUMULATE" in result["final_action"]

def test_calculate_weighted_action_disputed():
    engine = ConsensusEngine()
    prediction = {
        "ticker": "TSLA",
        "sentiment": "BUY", # Hunter (+70)
        "critic_score": 30,   # Critic (-40)
        "council_summary": "MAJORITY VERDICT: SELL (2/3)", # Council (-70)
        "council_full_debate": "ML Prediction: DOWN" # ML (-80)
    }
    
    result = engine.calculate_weighted_action(prediction, is_owned=True)
    
    assert "SELL" in result["final_action"]
    assert "Disputed" in result["final_action"]
    assert result["is_disputed"] is True
    
    # Test as non-owned
    result_ext = engine.calculate_weighted_action(prediction, is_owned=False)
    assert "AVOID" in result_ext["final_action"]

def test_calculate_weighted_action_hold():
    engine = ConsensusEngine()
    prediction = {
        "ticker": "INTC",
        "sentiment": "HOLD",
        "critic_score": 50,
        "council_summary": "DISPUTED VERDICT: HOLD (1/3)",
        "council_full_debate": ""
    }
    
    # As owned -> HOLD
    result_owned = engine.calculate_weighted_action(prediction, is_owned=True)
    assert result_owned["final_action"] == "HOLD"
    
    # As non-owned -> WATCH
    result_watch = engine.calculate_weighted_action(prediction, is_owned=False)
    assert result_watch["final_action"] == "WATCH"
    
    assert -10 < result_owned["final_score"] < 10
