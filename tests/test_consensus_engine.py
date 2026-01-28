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
    
    result = engine.calculate_weighted_action(prediction)
    
    assert "STRONG BUY" in result["final_action"]
    assert result["final_score"] > 60
    assert result["is_disputed"] is False

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
