import pytest
from main import format_alert_msg

def test_format_alert_msg_unanimous():
    ticker = "BTC"
    sentiment = "BUY"
    confidence = 0.85
    reasoning = "ETF flows are positive."
    source = "Bloomberg"
    pred = {
        "asset_type": "Crypto",
        "target_price": "€100,000",
        "upside_percentage": 15.0,
        "risk_score": 3,
        "council_full_debate": "🏛️ **COUNCIL DEBATE (UNANIMOUS)**\n- **THE_BULL**: BUY | Moon soon\n- **THE_BEAR**: BUY | Safe hedge\n- **THE_QUANT**: BUY | RSI low"
    }
    stop_loss = 90000.0
    take_profit = 110000.0
    critic_score = 80
    critic_reasoning = "Technicals are strong."
    council_summary = "UNANIMOUS VERDICT: BUY (3/3)"

    consensus_data = {"final_action": "STRONG BUY", "final_score": 85}
    alert = format_alert_msg(ticker, sentiment, confidence, reasoning, source, pred, stop_loss, take_profit, critic_score, critic_reasoning, council_summary, consensus_data=consensus_data)
    
    assert "🟢 **Signal: BTC (Crypto)**" in alert
    assert "🌟 **Consensus Action:** STRONG BUY" in alert
    assert "**Hunter Prediction:** BUY (85%)" in alert
    assert "🎯 **Target:** €100,000 (Up +15.0%)" in alert
    assert "🎲 **Risk Score:** 3/10" in alert
    assert "🛑 **Stop Loss:** €90000.0" in alert
    assert "🛡️ **Risk Check (Critic)**: Technicals are strong." in alert
    assert "🏛️ **Council Verdict**: UNANIMOUS VERDICT: BUY (3/3)" in alert
    assert "⚠️ **Dissent" not in alert

def test_format_alert_msg_majority_with_dissent():
    ticker = "NVDA"
    sentiment = "BUY"
    confidence = 0.75
    reasoning = "AI demand remains high."
    source = "Reuters"
    pred = {
        "asset_type": "Stock",
        "target_price": "€140",
        "upside_percentage": 10.0,
        "risk_score": 6,
        "council_full_debate": "🏛️ **COUNCIL DEBATE (MAJORITY)**\n- **THE_BULL**: BUY | Blackwell ramp\n- **THE_BEAR**: SELL | Valuation too high\n- **THE_QUANT**: BUY | Momentum\n⚠️ **Dissent (THE_BEAR)**: Argued for SELL because 'Valuation too high'"
    }
    stop_loss = 120.0
    take_profit = 160.0
    critic_score = 65
    critic_reasoning = "Watch for overextension."
    council_summary = "MAJORITY VERDICT: BUY (2/3)"

    consensus_data = {"final_action": "BUY (Disputed)", "final_score": 30}
    alert = format_alert_msg(ticker, sentiment, confidence, reasoning, source, pred, stop_loss, take_profit, critic_score, critic_reasoning, council_summary, consensus_data=consensus_data)
    
    assert "🟢 **Signal: NVDA (Stock)**" in alert
    assert "⚖️ **Consensus Action:** BUY (Disputed)" in alert
    assert "**Hunter Prediction:** BUY (75%)" in alert
    assert "🏛️ **Council Verdict**: MAJORITY VERDICT: BUY (2/3)" in alert
    assert "> ⚠️ **Dissent (THE_BEAR)**: Argued for SELL because 'Valuation too high'" in alert
    assert "🛡️ **Risk Check (Critic)**: Watch for overextension." in alert
