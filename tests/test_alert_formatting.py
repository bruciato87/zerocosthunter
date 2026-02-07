import pytest
from main import format_alert_msg, _split_telegram_message, _compose_telegram_messages_from_sections

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
    
    assert "🟢 **BTC** (Crypto)" in alert
    assert "⚖️ Azione consenso: **STRONG BUY**" in alert
    assert "🤖 Hunter: **BUY** (85%)" in alert
    assert "🎯 Target: €100,000 (+15.0%)" in alert
    assert "🎲 Rischio: 3/10" in alert
    assert "🛑 Stop loss: €90000.0" in alert
    assert "🛡️ Critic: Technicals are strong." in alert
    assert "🏛️ Council: VERDETTO UNANIME: BUY (3/3)" in alert
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
    
    assert "🟢 **NVDA** (Stock)" in alert
    assert "⚖️ Azione consenso: **BUY (Disputed)**" in alert
    assert "🤖 Hunter: **BUY** (75%)" in alert
    assert "🏛️ Council: VERDETTO DI MAGGIORANZA: BUY (2/3)" in alert
    assert "⚠️ Dissent: Argued for SELL because 'Valuation too high'" in alert

def test_format_alert_msg_with_ml_consensus():
    ticker = "ETH"
    sentiment = "BUY"
    confidence = 0.8
    # Simulate ML confirmed block in reasoning
    reasoning = "Catalyst: ETF inflow.\n🤖 ML Check: ✅ Confirmed (Confidence 80%)"
    source = "Coindesk"
    pred = {
        "asset_type": "Crypto",
        "target_price": "€4,000",
        "upside_percentage": 20.0,
        "risk_score": 4,
        "council_full_debate": "..."
    }
    stop_loss = 3000.0
    take_profit = 4200.0
    critic_score = 75
    critic_reasoning = "Solid."
    council_summary = "BUY"

    alert = format_alert_msg(ticker, sentiment, confidence, reasoning, source, pred, stop_loss, take_profit, critic_score, critic_reasoning, council_summary)
    
    assert "🤖 ML Check: ✅ Confirmed (Confidence 80%)" in alert

def test_format_alert_msg_with_ml_divergence():
    ticker = "SOL"
    sentiment = "BUY"
    confidence = 0.6
    # Simulate ML divergence block in reasoning
    reasoning = "Catalyst: Hype.\n🤖 ML Check: ⚠️ Divergence: DOWN (65%)"
    source = "Twitter"
    pred = {
        "asset_type": "Crypto",
        "target_price": "€150",
        "upside_percentage": 10.0,
        "risk_score": 8,
        "council_full_debate": "..."
    }
    stop_loss = 100.0
    take_profit = 180.0
    critic_score = 40
    critic_reasoning = "Risky."
    council_summary = "HOLD"

    alert = format_alert_msg(ticker, sentiment, confidence, reasoning, source, pred, stop_loss, take_profit, critic_score, critic_reasoning, council_summary)
    
    assert "🤖 ML Check: ⚠️ Divergence: DOWN (65%)" in alert

def test_split_telegram_message_preserves_all_content():
    long_text = "HEADER\n\n" + ("A" * 2500) + "\n\n" + ("B" * 2500) + "\n\nTAIL"
    parts = _split_telegram_message(long_text, max_len=1200)

    assert len(parts) > 1
    assert all(len(p) <= 1200 for p in parts)
    assert "".join(parts).replace("\n\n", "") == long_text.replace("\n\n", "")

def test_compose_messages_from_sections_preserves_order_and_boundaries():
    sections = [
        "S1\n" + ("A" * 900),
        "S2\n" + ("B" * 900),
        "S3\n" + ("C" * 900),
    ]
    parts = _compose_telegram_messages_from_sections(sections, max_len=1200)

    assert len(parts) >= 2
    assert all(len(p) <= 1200 for p in parts)
    assert parts[0].startswith("S1")
    assert "S3" in "".join(parts)
