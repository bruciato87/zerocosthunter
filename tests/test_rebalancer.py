
import pytest
from rebalancer import Rebalancer
from unittest.mock import MagicMock, call

@pytest.fixture
def mock_rebalancer_deps(mocker, mock_env):
    """Mock dependencies for Rebalancer."""
    mocker.patch("db_handler.DBHandler")
    mocker.patch("brain.Brain")
    mocker.patch("critic.Critic")
    # Mock file operations if needed (e.g. logging)
    mocker.patch("builtins.open")

def test_rebalance_logic_math(mock_rebalancer_deps):
    """Test non-AI rebalancing logic (concentration, pnl)."""
    r = Rebalancer()
    r._refresh_market_context = MagicMock()
    r._get_trading_status_for_ticker = MagicMock(return_value=(True, "US", "🟢 OPEN"))
    
    analysis = {
        "total_value": 1000,
        "deviations": {"Tech": 5.0, "Crypto": -5.0},
        "assets": [
            {"ticker": "NVDA", "value": 300, "allocation": 30.0, "pnl_pct": 60.0, "sector": "Tech"}, 
            {"ticker": "BTC-USD", "value": 100, "allocation": 10.0, "pnl_pct": -35.0, "sector": "Crypto"}
        ]
    }
    
    # We strip AI suggestions by mocking _get_ai_suggestion to return None
    r._get_ai_suggestion = MagicMock(return_value=None)
    
    suggestions = r.generate_rebalance_suggestions(analysis)
    
    # Check Logic 1: Concentration Warning
    assert any("NVDA" in s and "concentrazione" in s for s in suggestions)
    
    # Check Logic 2: Profit Taking
    assert any("Take Profit" in s for s in suggestions)
    
    # Check Logic 3: Tax Loss Harvest
    assert any("Tax-Loss Harvesting" in s for s in suggestions)


def test_quant_plan_defers_orders_when_market_closed(mock_rebalancer_deps):
    r = Rebalancer()
    r.strategy_manager.get_rule = MagicMock(return_value=None)
    r.constraint_engine.MAX_TICKER_EXPOSURE = 0.20
    r.market.calculate_correlation_matrix = MagicMock(return_value={"high_correlation_pairs": []})
    r._refresh_market_context = MagicMock()
    r._get_trading_status_for_ticker = MagicMock(return_value=(False, "US", "🔴 CLOSED"))

    analysis = {
        "total_value": 10000,
        "sector_allocation": {"Technology": 80.0, "Crypto": 20.0},
        "assets": [
            {"ticker": "AAPL", "value": 8000.0, "allocation": 80.0, "sector": "Technology", "pnl_pct": 25.0, "pnl_eur": 1200.0}
        ],
    }

    plan = r._build_quant_rebalance_plan(analysis)

    assert plan == []
    assert any(step.get("ticker") == "AAPL" and step.get("side") == "SELL" for step in r._last_market_deferred_orders)


def test_generate_suggestions_uses_market_closed_notes(mock_rebalancer_deps):
    r = Rebalancer()
    r._build_quant_rebalance_plan = MagicMock(return_value=[])
    r._refresh_market_context = MagicMock()
    r._get_trading_status_for_ticker = MagicMock(return_value=(False, "US", "🔴 CLOSED"))
    r.strategy_manager.get_rule = MagicMock(return_value=None)

    analysis = {
        "total_value": 5000,
        "deviations": {"Technology": 15.0},
        "assets": [
            {"ticker": "NVDA", "value": 2000.0, "allocation": 40.0, "pnl_pct": 60.0, "sector": "Technology"}
        ],
        "sector_allocation": {"Technology": 40.0},
    }

    suggestions = r.generate_rebalance_suggestions(analysis)

    assert any("mercato chiuso" in s.lower() or "market close" in s.lower() for s in suggestions)
    assert not any(s.startswith("💰 **Take Profit**:") for s in suggestions)

def test_ai_suggestion_flow(mock_rebalancer_deps, mocker):
    """Test that Rebalancer calls Brain and Critic correctly."""
    r = Rebalancer()
    r.api_key = "fake_key"
    
    # Mock Analysis Data
    analysis = {
        "total_value": 1000,
        "assets": [{
            "ticker": "BTC-USD",
            "value": 1000,
            "allocation": 100.0,
            "pnl_pct": 10.0,
            "pnl_eur": 100.0,
            "sector": "Crypto",
            "potential_tax": 26.0,
            "qty": 0.1,
            "avg_price_eur": 9000.0
        }],
        "deviations": {},
        "sector_allocation": {"Crypto": 100.0},
        "cash": 0
    }
    
    # Mock Market Regime via Strategy Manager
    r.strategy_manager.get_market_regime = MagicMock(return_value={
        "description": "BULLISH", "confidence": 0.8, "signals": ["Signal 1"], "risk_level": "LOW"
    })
    
    # Mock Brain to return a strategy
    mock_brain = r._last_brain = MagicMock() # Mock the property if needed or just patch class
    # Rebalancer instantiates Brain() internally. We need to catch that instance.
    MockBrain = mocker.patch("brain.Brain")
    mock_brain_instance = MockBrain.return_value
    mock_brain_instance._generate_with_fallback.return_value = "🟢 BUY NVDA"
    
    # Mock Critic
    MockCritic = mocker.patch("critic.Critic")
    mock_critic_instance = MockCritic.return_value
    mock_critic_instance.critique_rebalance_strategy.return_value = "🟢 BUY NVDA (Verified)"
    
    # Run
    result = r._get_ai_suggestion(analysis)
    
    # Verify Flow
    assert result == "🟢 BUY NVDA (Verified)"
    
    # Verify Critic was called with correct arguments involving held_assets
    # Rebalancer passes [a['ticker'] for a in analysis['assets']] -> ['BTC-USD']
    calls = mock_critic_instance.critique_rebalance_strategy.call_args_list
    assert len(calls) == 1
    args, _ = calls[0]
    # Signature: strategy, regime, portfolio_val, held_assets
    assert args[0] == "🟢 BUY NVDA"
    assert args[3] == ["BTC-USD"] # CRITICAL CHECK: Held assets passed to Critic verification logic
