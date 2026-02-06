import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

# We need to mock all dependencies before importing main
@pytest.fixture(autouse=True)
def mock_all_deps(mocker):
    # Mocking external modules to avoid network/DB calls
    # Patch where they are USED (main module)
    mocker.patch("main.DBHandler")
    mocker.patch("main.NewsHunter")
    mocker.patch("main.Brain")
    mocker.patch("main.MarketData")
    mocker.patch("main.TelegramNotifier")
    mocker.patch("main.Auditor")
    mocker.patch("main.Economist")
    mocker.patch("main.Advisor")
    mocker.patch("main.SignalIntelligence")
    mocker.patch("main.Sentinel")
    mocker.patch("main.PaperTrader")
    mocker.patch("main.MLPredictor")
    mocker.patch("main.SocialScraper")
    mocker.patch("main.OnChainWatcher")
    mocker.patch("main.Insider")
    mocker.patch("main.WhaleWatcher")
    mocker.patch("main.MarketRegimeClassifier")
    mocker.patch("main.Rebalancer")
    mocker.patch("main.ConsensusEngine")

@pytest.mark.asyncio
async def test_run_async_pipeline_minimal(mocker):
    """
    Test a minimal execution of run_async_pipeline.
    This should catch NameErrors and basic logic issues in the main loop.
    """
    # Fix for TypeError: sequence item 1: expected str instance, MagicMock found
    # Import main AFTER patching in the fixture to ensure it gets the mocks
    import main
    
    # Configure Mocks using the references from main
    # 0. Mock Insider
    mock_insider = main.Insider.return_value
    mock_insider.get_market_mood.return_value = {"overall": "Neutral", "crypto": {"value": "50"}}
    mock_insider.get_social_sentiment.return_value = []
    
    # 1. Mock WhaleWatcher
    mock_whale = main.WhaleWatcher.return_value
    mock_whale.analyze_flow.return_value = "Mock Whale Flow"
    
    # 2. Mock MarketRegime
    mock_regime = main.MarketRegimeClassifier.return_value
    mock_regime.classify.return_value = {"regime": "NEUTRAL", "confidence": 0.5, "recommendation": "normal"}
    
    # 3. Mock SocialScraper
    mock_social = main.SocialScraper.return_value
    mock_social.get_reddit_trending.return_value = {"BTC": 100}
    
    # 4. Mock Economist
    mock_eco = main.Economist.return_value
    mock_eco.get_macro_summary.return_value = "Mock Macro Summary"
    mock_eco.get_market_status.return_value = {"us_stocks": "🟢 Open", "eu_stocks": "🟢 Open"}
    
    # 5. Mock Auditor
    mock_auditor = main.Auditor.return_value
    mock_auditor.get_ticker_stats.return_value = {"status": "WIN", "win_rate": 80}
    mock_auditor.audit_open_signals = AsyncMock(return_value=[{"ticker": "BTC-USD", "pnl_percent": 10.0, "status": "WIN"}])
    
    # 6. Mock Advisor
    mock_adv = main.Advisor.return_value
    mock_adv.analyze_portfolio.return_value = {"total_value": 10000, "tips": ["HODL"]}
    
    # 7. Mock Rebalancer
    mock_rb = main.Rebalancer.return_value
    mock_rb.get_flash_recommendation.return_value = "Mock Flash Tip"

    # 8. Mock MarketData additional methods
    mock_market = main.MarketData.return_value
    mock_market.get_technical_summary.return_value = "Mock Technical Summary"
    mock_market.get_smart_price_eur.return_value = (50000.0, "EUR")
    mock_market.get_smart_price_eur_async = AsyncMock(return_value=(50000.0, "EUR"))
    mock_market.calculate_atr.return_value = {"atr": 1000}
    mock_market.get_multi_timeframe_trend.return_value = {"direction": "bullish", "confidence_boost": 1.05}
    
    # 8.5 Mock Sentinel
    mock_sentinel = main.Sentinel.return_value
    mock_sentinel.check_alerts = AsyncMock(return_value=[])
    
    # 9. Mock SignalIntelligence additional methods
    mock_si = main.SignalIntelligence.return_value
    mock_si.generate_context_for_ai.return_value = "Mock SI Context"
    mock_si.analyze_signal.return_value = {"adjusted_sentiment": "BUY", "adjusted_confidence": 0.8, "actions": []}
    mock_si.check_technical_confluence.return_value = {"multiplier": 1.0}
    mock_si.check_divergence.return_value = {"has_divergence": False}

    # 10. Mock MLPredictor additional methods
    mock_ml = main.MLPredictor.return_value
    mock_ml.get_confidence_modifier_from_pred.return_value = 1.0

    # 11. Mock ConsensusEngine
    mock_ce = main.ConsensusEngine.return_value
    mock_ce.calculate_weighted_action.return_value = {
        "final_action": "BUY",
        "final_score": 75.0,
        "is_disputed": False,
        "components": {"analyst": 70, "critic": 80, "council": 70, "ml": 80}
    }

    # 1. Setup minimal prediction data
    mock_prediction = {
        "ticker": "BTC-USD",
        "sentiment": "BUY",
        "reasoning": "Test reasoning",
        "confidence": 0.8,
        "source": "Test Source",
        "critic_verdict": "Pass",
        "critic_score": 90,
        "critic_reasoning": "Broker agrees"
    }
    
    # 11. Mock Brain to return our prediction
    mock_brain_instance = main.Brain.return_value
    mock_brain_instance.analyze_news_batch.return_value = [mock_prediction]
    mock_brain_instance.last_run_details = {"model": "mock-gpt"}
    
    # 12. Mock NewsHunter to return minimal news
    mock_hunter_instance = main.NewsHunter.return_value
    mock_hunter_instance.fetch_news.return_value = [{"title": "BTC News", "summary": "Bullish", "link": "http", "ticker": "BTC-USD"}]
    
    # 13. Mock DBHandler to return dummy data
    mock_db = main.DBHandler.return_value
    mock_db.get_settings.return_value = {"min_confidence": 0.5, "only_portfolio": False}
    mock_db.portfolio_map = {}
    mock_db.get_ticker_cache_batch.return_value = {}
    mock_db.log_prediction.return_value = "signal_123"
    mock_db.check_if_analyzed_recently.return_value = False
    mock_db.get_api_usage.return_value = {"date": "2026-01-25"}
    mock_db.acquire_hunt_lock.return_value = True

    # Mock Notifier
    mock_notifier = main.TelegramNotifier.return_value
    mock_notifier.send_alert = AsyncMock()
    mock_notifier.send_message = AsyncMock()
    
    # Run the pipeline
    from main import run_async_pipeline
    
    # Run the pipeline
    # We use a timeout to ensure it doesn't hang if there's an infinite loop
    try:
        await asyncio.wait_for(run_async_pipeline(), timeout=10)
    except asyncio.TimeoutError:
        pytest.fail("Pipeline timed out - possible infinite loop or deadlock")
    except NameError as e:
        pytest.fail(f"NameError detected in pipeline: {e}")
    except Exception as e:
        # We allow some exceptions if they are expected during mock setup, 
        # but NameError is what we specifically want to catch.
        if "name '" in str(e) and "' is not defined" in str(e):
             pytest.fail(f"Variable definition error: {e}")
        print(f"Caught expected/other exception: {e}")
        # pass # If we want to be lenient
        raise e

    # 7. Verify that log_prediction was called with correct arguments
    # Especially checking for the fields that caused the recent failure
    mock_db.log_prediction.assert_called()
    call_args = mock_db.log_prediction.call_args[1]
    assert call_args['critic_verdict'] == "Pass"
    assert call_args['critic_score'] == 90

    # Regression guard: run metrics must be persisted and include processed news count
    mock_db.save_run_metrics.assert_called_once()
    run_metrics = mock_db.save_run_metrics.call_args[0][0]
    assert run_metrics["news_items_processed"] == 1
