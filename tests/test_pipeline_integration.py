import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

# We need to mock all dependencies before importing main
@pytest.fixture(autouse=True)
def mock_all_deps(mocker):
    # Mocking external modules to avoid network/DB calls
    mocker.patch("db_handler.DBHandler")
    mocker.patch("hunter.NewsHunter")
    mocker.patch("brain.Brain")
    mocker.patch("market_data.MarketData")
    mocker.patch("telegram_bot.TelegramNotifier")
    mocker.patch("auditor.Auditor")
    mocker.patch("economist.Economist")
    mocker.patch("advisor.Advisor")
    mocker.patch("signal_intelligence.SignalIntelligence")
    mocker.patch("sentinel.Sentinel")
    mocker.patch("paper_trader.PaperTrader")
    mocker.patch("ml_predictor.MLPredictor")
    mocker.patch("social_scraper.SocialScraper")
    mocker.patch("onchain_watcher.OnChainWatcher")
    mocker.patch("insider.Insider")
    mocker.patch("whale_watcher.WhaleWatcher")
    mocker.patch("market_regime.MarketRegimeClassifier")
    mocker.patch("rebalancer.Rebalancer")

@pytest.mark.asyncio
async def test_run_async_pipeline_minimal(mocker):
    """
    Test a minimal execution of run_async_pipeline.
    This should catch NameErrors and basic logic issues in the main loop.
    """
    # Fix for TypeError: sequence item 1: expected str instance, MagicMock found
    # Mock Insider
    import insider
    mock_insider = insider.Insider.return_value
    mock_insider.get_market_mood.return_value = {"overall": "Neutral", "crypto": {"value": "50"}}
    mock_insider.get_social_sentiment.return_value = []
    
    # Mock WhaleWatcher
    import whale_watcher
    mock_whale = whale_watcher.WhaleWatcher.return_value
    mock_whale.analyze_flow.return_value = "Mock Whale Flow"
    
    # Mock MarketRegime
    import market_regime
    mock_regime = market_regime.MarketRegimeClassifier.return_value
    mock_regime.classify.return_value = {"regime": "NEUTRAL", "confidence": 0.5, "recommendation": "normal"}
    
    # Mock SocialScraper
    import social_scraper
    mock_social = social_scraper.SocialScraper.return_value
    mock_social.get_reddit_trending.return_value = {"BTC": 100}
    
    # Mock Economist
    import economist
    mock_eco = economist.Economist.return_value
    mock_eco.get_macro_summary.return_value = "Mock Macro Summary"
    mock_eco.get_market_status.return_value = {"us_stocks": "🟢 Open", "eu_stocks": "🟢 Open"}
    
    # Mock Auditor
    import auditor
    mock_auditor = auditor.Auditor.return_value
    mock_auditor.get_ticker_stats.return_value = {"status": "WIN", "win_rate": 80}
    mock_auditor.audit_open_signals.return_value = ["Audit Result 1"]
    
    # Mock Advisor
    import advisor
    mock_adv = advisor.Advisor.return_value
    mock_adv.analyze_portfolio.return_value = {"total_value": 10000, "tips": ["HODL"]}
    
    # Mock Rebalancer
    import rebalancer
    mock_rb = rebalancer.Rebalancer.return_value
    mock_rb.get_flash_recommendation.return_value = "Mock Flash Tip"

    # Mock MarketData additional methods
    import market_data
    mock_market = market_data.MarketData.return_value
    mock_market.get_technical_summary.return_value = "Mock Technical Summary"
    
    # Mock SignalIntelligence additional methods
    import signal_intelligence
    mock_si = signal_intelligence.SignalIntelligence.return_value
    mock_si.generate_context_for_ai.return_value = "Mock SI Context"

    # Mock MLPredictor additional methods
    import ml_predictor
    mock_ml = ml_predictor.MLPredictor.return_value
    mock_ml.get_confidence_modifier_from_pred.return_value = 1.0

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
    
    # 2. Mock Brain to return our prediction
    import brain
    mock_brain_instance = brain.Brain.return_value
    mock_brain_instance.analyze_news_batch.return_value = [mock_prediction]
    mock_brain_instance.last_run_details = {"model": "mock-gpt"}
    
    # 3. Mock NewsHunter to return minimal news
    import hunter
    mock_hunter_instance = hunter.NewsHunter.return_value
    mock_hunter_instance.fetch_news.return_value = [{"title": "BTC News", "summary": "Bullish", "link": "http", "ticker": "BTC-USD"}]
    
    # 4. Mock DBHandler to return dummy data
    import db_handler
    mock_db = db_handler.DBHandler.return_value
    mock_db.get_settings.return_value = {"min_confidence": 0.5, "only_portfolio": False}
    mock_db.portfolio_map = {}
    mock_db.get_ticker_cache_batch.return_value = {}
    mock_db.log_prediction.return_value = "signal_123"
    mock_db.check_if_analyzed_recently.return_value = False
    mock_db.get_api_usage.return_value = {"date": "2026-01-25"}
    mock_db.acquire_hunt_lock.return_value = True
    
    # 5. Mock MarketData
    import market_data
    mock_market = market_data.MarketData.return_value
    mock_market.get_smart_price_eur.return_value = (50000.0, "EUR")
    mock_market.calculate_atr.return_value = {"atr": 1000}
    mock_market.get_multi_timeframe_trend.return_value = {"direction": "bullish", "confidence_boost": 1.05}
    
    # 6. Mock Notifier
    import telegram_bot
    mock_notifier = telegram_bot.TelegramNotifier.return_value
    mock_notifier.send_alert = AsyncMock()
    
    # Additional SI Mocks
    mock_si.analyze_signal.return_value = {"adjusted_sentiment": "BUY", "adjusted_confidence": 0.8, "actions": []}
    mock_si.check_technical_confluence.return_value = {"multiplier": 1.0}
    mock_si.check_divergence.return_value = {"has_divergence": False}
    
    # Import run_async_pipeline after mocking
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
