import pytest
from unittest.mock import MagicMock, patch
from main import run_async_pipeline as main
import asyncio

@pytest.mark.asyncio
async def test_analysis_pipeline_includes_synthetic_items():
    """
    Regression Test: Ensures that even if RSS feed is empty, 
    synthetic portfolio items are added to the analysis batch.
    """
    # Mocking all external dependencies
    mock_db = MagicMock()
    mock_db.get_settings.return_value = {"last_successful_hunt_ts": "2020-01-01T00:00:00Z"}
    mock_db.get_portfolio_map.return_value = {"BTC-USD": {"quantity": 1, "avg_price": 50000}}
    
    mock_hunter = MagicMock()
    mock_hunter.fetch_news.return_value = [] # EMPTY RSS
    
    mock_market = MagicMock()
    mock_market.get_smart_price_eur.return_value = (60000.0, "USD")
    mock_market.get_technical_summary.return_value = "Bullish"
    
    mock_brain = MagicMock()
    # Capture the news items passed to analyze_news_batch
    captured_batch = []
    def side_effect(batch, **kwargs):
        nonlocal captured_batch
        captured_batch = batch
        return []

    mock_brain.analyze_news_batch.side_effect = side_effect
    
    from unittest.mock import AsyncMock
    mock_sentinel = MagicMock()
    mock_sentinel.check_alerts = AsyncMock(return_value=[])
    mock_sentinel.get_strategic_forecast = MagicMock(return_value={"regime": "NEUTRAL", "recommendation": "normal"})

    with patch('main.DBHandler', return_value=mock_db), \
         patch('main.NewsHunter', return_value=mock_hunter), \
         patch('main.MarketData', return_value=mock_market), \
         patch('main.Brain', return_value=mock_brain), \
         patch('main.TelegramNotifier', return_value=MagicMock(send_message=AsyncMock())), \
         patch('main.WhaleWatcher', return_value=MagicMock(analyze_flow=MagicMock(return_value=""))), \
         patch('main.PulseHunter', return_value=MagicMock(scan=MagicMock(return_value=[]))), \
         patch('main.Sentinel', return_value=mock_sentinel), \
         patch('main.Economist', return_value=MagicMock(get_macro_summary=MagicMock(return_value=""), get_market_status=MagicMock(return_value={"us_stocks": "🟢", "eu_stocks": "🟢"}))), \
         patch('main.Insider', return_value=MagicMock(get_market_mood=MagicMock(return_value={}), get_social_sentiment=MagicMock(return_value=[]))), \
         patch('main.Auditor', return_value=MagicMock(get_ticker_stats=MagicMock(return_value=None))), \
         patch('main.Advisor', return_value=MagicMock(analyze_portfolio=MagicMock(return_value={}))), \
         patch('main.SignalIntelligence', return_value=MagicMock(generate_context_for_ai=MagicMock(return_value=""))):
        
        # Run main in a controlled way (or a slice of it if possible)
        # Since main() is a massive function, we might need a slice or just run it and expect it to finish
        try:
            await main()
        except SystemExit:
            pass
        except Exception as e:
            # We expect it might fail later due to other mocks, but we check captured_batch
            print(f"Main execution trace: {e}")

    # VERIFICATION
    assert len(captured_batch) > 0, "Analysis batch should NOT be empty even with zero RSS news"
    assert any(item.get('synthetic') for item in captured_batch), "Analysis batch should contain a synthetic portfolio item"
    assert any("BTC-USD" in item.get('ticker', '') for item in captured_batch), "Batch should contain BTC-USD synthetic check"

if __name__ == "__main__":
    asyncio.run(test_analysis_pipeline_includes_synthetic_items())
