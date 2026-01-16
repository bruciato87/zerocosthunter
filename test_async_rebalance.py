
import os
import asyncio
import logging
from unittest.mock import MagicMock, patch

# Mock dependencies BEFORE importing rebalancer
with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key", "SUPABASE_URL": "https://fake.supabase.co", "SUPABASE_KEY": "fake_key"}):
    with patch("db_handler.DBHandler") as MockDB:
        with patch("market_data.MarketData") as MockMarket:
            with patch("advisor.Advisor") as MockAdvisor:
                from rebalancer import Rebalancer

# Configure Mock Rebalancer
def test_async_worker():
    print("🧪 Testing Async Rebalancer Worker...")
    
    # Setup Mocks
    mock_db = MagicMock()
    mock_db.get_portfolio.return_value = [
        {"ticker": "BTC-USD", "quantity": 0.5, "avg_price": 50000}
    ]
    
    # Mock Market Data to avoid API calls
    mock_market = MagicMock()
    mock_market.get_smart_price_eur.return_value = (60000, "live")
    mock_market.get_technical_summary.return_value = "RSI: 50"
    
    # Initialize Rebalancer with mocks
    with patch("db_handler.DBHandler", return_value=mock_db), \
         patch("market_data.MarketData", return_value=mock_market), \
         patch("advisor.Advisor", MagicMock()):
        
        rebalancer = Rebalancer()
        
        # Override internal dependencies just in case
        rebalancer.db = mock_db
        rebalancer.market = mock_market
        
        # Mock Telegram Notifier
        with patch("telegram_bot.TelegramNotifier") as MockNotifier:
            mock_notifier_instance = MockNotifier.return_value
            mock_notifier_instance.send_message = MagicMock()
            mock_notifier_instance.send_alert = MagicMock()
            
            # --- TEST CASE 1: Targeted Execution ---
            target_chat_id = "123456789"
            os.environ["TARGET_CHAT_ID"] = target_chat_id
            
            print(f"👉 Running run_daily with TARGET_CHAT_ID={target_chat_id}")
            asyncio.run(rebalancer.run_daily())
            
            # Verify send_message was called with correct ID
            if mock_notifier_instance.send_message.called:
                args = mock_notifier_instance.send_message.call_args
                if str(args.kwargs.get('chat_id')) == target_chat_id:
                    print("✅ PASS: Targeted message sent to correct Chat ID.")
                else:
                    print(f"❌ FAIL: Wrong Chat ID. Expected {target_chat_id}, got {args.kwargs.get('chat_id')}")
            else:
                print("❌ FAIL: send_message NOT called.")

            # --- TEST CASE 2: Broadcast Execution ---
            del os.environ["TARGET_CHAT_ID"]
            print("\n👉 Running run_daily with NO Target (Broadcast)")
            asyncio.run(rebalancer.run_daily())
            
            # Verify send_alert was called
            if mock_notifier_instance.send_alert.called:
                print("✅ PASS: Broadcast alert sent.")
            else:
                print("❌ FAIL: send_alert NOT called.")

if __name__ == "__main__":
    test_async_worker()
