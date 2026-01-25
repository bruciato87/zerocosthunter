import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from api.webhook import handle_text, handle_document

@pytest.mark.asyncio
async def test_handle_text_quick_sell_qty(mocker):
    """Test entering quantity after clicking Sell button."""
    # Mock Update and Context
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "10.5"
    update.message.reply_text = AsyncMock() # Must be AsyncMock
    update.effective_chat.id = 12345
    
    context = MagicMock()
    context.user_data = {'expecting_sell_qty': 'BTC-USD'}
    
    # Mock DB and MarketData
    mocker.patch("db_handler.DBHandler")
    mock_market_class = mocker.patch("api.webhook.MarketData")
    mock_market_instance = mock_market_class.return_value
    mock_market_instance.get_smart_price_eur_async = AsyncMock(return_value=(50000.0, "BTC-USD"))
    mock_market_instance.get_smart_price_eur = MagicMock(return_value=(50000.0, "BTC-USD"))
    
    # Execute
    await handle_text(update, context)
    
    # Verify
    assert 'pending_sell' in context.user_data
    assert context.user_data['pending_sell']['quantity'] == 10.5
    assert context.user_data['pending_sell']['ticker'] == 'BTC-USD'
    assert update.message.reply_text.called
    assert "Conferma Vendita (Rapida)" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_handle_document_pdf(mocker):
    """Test receiving a Trade Republic PDF."""
    # Mock Update and Context
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    doc = MagicMock()
    doc.file_name = "order.pdf"
    update.message.document = doc
    update.effective_user.id = 123
    
    # Mock file download
    file_obj = MagicMock()
    file_obj.download_to_drive = AsyncMock()
    doc.get_file = AsyncMock(return_value=file_obj)
    
    context = MagicMock()
    context.user_data = {}
    
    # Mock Brain and PDF parser
    mock_brain = mocker.patch("brain.Brain")
    mock_brain_instance = mock_brain.return_value
    mock_brain_instance.parse_trade_republic_pdf = MagicMock(return_value={
        "ticker": "NVDA",
        "action": "BUY",
        "quantity": 5.0,
        "price": 120.0,
        "net_total": 601.0,
        "asset_name": "Nvidia"
    })
    
    # Execute
    await handle_document(update, context)
    
    # Verify
    assert 'pending_add' in context.user_data
    assert context.user_data['pending_add']['ticker'] == 'NVDA'
    assert "Conferma ACQUISTO (da PDF)" in update.message.reply_text.call_args_list[-1][0][0]

@pytest.mark.asyncio
async def test_handle_text_smart_sell(mocker):
    """Test smart text parsing for selling assets."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "Venduti 5 BTC"
    update.message.reply_text = AsyncMock()
    
    context = MagicMock()
    context.user_data = {}
    
    await handle_text(update, context)
    
    # Verify smart detection message
    assert any("Rilevata vendita" in call[0][0] for call in update.message.reply_text.call_args_list)
