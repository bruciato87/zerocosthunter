import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from api.webhook import handle_text, handle_document

@pytest.mark.asyncio
async def test_handle_text_quick_sell_qty(mocker):
    update = MagicMock()
    update.message.text = "10.5"
    update.message.reply_text = AsyncMock() 
    update.effective_chat.id = 12345
    context = MagicMock()
    mock_db = mocker.patch("api.webhook.DBHandler").return_value
    mock_db.get_user_state.return_value = {'ticker': 'BTC-USD'}
    mocker.patch("api.webhook.MarketData").return_value.get_smart_price_eur_async = AsyncMock(return_value=(50000.0, "BTC-USD"))
    await handle_text(update, context)
    mock_db.save_user_state.assert_called()
    assert "Conferma Vendita (Rapida)" in update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_handle_document_pdf(mocker):
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = 12345
    doc = MagicMock()
    doc.file_name = "order.pdf"
    update.message.document = doc
    update.effective_user.id = 123
    doc.get_file = AsyncMock(return_value=MagicMock(download_to_drive=AsyncMock()))
    context = MagicMock()
    mock_db = mocker.patch("api.webhook.DBHandler").return_value
    mocker.patch("brain.Brain").return_value.parse_trade_republic_pdf = MagicMock(return_value={"ticker": "NVDA", "action": "BUY", "quantity": 5.0, "price": 120.0, "net_total": 601.0, "asset_name": "Nvidia"})
    await handle_document(update, context)
    mock_db.save_user_state.assert_called()
    assert "Conferma ACQUISTO (da PDF)" in update.message.reply_text.call_args_list[-1][0][0]

@pytest.mark.asyncio
async def test_handle_text_smart_sell(mocker):
    update = MagicMock()
    update.message.text = "Venduti 5 BTC"
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = 12345
    context = MagicMock()
    mock_db = mocker.patch("api.webhook.DBHandler").return_value
    mock_db.get_user_state.return_value = None
    await handle_text(update, context)
    assert any("Rilevata vendita" in call[0][0] for call in update.message.reply_text.call_args_list)
