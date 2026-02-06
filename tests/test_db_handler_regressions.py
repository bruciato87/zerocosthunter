from unittest.mock import MagicMock, call
from types import SimpleNamespace

from db_handler import DBHandler


def _build_db_with_mocked_supabase():
    db = DBHandler.__new__(DBHandler)
    db.supabase = MagicMock()
    db._prediction_source_column = "auto"
    db._supports_extended_sentiment = None
    db._social_stats_enabled = True
    return db


def test_update_asset_quantity_applies_expected_filters():
    db = _build_db_with_mocked_supabase()
    table = MagicMock()
    table.update.return_value = table
    table.eq.return_value = table
    db.supabase.table.return_value = table

    ok = db.update_asset_quantity(chat_id=123, ticker="BTC-USD", new_quantity=2.75)

    assert ok is True
    db.supabase.table.assert_called_once_with("portfolio")
    table.update.assert_called_once_with({"quantity": 2.75})
    assert table.eq.call_args_list == [call("chat_id", 123), call("ticker", "BTC-USD")]
    table.execute.assert_called_once()


def test_update_asset_quantity_returns_false_on_db_error():
    db = _build_db_with_mocked_supabase()
    table = MagicMock()
    table.update.return_value = table
    table.eq.return_value = table
    table.execute.side_effect = RuntimeError("db failure")
    db.supabase.table.return_value = table

    ok = db.update_asset_quantity(chat_id=123, ticker="ETH-USD", new_quantity=1.0)

    assert ok is False


def test_log_prediction_falls_back_to_source_news_url_and_caches_column():
    db = _build_db_with_mocked_supabase()
    table = MagicMock()
    payloads = []

    def insert_side_effect(payload):
        payloads.append(dict(payload))
        if "source_url" in payload:
            raise RuntimeError("PGRST204: Could not find the 'source_url' column")
        query = MagicMock()
        query.execute.return_value = SimpleNamespace(data=[{"id": f"sig_{len(payloads)}"}])
        return query

    table.insert.side_effect = insert_side_effect
    db.supabase.table.return_value = table

    first = db.log_prediction(
        ticker="AAPL",
        sentiment="BUY",
        reasoning="r",
        prediction_sentence="p",
        confidence_score=0.9,
        source_url="https://example.com/news-1",
    )
    second = db.log_prediction(
        ticker="MSFT",
        sentiment="BUY",
        reasoning="r",
        prediction_sentence="p",
        confidence_score=0.8,
        source_url="https://example.com/news-2",
    )

    assert first is not None
    assert second is not None
    assert payloads[0]["source_url"] == "https://example.com/news-1"
    assert payloads[1]["source_news_url"] == "https://example.com/news-1"
    assert "source_url" not in payloads[1]
    assert payloads[2]["source_news_url"] == "https://example.com/news-2"
    assert "source_url" not in payloads[2]
    assert db._prediction_source_column == "source_news_url"


def test_log_prediction_falls_back_to_legacy_sentiment_and_caches_mode():
    db = _build_db_with_mocked_supabase()
    db._prediction_source_column = None  # keep test focused on sentiment fallback
    table = MagicMock()
    payloads = []

    def insert_side_effect(payload):
        payloads.append(dict(payload))
        if payload.get("sentiment") == "AVOID":
            raise RuntimeError('new row violates check constraint "predictions_sentiment_check"')
        query = MagicMock()
        query.execute.return_value = SimpleNamespace(data=[{"id": "sig_ok"}])
        return query

    table.insert.side_effect = insert_side_effect
    db.supabase.table.return_value = table

    first = db.log_prediction(
        ticker="COLOB",
        sentiment="AVOID",
        reasoning="r",
        prediction_sentence="p",
        confidence_score=0.65,
        source_url=None,
    )

    assert first == "sig_ok"
    assert payloads[0]["sentiment"] == "AVOID"
    assert payloads[1]["sentiment"] == "SELL"
    assert db._supports_extended_sentiment is False

    payloads.clear()
    second = db.log_prediction(
        ticker="TSLA",
        sentiment="WATCH",
        reasoning="r",
        prediction_sentence="p",
        confidence_score=0.55,
        source_url=None,
    )

    assert second == "sig_ok"
    assert len(payloads) == 1
    assert payloads[0]["sentiment"] == "HOLD"


def test_social_stats_is_disabled_after_missing_table_error():
    db = _build_db_with_mocked_supabase()
    table = MagicMock()
    table.insert.return_value = table
    table.execute.side_effect = RuntimeError("HTTP/2 404 Not Found: /rest/v1/social_stats")
    db.supabase.table.return_value = table

    db.log_social_mentions("BTC", 10)
    db.log_social_mentions("ETH", 12)

    assert db._social_stats_enabled is False
    assert table.insert.call_count == 1
    assert table.execute.call_count == 1

    history = db.get_social_history("BTC")
    assert history == []
    assert table.select.call_count == 0
