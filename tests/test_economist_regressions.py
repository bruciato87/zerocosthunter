from types import SimpleNamespace
import pandas as pd

from economist import Economist


def test_fear_greed_snapshot_accepts_classification_key(mocker):
    eco = Economist()

    class InsiderStub:
        def get_crypto_fear_greed(self):
            return {"value": "42", "classification": "FEAR"}

    mocker.patch("insider.Insider", return_value=InsiderStub())

    fg = eco._get_fear_greed_snapshot()

    assert fg == {"value": 42, "label": "FEAR"}


def test_fear_greed_snapshot_supports_legacy_get_fear_greed(mocker):
    eco = Economist()
    insider_stub = SimpleNamespace(get_fear_greed=lambda: {"value": 55, "label": "Neutral"})
    mocker.patch("insider.Insider", return_value=insider_stub)

    fg = eco._get_fear_greed_snapshot()

    assert fg == {"value": 55, "label": "Neutral"}


def test_safe_last_close_returns_none_on_fetch_error(mocker):
    eco = Economist()
    mocker.patch("economist.yf.Ticker", side_effect=RuntimeError("network down"))

    value = eco._safe_last_close("^VIX", "1d", "VIX")

    assert value is None


def test_safe_last_close_returns_latest_value(mocker):
    eco = Economist()

    class TickerStub:
        def history(self, period):
            return pd.DataFrame({"Close": [18.2, 19.7, 21.4]})

    mocker.patch("economist.yf.Ticker", return_value=TickerStub())

    value = eco._safe_last_close("^VIX", "1d", "VIX")

    assert value == 21.4


def test_classify_market_for_ticker_uses_resolution_metadata():
    eco = Economist()

    assert eco.classify_market_for_ticker("3XC", resolved_ticker="3CP.F", currency="EUR") == "EU"
    assert eco.classify_market_for_ticker("BTC-USD") == "CRYPTO"
    assert eco.classify_market_for_ticker("AAPL") == "US"


def test_get_trading_status_for_ticker_respects_snapshot():
    eco = Economist()
    snapshot = {
        "us_stocks": "🔴 CLOSED",
        "eu_stocks": "🟢 OPEN",
        "crypto": "OPEN (24/7)",
    }

    us_open, us_bucket, us_label = eco.get_trading_status_for_ticker("AAPL", market_status=snapshot)
    eu_open, eu_bucket, eu_label = eco.get_trading_status_for_ticker("EUNL.DE", market_status=snapshot)
    cr_open, cr_bucket, cr_label = eco.get_trading_status_for_ticker("BTC-USD", market_status=snapshot)

    assert us_bucket == "US"
    assert us_open is False
    assert "CLOSED" in us_label

    assert eu_bucket == "EU"
    assert eu_open is True
    assert "OPEN" in eu_label

    assert cr_bucket == "CRYPTO"
    assert cr_open is True
    assert "OPEN" in cr_label
