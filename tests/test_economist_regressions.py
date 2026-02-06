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
