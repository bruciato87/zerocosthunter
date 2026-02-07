import pandas as pd
from unittest.mock import patch

from market_data import MarketData


@patch("db_handler.DBHandler")
@patch("market_data.yf.download")
def test_get_historical_data_prefers_original_symbol_before_plain_alias(
    mock_download,
    _mock_db,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    market = MarketData()

    sample = pd.DataFrame(
        {
            "Open": [1.0, 1.1, 1.2],
            "High": [1.1, 1.2, 1.3],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.05, 1.15, 1.25],
            "Volume": [100, 120, 110],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="D", name="Date"),
    )

    def download_side_effect(symbol, start=None, progress=False, auto_adjust=True):
        if symbol == "AMZN":
            return sample
        if symbol == "AMZ":
            return pd.DataFrame()
        return pd.DataFrame()

    mock_download.side_effect = download_side_effect

    df = market.get_historical_data("AMZN", days=30, force_refresh=True)

    assert not df.empty
    called_symbols = [c.args[0] for c in mock_download.call_args_list]
    assert called_symbols[0] == "AMZN"
    assert "AMZ" not in called_symbols
