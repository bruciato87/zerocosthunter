import pytest
from unittest.mock import MagicMock, patch
from market_data import MarketData
import pandas as pd

@pytest.fixture
def mock_db():
    with patch('db_handler.DBHandler') as MockDB:
        db_instance = MockDB.return_value
        # Mock specific DB methods
        db_instance.get_cached_price.return_value = None
        db_instance.get_ticker_cache.return_value = None
        yield db_instance

@pytest.fixture
def market(mock_db):
    return MarketData()

@patch('market_data.MarketData.get_crypto_data_coingecko')
@patch('yfinance.Ticker')
def test_get_smart_price_eur_btc_usd(mock_ticker, mock_cg, market, mock_db):
    """
    Test that BTC-USD is correctly identified as a USD asset,
    converted to EUR, and saved to DB with currency='USD'.
    """
    # Force CG failure to test Yahoo fallback logic
    mock_cg.return_value = (None, None)

    # Setup Mock for EURUSD rate
    mock_eur_usd = MagicMock()
    mock_eur_usd.history.return_value.empty = False
    mock_eur_usd.history.return_value = pd.DataFrame({'Close': [1.10]})
    # ... (rest of setup) ...
    # Simpler: mock the internal rate fetch or the result of the Ticker call
    
    # Setup Mock for BTC-USD
    mock_btc = MagicMock()
    mock_btc.history.return_value.empty = False
    # Mocking pandas Series lookup: .iloc[-1] returns 100000
    mock_series = MagicMock()
    mock_series.iloc = [-1] 
    # Actually simpler to just mock the return of history()
    mock_btc.history.return_value = pd.DataFrame({'Close': [100000.0]})
    
    # Reset session cache and FORCE rate for deterministic math
    market._eur_usd_rate = 1.10
    
    # Configure Ticker side effects
    def ticker_side_effect(symbol):
        if symbol == "EURUSD=X":
            m = MagicMock()
            m.history.return_value = pd.DataFrame({'Close': [1.10]})
            return m
        if symbol == "BTC-USD":
            # Return new instance each time to avoid side effect leakage
            m = MagicMock()
            m.history.return_value = pd.DataFrame({'Close': [100000.0]})
            return m
        if symbol.endswith(".DE") or symbol.endswith("-EUR") or "EUR" in symbol and "USD" not in symbol: 
            # Fail EU suffixes/pairs to force USD fallback
            m = MagicMock()
            m.history.return_value = pd.DataFrame() 
            return m
        return MagicMock()

    mock_ticker.side_effect = ticker_side_effect

    # EXECUTE
    price_eur, resolved_ticker = market.get_smart_price_eur("BTC-USD")

    # VERIFY
    # 1. Price should be 100,000 / 1.10 = 90,909.09
    assert resolved_ticker == "BTC-USD"
    expected = 100000.0 / 1.10
    # Use wider tolerance or approximate
    assert abs(price_eur - expected) < 1.0, f"Got {price_eur}, expected {expected}"
    
    # 2. Verify DB Save was called with CURRENCY='USD' (Critical)
    # The fix uses `db.save_ticker_price` (not save_ticker_cache)
    
    # Inspect all calls to save_ticker_price
    calls = mock_db.save_ticker_price.call_args_list
    assert len(calls) > 0, "save_ticker_price should have been called"
    
    args, kwargs = calls[0]
    # args: (ticker, price, is_crypto, currency)
    # kwargs: currency='USD'
    
    saved_currency = kwargs.get('currency')
    if not saved_currency and len(args) > 3:
        saved_currency = args[3]
        
    assert saved_currency == 'USD', f"Expected currency='USD' but got '{saved_currency}'"

@patch('yfinance.Ticker')
def test_get_smart_price_eur_german_stock(mock_ticker, market, mock_db):
    """Test that a German stock (already EUR) is identified as EUR."""
    
    # Setup Mock for EUNL.DE with FRESH date
    import datetime
    today_str = datetime.date.today().isoformat()
    mock_stock = MagicMock()
    mock_stock.history.return_value = pd.DataFrame({'Close': [50.0]}, index=pd.date_range(end=today_str, periods=1))
    
    def ticker_side_effect(symbol):
        if symbol == "EURUSD=X":
            m = MagicMock()
            m.history.return_value = pd.DataFrame({'Close': [1.10]}, index=pd.date_range(end=today_str, periods=1))
            return m
        if symbol == "EUNL.DE":
            return mock_stock
        return MagicMock()

    mock_ticker.side_effect = ticker_side_effect
    
    price, resolved = market.get_smart_price_eur("EUNL.DE")
    
    assert resolved == "EUNL.DE"
    assert price == 50.0 # No conversion
    
    # Verify DB Save currency='EUR'
    calls = mock_db.save_ticker_price.call_args_list
    assert len(calls) > 0, "save_ticker_price should have been called"
    
    args, kwargs = calls[0]
    # In save_ticker_price(ticker, price, is_crypto, currency)
    saved_currency = kwargs.get('currency') or (args[3] if len(args) > 3 else None)
    assert saved_currency == 'EUR'

@patch('yfinance.Ticker')
def test_meta_dynamic_resolution(mock_ticker, market, mock_db):
    """Test that META resolves dynamically to available EU tickers since alias was removed."""
    
    # Setup Mock for MEW2.F (Fresh data for Today)
    import datetime
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    mock_stock_f = MagicMock()
    dates_f = pd.date_range(end=today.isoformat(), periods=1, tz="UTC")
    df_f = pd.DataFrame({'Close': [601.40]}, index=dates_f)
    mock_stock_f.history.return_value = df_f
    mock_stock_f.info = {'regularMarketChangePercent': 8.89}
    
    # Mocking datetime for morning hours (09:00 CET)
    with patch('market_data.datetime') as mock_md_datetime:
        # Mock today()
        mock_md_datetime.date.today.return_value = today
        # Mock now() for CET
        mock_now = MagicMock()
        mock_now.hour = 9
        mock_md_datetime.datetime.now.return_value = mock_now
        
        # Setup Mock for MEW2.DE (Stale data from yesterday)
        mock_stock_de = MagicMock()
        dates_de = pd.date_range(end=yesterday.isoformat(), periods=1, tz="UTC")
        df_de = pd.DataFrame({'Close': [560.90]}, index=dates_de)
        mock_stock_de.history.return_value = df_de
        
        def ticker_side_effect(symbol):
            if symbol == "EURUSD=X":
                m = MagicMock()
                m.history.return_value = pd.DataFrame({'Close': [1.10]})
                return m
            if symbol == "MEW2.F":
                return mock_stock_f
            if symbol == "MEW2.DE":
                return mock_stock_de
            return MagicMock()

        mock_ticker.side_effect = ticker_side_effect
        
        # EXECUTE
        price, resolved = market.get_smart_price_eur("META")
        
        # VERIFY: Should pick MEW2.F because it's today vs .DE which is yesterday
        assert resolved == "MEW2.F"
        assert price == 601.40

def test_price_freshness_logic(market):
    """Verify the freshness check logic manually if needed, or via integration."""
    pass
