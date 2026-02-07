from ticker_resolver import is_probable_ticker


def test_is_probable_ticker_rejects_known_news_noise_tokens():
    assert not is_probable_ticker("DELHI")
    assert not is_probable_ticker("DETAILS")
    assert not is_probable_ticker("TARIFF")
    assert not is_probable_ticker("NATO")
    assert not is_probable_ticker("CNY")


def test_is_probable_ticker_keeps_known_valid_assets():
    assert is_probable_ticker("RENDER")
    assert is_probable_ticker("BYDDF")
    assert is_probable_ticker("AAPL")
    assert is_probable_ticker("0700.HK")


def test_is_probable_ticker_rejects_unknown_long_plain_words():
    assert not is_probable_ticker("MEGACORP")
