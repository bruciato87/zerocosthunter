from datetime import datetime, timezone

from main import _safe_parse_published_datetime, _safe_total_tokens_from_usage


def test_safe_parse_published_datetime_normalizes_naive_datetime():
    raw = {"published_datetime": datetime(2026, 2, 6, 12, 30, 0)}

    parsed = _safe_parse_published_datetime(raw)

    assert parsed is not None
    assert parsed.tzinfo == timezone.utc


def test_safe_parse_published_datetime_parses_string_with_timezone():
    raw = {"published": "2026-02-06T12:30:00+01:00"}

    parsed = _safe_parse_published_datetime(raw)

    assert parsed is not None
    assert parsed.tzinfo is not None


def test_safe_parse_published_datetime_returns_none_on_invalid_value():
    raw = {"published": "not-a-date"}

    parsed = _safe_parse_published_datetime(raw)

    assert parsed is None


def test_safe_total_tokens_from_usage_handles_common_shapes():
    assert _safe_total_tokens_from_usage({"total_tokens": "?"}) == "Direct"
    assert _safe_total_tokens_from_usage({"total_tokens": "FAILED_429"}) == "Exhausted (429)"
    assert _safe_total_tokens_from_usage({"total_tokens": 1234}) == "1234"
    assert _safe_total_tokens_from_usage("raw-usage") == "raw-usage"
