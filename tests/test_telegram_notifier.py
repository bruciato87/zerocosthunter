"""Tests for Telegram notifier HTML-first delivery and fallback behavior."""

import asyncio
from unittest.mock import patch

from telegram_bot import TelegramNotifier


class _DummyBot:
    def __init__(self, fail_html: bool = False):
        self.fail_html = fail_html
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send_message(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_html and kwargs.get("parse_mode") == "HTML":
            raise RuntimeError("telegram 400")
        return {"ok": True}


def test_markdownish_to_html_conversion():
    raw = "**Training** `pure_gb_v1`"
    html = TelegramNotifier._markdownish_to_html(raw)
    assert "<b>Training</b>" in html
    assert "<code>pure_gb_v1</code>" in html


def test_send_message_uses_html_first(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    notifier = TelegramNotifier()
    bot = _DummyBot(fail_html=False)

    with patch("telegram_bot.Bot", return_value=bot):
        asyncio.run(notifier.send_message("42", "**Hi** `there`"))

    assert len(bot.calls) == 1
    assert bot.calls[0].get("parse_mode") == "HTML"
    assert "<b>Hi</b>" in bot.calls[0].get("text", "")


def test_send_message_falls_back_to_plain_text_on_html_error(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:token")
    notifier = TelegramNotifier()
    bot = _DummyBot(fail_html=True)

    with patch("telegram_bot.Bot", return_value=bot):
        asyncio.run(notifier.send_message("42", "**Hi** `there`"))

    assert len(bot.calls) == 2
    assert bot.calls[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in bot.calls[1]
    assert bot.calls[1].get("text") == "Hi there"

