import json
from pathlib import Path

from api.webhook import (
    build_observability_dashboard,
    _dashboard_fast_mode_enabled,
    _make_dashboard_health,
)


def test_build_observability_dashboard_reads_latest_reports(tmp_path):
    report_dir = Path(tmp_path)
    (report_dir / "hunt_latest.json").write_text(
        json.dumps(
            {
                "run_type": "hunt",
                "status": "success",
                "summary": "Hunt completed.",
                "completed_at": "2026-02-06T10:00:00+00:00",
                "duration_seconds": 12.3,
                "kpis": {"signals_generated": 4, "news_items_processed": 20, "signal_yield": 0.2},
                "context": {"model_used": "gemini-1.5-flash"},
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "trainml_latest.json").write_text(
        json.dumps(
            {
                "run_type": "trainml",
                "status": "error",
                "summary": "No usable champion.",
                "duration_seconds": 45.6,
                "kpis": {"promotions": 0, "rollbacks": 1, "usable_components": 1},
            }
        ),
        encoding="utf-8",
    )

    payload = build_observability_dashboard(str(report_dir))
    runs = {r["run_type"]: r for r in payload["runs"]}

    assert runs["hunt"]["status"] == "success"
    assert runs["hunt"]["exists"] is True
    assert ("signals_generated", 4) in runs["hunt"]["kpi_items"]

    assert runs["trainml"]["status"] == "error"
    assert ("rollbacks", 1) in runs["trainml"]["kpi_items"]

    assert runs["analyze"]["status"] == "missing"
    assert runs["rebalance"]["status"] == "missing"


def test_build_observability_dashboard_handles_invalid_json(tmp_path):
    report_dir = Path(tmp_path)
    (report_dir / "analyze_latest.json").write_text("{invalid-json", encoding="utf-8")

    payload = build_observability_dashboard(str(report_dir))
    runs = {r["run_type"]: r for r in payload["runs"]}

    assert runs["analyze"]["status"] == "invalid"
    assert runs["analyze"]["exists"] is False


def test_dashboard_fast_mode_default_true_on_vercel(monkeypatch):
    monkeypatch.delenv("DASHBOARD_FAST_MODE", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert _dashboard_fast_mode_enabled(force_full=False) is True


def test_dashboard_fast_mode_can_be_forced_off(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    assert _dashboard_fast_mode_enabled(force_full=True) is False


def test_make_dashboard_health_payload():
    payload = _make_dashboard_health(
        fast_mode=True,
        backend_ms=123.456,
        httpx_calls=17,
        fallbacks=["benchmark_comparison", "whale_watcher"],
        force_full=False,
    )
    assert payload["mode"] == "FAST"
    assert payload["backend_ms"] == 123.5
    assert payload["httpx_calls"] == 17
    assert payload["fallbacks_count"] == 2
    assert payload["force_full"] is False
