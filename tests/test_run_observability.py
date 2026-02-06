import json

from run_observability import RunObservability


def test_run_observability_persists_standard_files(tmp_path):
    obs = RunObservability(
        "analyze",
        dry_run=True,
        run_id="run/123",
        output_dir=str(tmp_path),
        context={"ticker": "BTC-USD"},
    )
    obs.set_kpi("news_items_fetched", 3)
    obs.add_event("fetch_news", {"count": 3})
    obs.add_error("quant_path", "simulated warning")

    report = obs.finalize(
        status="error",
        summary="Analyze failed in test.",
        kpis={"report_length_chars": 120},
        context={"market_regime": "NEUTRAL"},
    )

    assert report["run_type"] == "analyze"
    assert report["status"] == "error"
    assert report["event_count"] == 1
    assert report["kpis"]["news_items_fetched"] == 3
    assert report["kpis"]["report_length_chars"] == 120
    assert report["context"]["ticker"] == "BTC-USD"
    assert report["context"]["market_regime"] == "NEUTRAL"

    run_json = tmp_path / "analyze_run_run_123.json"
    latest_json = tmp_path / "analyze_latest.json"
    latest_md = tmp_path / "analyze_latest.md"

    assert run_json.exists()
    assert latest_json.exists()
    assert latest_md.exists()

    persisted = json.loads(run_json.read_text(encoding="utf-8"))
    assert persisted["status"] == "error"
    assert persisted["errors"][0]["stage"] == "quant_path"


def test_run_observability_finalize_is_idempotent(tmp_path):
    obs = RunObservability("rebalance", run_id="abc", output_dir=str(tmp_path))
    first = obs.finalize(status="success", summary="first")
    second = obs.finalize(status="error", summary="second")

    assert first == second
    assert second["status"] == "success"
