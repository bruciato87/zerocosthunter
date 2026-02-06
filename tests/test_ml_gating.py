"""Tests for TrainML walk-forward gating and champion rollback behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ml_predictor import MLPredictor


def _build_predictor():
    with patch.object(MLPredictor, "_load_model"), \
         patch.object(MLPredictor, "_load_regression_model"), \
         patch.object(MLPredictor, "_load_lstm_model"), \
         patch("db_handler.DBHandler") as mock_db:
        db_instance = mock_db.return_value
        db_instance.supabase = MagicMock()
        ml = MLPredictor()
    return ml


def test_walk_forward_slices_are_chronological():
    ml = _build_predictor()
    splits = ml._walk_forward_slices(40, max_folds=4)

    assert splits
    assert len(splits) <= 4
    for tr_slice, val_slice in splits:
        assert tr_slice.start == 0
        assert tr_slice.stop <= val_slice.start
        assert val_slice.stop > val_slice.start


def test_gate_candidate_passes_when_beating_baseline_and_champion():
    ml = _build_predictor()
    gate = ml._gate_candidate(
        model_type="classifier",
        candidate_metric=0.64,
        baseline_metric=0.60,
        champion_metric=0.62,
        min_gain=0.02,
        metric_name="accuracy",
    )

    assert gate["promoted"] is True
    assert gate["gate_status"] == "PASS"


def test_gate_candidate_rolls_back_when_gain_is_insufficient():
    ml = _build_predictor()
    gate = ml._gate_candidate(
        model_type="classifier",
        candidate_metric=0.61,
        baseline_metric=0.60,
        champion_metric=0.63,
        min_gain=0.02,
        metric_name="accuracy",
    )

    assert gate["promoted"] is False
    assert gate["gate_status"] == "ROLLED_BACK"


def test_train_classification_keeps_champion_on_gate_reject():
    ml = _build_predictor()
    ml.dry_run = True
    ml.model = MagicMock()
    ml.is_ml_ready = True
    ml.model_version = "champion_clf_v1"

    # Mock enough closed signals.
    signals = [
        {
            "ticker": f"TICK{i}",
            "pnl_percent": 1.0 if i % 2 == 0 else -1.0,
            "status": "WIN" if i % 2 == 0 else "LOSS",
            "created_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
        }
        for i in range(35)
    ]

    signal_table = MagicMock()
    signal_table.select.return_value = signal_table
    signal_table.in_.return_value = signal_table
    signal_table.order.return_value = signal_table
    signal_table.limit.return_value = signal_table
    signal_table.execute.return_value = SimpleNamespace(data=signals)
    ml.db.supabase.table.side_effect = lambda table_name: signal_table if table_name == "signal_tracking" else MagicMock()

    feature_stub = {name: 0.1 for name in ml.FEATURE_COLUMNS}

    with patch.object(ml, "_get_features", return_value=feature_stub), \
         patch.object(ml, "_get_champion_snapshot", return_value={"model_version": "champion_clf_v1", "metric": 0.65}), \
         patch.object(ml, "_gate_candidate", return_value={
             "promoted": False,
             "gate_status": "ROLLED_BACK",
             "reason": "Rejected by gate",
             "gain_vs_baseline": 0.0,
             "gap_vs_champion": -0.03,
         }), \
         patch.object(ml, "_log_model_registry_event", return_value=None):
        usable = ml._train_classification()

    report = ml.last_training_report.get("components", {}).get("classifier", {})
    assert usable is True
    assert report.get("promoted") is False
    assert report.get("rolled_back") is True
    assert report.get("usable") is True

