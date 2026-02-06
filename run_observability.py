"""
Run observability helpers.

Standardizes run tracing/reporting across hunt, analyze, rebalance and trainml.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RunObservability")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _safe_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


class RunObservability:
    """Collects KPI/context/errors and emits standardized run reports."""

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        run_type: str,
        *,
        dry_run: bool = False,
        run_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.run_type = (run_type or "unknown").strip().lower()
        self.dry_run = bool(dry_run)
        self.run_id = str(run_id or os.environ.get("GITHUB_RUN_ID") or f"{self.run_type}_{int(time.time())}")
        self.started_at = _utc_now_iso()
        self._started_perf = time.perf_counter()
        self._kpis: Dict[str, Any] = {}
        self._context: Dict[str, Any] = dict(context or {})
        self._errors: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._finalized = False
        self._report: Dict[str, Any] = {}
        default_output_dir = "/tmp/zero_cost_hunter_run_reports" if os.environ.get("PYTEST_CURRENT_TEST") else "run_logs/latest"
        self.output_dir = Path(output_dir or os.environ.get("RUN_REPORT_DIR", default_output_dir))

        logger.info(
            "RUN_TRACE_START type=%s run_id=%s dry_run=%s",
            self.run_type,
            self.run_id,
            self.dry_run,
        )

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = _json_safe(value)

    def set_kpi(self, key: str, value: Any) -> None:
        self._kpis[key] = _json_safe(value)

    def set_kpis(self, metrics: Optional[Dict[str, Any]]) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            self.set_kpi(key, value)

    def add_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._events.append(
            {
                "ts": _utc_now_iso(),
                "name": str(name),
                "payload": _json_safe(payload or {}),
            }
        )

    def add_error(self, stage: str, err: Any) -> None:
        self._errors.append(
            {
                "ts": _utc_now_iso(),
                "stage": str(stage),
                "error": str(err),
            }
        )

    def finalize(
        self,
        *,
        status: str,
        summary: str = "",
        kpis: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._finalized:
            return self._report

        self.set_kpis(kpis)
        if context:
            for key, value in context.items():
                self.set_context(key, value)

        completed_at = _utc_now_iso()
        duration_seconds = round(time.perf_counter() - self._started_perf, 3)

        report: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "run_type": self.run_type,
            "run_id": self.run_id,
            "status": status,
            "summary": summary or "",
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "completed_at": completed_at,
            "duration_seconds": duration_seconds,
            "kpis": _json_safe(self._kpis),
            "context": _json_safe(self._context),
            "errors": _json_safe(self._errors),
            "event_count": len(self._events),
        }
        if self._events:
            report["events"] = _json_safe(self._events)

        self._report = report
        self._finalized = True
        self._persist_report(report)

        try:
            logger.info("RUN_REPORT_JSON %s", json.dumps(report, sort_keys=True, ensure_ascii=False))
        except Exception:
            logger.info("RUN_REPORT_JSON %s", report)

        return report

    def _persist_report(self, report: Dict[str, Any]) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            run_token = _safe_token(self.run_id)
            base = f"{self.run_type}_run_{run_token}"
            run_json_path = self.output_dir / f"{base}.json"
            run_md_path = self.output_dir / f"{base}.md"
            latest_json_path = self.output_dir / f"{self.run_type}_latest.json"
            latest_md_path = self.output_dir / f"{self.run_type}_latest.md"

            serialized = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
            markdown = self._render_markdown(report)

            run_json_path.write_text(serialized + "\n", encoding="utf-8")
            run_md_path.write_text(markdown + "\n", encoding="utf-8")
            latest_json_path.write_text(serialized + "\n", encoding="utf-8")
            latest_md_path.write_text(markdown + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning("Run report persistence failed for %s: %s", self.run_type, e)

    def _render_markdown(self, report: Dict[str, Any]) -> str:
        lines: List[str] = [
            f"# Run Report: {report.get('run_type', 'unknown')}",
            "",
            f"- Run ID: `{report.get('run_id', '')}`",
            f"- Status: `{report.get('status', 'unknown')}`",
            f"- Dry Run: `{report.get('dry_run', False)}`",
            f"- Started At (UTC): `{report.get('started_at', '')}`",
            f"- Completed At (UTC): `{report.get('completed_at', '')}`",
            f"- Duration (s): `{report.get('duration_seconds', 0)}`",
        ]

        summary = (report.get("summary") or "").strip()
        if summary:
            lines.extend(["", "## Summary", "", summary])

        context = report.get("context") or {}
        if context:
            lines.extend(["", "## Context"])
            for key in sorted(context.keys()):
                lines.append(f"- {key}: `{context[key]}`")

        kpis = report.get("kpis") or {}
        if kpis:
            lines.extend(["", "## KPIs"])
            for key in sorted(kpis.keys()):
                lines.append(f"- {key}: `{kpis[key]}`")

        errors = report.get("errors") or []
        if errors:
            lines.extend(["", "## Errors"])
            for item in errors:
                stage = item.get("stage", "unknown")
                err = item.get("error", "")
                lines.append(f"- [{stage}] {err}")

        return "\n".join(lines)
