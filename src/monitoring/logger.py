"""
Monitoring : logs structurés, métriques, feedback.

On loggue :
- Chaque email traité (catégorie, confiance, action, durée)
- Les escalades (pour mesurer le taux d'erreur)
- Les erreurs non attrapées (pour détecter les dérives)
- Les warnings de validation (pour audit comptable)

En prod : brancher sur Datadog, Sentry, ou ELK.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
_logger = logging.getLogger("agent")


class AgentMetrics:
    """
    Compteurs thread-safe.
    En prod : remplacer par Prometheus / Datadog.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.total_processed = 0
        self.escalated = 0
        self.errors = 0
        self.by_category: dict[str, int] = {}

    def record(self, category: str, escalated: bool, error: bool = False) -> None:
        with self._lock:
            self.total_processed += 1
            if escalated:
                self.escalated += 1
            if error:
                self.errors += 1
            self.by_category[category] = self.by_category.get(category, 0) + 1

    @property
    def escalation_rate(self) -> float:
        with self._lock:
            if self.total_processed == 0:
                return 0.0
            return self.escalated / self.total_processed

    def summary(self) -> dict:
        with self._lock:
            return {
                "total": self.total_processed,
                "escalated": self.escalated,
                "escalation_rate": f"{self.escalated / max(self.total_processed, 1):.1%}",
                "errors": self.errors,
                "by_category": dict(self.by_category),
            }


metrics = AgentMetrics()


def log_result(result) -> None:
    """Log structuré d'un AgentResult — inclut warnings et erreurs de validation."""
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "email_id": result.email_id,
        "category": result.classification.category,
        "confidence": result.classification.confidence,
        "action": result.action.action_type,
        "valid": result.validation.is_valid,
        "processing_ms": result.processing_time_ms,
        "rag_used": result.rag_context_used,
    }

    if result.validation.errors:
        record["validation_errors"] = result.validation.errors

    if result.validation.warnings:
        record["validation_warnings"] = result.validation.warnings

    if result.action.escalation_reason:
        record["escalation_reason"] = result.action.escalation_reason

    level = logging.WARNING if result.action.action_type == "escalate" else logging.INFO
    _logger.log(level, json.dumps(record, ensure_ascii=False))

    metrics.record(
        category=result.classification.category,
        escalated=result.action.action_type == "escalate",
    )


def log_error(email_id: str, exc: Exception) -> None:
    """Log d'une erreur pipeline non attrapée."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "email_id": email_id,
        "event": "pipeline_error",
        "error_type": type(exc).__name__,
        "error_msg": str(exc),
    }
    _logger.error(json.dumps(record, ensure_ascii=False))
    metrics.record(category="error", escalated=True, error=True)
