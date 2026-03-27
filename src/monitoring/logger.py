"""
Monitoring : logs structurés, métriques, feedback.

On loggue :
- Chaque email traité (catégorie, confiance, action, durée)
- Les escalades (pour mesurer le taux d'erreur)
- Les erreurs (pour détecter les dérives du LLM)

En prod : brancher sur Datadog, Sentry, ou ELK.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
_logger = logging.getLogger("agent")


class AgentMetrics:
    """Compteurs simples en mémoire. En prod : Prometheus / Datadog."""

    def __init__(self):
        self.total_processed = 0
        self.escalated = 0
        self.errors = 0
        self.by_category: dict[str, int] = {}

    def record(self, category: str, escalated: bool, error: bool = False) -> None:
        self.total_processed += 1
        if escalated:
            self.escalated += 1
        if error:
            self.errors += 1
        self.by_category[category] = self.by_category.get(category, 0) + 1

    @property
    def escalation_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.escalated / self.total_processed

    def summary(self) -> dict:
        return {
            "total": self.total_processed,
            "escalated": self.escalated,
            "escalation_rate": f"{self.escalation_rate:.1%}",
            "errors": self.errors,
            "by_category": self.by_category,
        }


metrics = AgentMetrics()


def log_result(result) -> None:
    """Log structuré d'un AgentResult."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "email_id": result.email_id,
        "category": result.classification.category,
        "confidence": result.classification.confidence,
        "action": result.action.action_type,
        "valid": result.validation.is_valid,
        "processing_ms": result.processing_time_ms,
        "rag_used": result.rag_context_used,
    }
    if result.action.escalation_reason:
        record["escalation_reason"] = result.action.escalation_reason

    _logger.info(json.dumps(record, ensure_ascii=False))

    metrics.record(
        category=result.classification.category,
        escalated=result.action.action_type == "escalate",
    )
