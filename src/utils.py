"""
Utilitaires partagés : retry avec backoff exponentiel, helpers JSON.
"""
import json
import logging
import re
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import anthropic

_logger = logging.getLogger("agent")

F = TypeVar("F", bound=Callable[..., Any])

# Exceptions Anthropic qui valent la peine d'être retentées
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def with_retry(max_attempts: int = 3, base_delay: float = 2.0) -> Callable[[F], F]:
    """
    Décorateur : retente sur erreurs réseau/rate-limit avec backoff exponentiel.
    Lève l'exception d'origine après épuisement des tentatives.
    Erreurs fatales (AuthenticationError, PermissionDeniedError) → pas de retry.
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except _RETRYABLE as exc:
                    last_exc = exc
                    delay = base_delay * (2 ** (attempt - 1))
                    _logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retry in %.1fs",
                        attempt, max_attempts, type(exc).__name__, delay,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
                except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                    # Erreur fatale — inutile de retenter
                    _logger.error("LLM fatal error: %s", exc)
                    raise
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator


def safe_parse_json(raw: str) -> dict[str, Any] | None:
    """
    Extrait et parse le premier objet JSON d'une chaîne, même bruitée.
    Retourne None si aucun JSON valide trouvé.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        _logger.warning("JSON parse error: %s | raw=%r", exc, raw[:200])
        return None


def normalize_literal(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """
    Normalise une valeur LLM vers un Literal autorisé.
    Insensible à la casse. Retourne `default` si aucun match.
    """
    if not isinstance(value, str):
        return default
    v = value.strip().lower()
    for allowed_val in allowed:
        if v == allowed_val.lower():
            return allowed_val
    _logger.warning("Unexpected LLM value %r not in %s → using %r", value, allowed, default)
    return default


def safe_float(value: Any, default: float) -> float:
    """Convertit en float sans lever d'exception."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
