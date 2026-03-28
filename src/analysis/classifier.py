"""
Classification des emails.

STRATÉGIE :
1. Règles rapides non-LLM (mots-clés évidents) → décision immédiate
2. Si ambigu → LLM (Claude) avec sortie JSON structurée
3. Si confidence < seuil → escalade humaine

Pourquoi ce mix ?
- Les règles métier couvrent 40-60% des cas courants de façon déterministe.
- Le LLM gère la nuance et les cas complexes.
- On évite des appels LLM inutiles et coûteux.
"""
import logging
import os

import anthropic

from src.models import Classification
from src.utils import normalize_literal, safe_float, safe_parse_json, with_retry

_logger = logging.getLogger("agent")

_CATEGORY_VALUES = (
    "facture",
    "devis",
    "question_comptable",
    "document_upload",
    "relance",
    "autre",
)

_KEYWORD_RULES: dict[str, list[str]] = {
    "facture": ["facture", "invoice", "règlement", "paiement dû", "à payer"],
    "devis": ["devis", "estimation", "proposition commerciale", "offre de prix"],
    "relance": ["relance", "sans réponse", "toujours en attente", "rappel"],
    "document_upload": ["ci-joint", "veuillez trouver", "en pièce jointe", "je vous transmets"],
}

_SYSTEM_PROMPT = """Tu es un assistant comptable expert chez Dougs.
Analyse l'email et retourne UNIQUEMENT un JSON valide avec ces champs :
{
  "category": "<facture|devis|question_comptable|document_upload|relance|autre>",
  "confidence": <float entre 0.0 et 1.0>,
  "reasoning": "<explication courte en français>"
}
Ne génère rien d'autre que ce JSON."""


def _apply_keyword_rules(text: str) -> Classification | None:
    """Règles métier rapides sans LLM. Retourne None si ambigu."""
    text_lower = text.lower()
    matches: dict[str, int] = {}
    for category, keywords in _KEYWORD_RULES.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            matches[category] = count

    if not matches:
        return None
    if len(matches) > 1:
        return None  # ambigu → laisser le LLM décider

    category = max(matches, key=matches.__getitem__)
    return Classification(
        category=category,
        confidence=0.85,
        reasoning=f"Règle mots-clés détectée ({matches[category]} occurrence(s))",
    )


@with_retry(max_attempts=3, base_delay=2.0)
def _call_llm(email_text: str) -> str:
    """Appel LLM isolé pour permettre le retry."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": email_text[:3000]}],
    )
    return message.content[0].text.strip()


def classify_email(email_text: str) -> Classification:
    """
    Classe un email. Essaie d'abord les règles, puis le LLM si nécessaire.
    En cas d'échec LLM, retourne confidence=0 → escalade automatique.
    """
    # Tente les règles métier en premier
    rule_result = _apply_keyword_rules(email_text)
    if rule_result is not None:
        return rule_result

    # Fallback LLM
    try:
        raw = _call_llm(email_text)
    except Exception as exc:
        _logger.error("Classification LLM failed after retries: %s", exc)
        return Classification(
            category="autre",
            confidence=0.0,
            reasoning=f"Erreur LLM ({type(exc).__name__}) → escalade humaine",
        )

    data = safe_parse_json(raw)
    if data is None:
        _logger.warning("Classification: JSON unparseable → escalade. raw=%r", raw[:100])
        return Classification(
            category="autre",
            confidence=0.0,
            reasoning="Réponse LLM non parseable → escalade humaine",
        )

    category = normalize_literal(data.get("category"), _CATEGORY_VALUES, "autre")
    confidence = safe_float(data.get("confidence"), default=0.0)

    return Classification(
        category=category,
        confidence=confidence,
        reasoning=data.get("reasoning") or "",
    )
