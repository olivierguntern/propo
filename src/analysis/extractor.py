"""
Extraction d'informations structurées depuis un email.

UTILISE le LLM : la compréhension du langage naturel est son point fort.
Le résultat est ensuite validé par validator.py (sans LLM).
"""
import logging
import os
from datetime import date

import anthropic

from src.models import ExtractedInfo
from src.utils import normalize_literal, safe_float, safe_parse_json, with_retry

_logger = logging.getLogger("agent")

_URGENCY_VALUES = ("low", "medium", "high")
_DEVISE_VALUES = ("EUR", "USD", "GBP", "CHF")  # extensible

_SYSTEM_PROMPT = """Tu es un assistant comptable.
Extrais les informations de cet email et retourne UNIQUEMENT un JSON valide :
{
  "montant": <float ou null>,
  "devise": "<EUR|USD|GBP|CHF>",
  "date_document": "<YYYY-MM-DD ou null>",
  "client_name": "<nom ou null>",
  "urgency": "<low|medium|high>",
  "action_required": "<description courte de l'action à faire>",
  "tva_rate": <float ou null>
}
Ne génère rien d'autre que ce JSON."""


@with_retry(max_attempts=3, base_delay=2.0)
def _call_llm(email_text: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": email_text[:3000]}],
    )
    return message.content[0].text.strip()


def extract_info(email_text: str) -> ExtractedInfo:
    """
    Extrait les infos clés d'un email via LLM.
    En cas d'échec, retourne un ExtractedInfo minimal (action_required signale l'erreur).
    """
    try:
        raw = _call_llm(email_text)
    except Exception as exc:
        _logger.error("Extraction LLM failed after retries: %s", exc)
        return ExtractedInfo(action_required=f"Extraction échouée ({type(exc).__name__}) - révision manuelle")

    data = safe_parse_json(raw)
    if data is None:
        _logger.warning("Extraction: JSON unparseable. raw=%r", raw[:100])
        return ExtractedInfo(action_required="Extraction échouée (JSON invalide) - révision manuelle")

    # Conversion date string → date objet
    parsed_date: date | None = None
    date_str = data.get("date_document")
    if date_str and isinstance(date_str, str):
        try:
            parsed_date = date.fromisoformat(date_str)
        except ValueError:
            _logger.warning("Invalid date_document value: %r", date_str)

    # Normalisation des valeurs Literal
    urgency = normalize_literal(data.get("urgency"), _URGENCY_VALUES, "low")
    devise = normalize_literal(data.get("devise"), _DEVISE_VALUES, "EUR")

    # Montant : LLM peut retourner une string comme "2 450,00" → safe_float gère
    raw_montant = data.get("montant")
    if isinstance(raw_montant, str):
        raw_montant = raw_montant.replace(" ", "").replace(",", ".")
    montant = safe_float(raw_montant, default=None) if raw_montant is not None else None  # type: ignore[arg-type]

    return ExtractedInfo(
        montant=montant,
        devise=devise,
        date_document=parsed_date,
        client_name=data.get("client_name") or None,
        urgency=urgency,
        action_required=data.get("action_required") or "",
        tva_rate=safe_float(data.get("tva_rate"), default=None) if data.get("tva_rate") is not None else None,  # type: ignore[arg-type]
    )
