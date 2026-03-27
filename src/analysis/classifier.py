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
import json
import os
import re

import anthropic

from src.models import Classification

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


def classify_email(email_text: str) -> Classification:
    """
    Classe un email. Essaie d'abord les règles, puis le LLM si nécessaire.
    """
    # Tente les règles métier en premier
    rule_result = _apply_keyword_rules(email_text)
    if rule_result is not None:
        return rule_result

    # Fallback LLM
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": email_text[:3000]}],
    )

    raw = message.content[0].text.strip()

    # Sécurité : extraction JSON robuste même si le LLM ajoute du bruit
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return Classification(
            category="autre",
            confidence=0.0,
            reasoning="Réponse LLM non parseable → escalade humaine",
        )

    data = json.loads(json_match.group())
    return Classification(
        category=data.get("category", "autre"),
        confidence=float(data.get("confidence", 0.5)),
        reasoning=data.get("reasoning", ""),
    )
