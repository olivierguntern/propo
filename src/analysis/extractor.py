"""
Extraction d'informations structurées depuis un email.

UTILISE le LLM : la compréhension du langage naturel est son point fort.
Le résultat est ensuite validé par validator.py (sans LLM).
"""
import json
import os
import re
from datetime import date

import anthropic

from src.models import ExtractedInfo

_SYSTEM_PROMPT = """Tu es un assistant comptable.
Extrais les informations de cet email et retourne UNIQUEMENT un JSON valide :
{
  "montant": <float ou null>,
  "devise": "<EUR|USD|GBP>",
  "date_document": "<YYYY-MM-DD ou null>",
  "client_name": "<nom ou null>",
  "urgency": "<low|medium|high>",
  "action_required": "<description courte de l'action à faire>",
  "tva_rate": <float ou null>
}
Ne génère rien d'autre que ce JSON."""


def extract_info(email_text: str) -> ExtractedInfo:
    """Extrait les infos clés d'un email via LLM."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": email_text[:3000]}],
    )

    raw = message.content[0].text.strip()

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return ExtractedInfo(action_required="Extraction échouée - révision manuelle")

    data = json.loads(json_match.group())

    # Conversion date string → date objet
    date_str = data.get("date_document")
    parsed_date: date | None = None
    if date_str:
        try:
            parsed_date = date.fromisoformat(date_str)
        except ValueError:
            parsed_date = None

    return ExtractedInfo(
        montant=data.get("montant"),
        devise=data.get("devise", "EUR"),
        date_document=parsed_date,
        client_name=data.get("client_name"),
        urgency=data.get("urgency", "low"),
        action_required=data.get("action_required", ""),
        tva_rate=data.get("tva_rate"),
    )
