"""
Génération de la réponse au client.

UTILISE le LLM : génération de langage naturel contextualisé.
Le template de base vient de la knowledge base (déterministe).
Le LLM personnalise et enrichit avec le contexte RAG.

Séquence :
1. Récupérer le template de réponse (règle métier)
2. Enrichir le contexte avec le RAG (règles comptables)
3. Générer la réponse finale via LLM
"""
import logging
import os

import anthropic

from src.models import Classification, ExtractedInfo
from src.rag.knowledge_base import get_template
from src.utils import with_retry

_logger = logging.getLogger("agent")

_SYSTEM_PROMPT = """Tu es un comptable expert chez Dougs, un cabinet comptable en ligne.
Tu réponds aux clients de façon professionnelle, claire et concise en français.
Tu t'appuies sur les règles comptables fournies dans le contexte.
Ne génère pas d'informations inventées sur les montants ou les dates si tu n'en as pas.
"""

# Réponse de secours si le LLM est indisponible
_FALLBACK_RESPONSE = (
    "Bonjour,\n\nNous avons bien reçu votre message et nous revenons vers vous "
    "dans les meilleurs délais.\n\nCordialement,\nL'équipe Dougs"
)


@with_retry(max_attempts=3, base_delay=2.0)
def _call_llm(user_prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()


def generate_response(
    email_text: str,
    classification: Classification,
    extracted_info: ExtractedInfo,
    rag_context: list[str],
) -> str:
    """
    Génère une réponse email personnalisée.
    En cas d'échec LLM après retries, retourne une réponse générique de secours.
    """
    # Base template (déterministe, sans LLM)
    template = get_template(classification.category) or get_template("escalate") or _FALLBACK_RESPONSE

    # Substitution des variables connues
    template = template.replace("{client_name}", extracted_info.client_name or "Monsieur/Madame")
    template = template.replace(
        "{date_document}", str(extracted_info.date_document) if extracted_info.date_document else "reçu"
    )
    template = template.replace(
        "{montant}", f"{extracted_info.montant:.2f}" if extracted_info.montant else "indiqué"
    )
    template = template.replace("{devise}", extracted_info.devise)

    # Cas simples : template suffit, pas besoin de LLM
    if classification.category in ("document_upload", "relance") and not rag_context:
        return template

    # LLM pour personnaliser
    context_str = "\n---\n".join(rag_context) if rag_context else "Aucun contexte supplémentaire."
    user_prompt = f"""Email reçu :
{email_text[:2000]}

Catégorie détectée : {classification.category}
Action requise : {extracted_info.action_required}

Règles comptables pertinentes :
{context_str}

Template de base à adapter :
{template}

Génère une réponse email professionnelle et personnalisée."""

    try:
        return _call_llm(user_prompt)
    except Exception as exc:
        _logger.error("Response generation LLM failed after retries: %s — using fallback template", exc)
        return template  # le template pré-rempli reste une réponse correcte
