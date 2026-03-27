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
import os

import anthropic

from src.models import Classification, ExtractedInfo
from src.rag.knowledge_base import get_template

_SYSTEM_PROMPT = """Tu es un comptable expert chez Dougs, un cabinet comptable en ligne.
Tu réponds aux clients de façon professionnelle, claire et concise en français.
Tu t'appuies sur les règles comptables fournies dans le contexte.
Ne génère pas d'informations inventées sur les montants ou les dates si tu n'en as pas.
"""


def generate_response(
    email_text: str,
    classification: Classification,
    extracted_info: ExtractedInfo,
    rag_context: list[str],
) -> str:
    """Génère une réponse email personnalisée."""

    # Base template (déterministe, sans LLM)
    template = get_template(classification.category) or get_template("escalate") or ""

    # Variables connues
    template = template.replace("{client_name}", extracted_info.client_name or "Monsieur/Madame")
    template = template.replace(
        "{date_document}", str(extracted_info.date_document) if extracted_info.date_document else "reçu"
    )
    template = template.replace(
        "{montant}", f"{extracted_info.montant:.2f}" if extracted_info.montant else "indiqué"
    )
    template = template.replace("{devise}", extracted_info.devise)

    # Si pas besoin de personnalisation poussée, retourner le template
    if classification.category in ("document_upload", "relance") and not rag_context:
        return template

    # Sinon, LLM pour enrichir
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

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text.strip()
