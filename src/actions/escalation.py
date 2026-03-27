"""
Gestion de l'escalade vers un humain.

Cas d'escalade :
1. Confidence de classification < seuil
2. Erreurs de validation (données financières incorrectes)
3. Urgence haute + catégorie sensible
4. Catégorie "autre" (incompris)
5. Hallucination LLM détectée (parsing JSON échoué)
"""
from src.models import AgentAction, Classification, ExtractedInfo, ValidationResult


_SENSITIVE_CATEGORIES = {"facture"}  # erreur financière = impact direct


def should_escalate(
    classification: Classification,
    validation: ValidationResult,
    extracted_info: ExtractedInfo,
) -> tuple[bool, str]:
    """
    Retourne (doit_escalader, raison).
    Logique purement déterministe — PAS de LLM.
    """
    # 1. Confiance trop faible
    if classification.needs_human_review:
        return True, f"Confiance insuffisante ({classification.confidence:.0%}) pour la catégorie '{classification.category}'"

    # 2. Erreurs de validation
    if not validation.is_valid:
        return True, f"Erreurs de validation : {'; '.join(validation.errors)}"

    # 3. Montant élevé sur une facture
    if (
        classification.category in _SENSITIVE_CATEGORIES
        and extracted_info.montant is not None
        and extracted_info.montant > 10_000
    ):
        return True, f"Facture de montant élevé ({extracted_info.montant} {extracted_info.devise}) nécessite validation humaine"

    # 4. Urgence haute
    if extracted_info.urgency == "high":
        return True, "Email marqué urgent — traitement humain prioritaire"

    # 5. Catégorie non reconnue
    if classification.category == "autre":
        return True, "Catégorie non reconnue"

    return False, ""


def build_escalation_action(reason: str) -> AgentAction:
    return AgentAction(
        action_type="escalate",
        escalation_reason=reason,
        response=None,
    )
