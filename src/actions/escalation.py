"""
Gestion de l'escalade vers un humain.

Cas d'escalade :
1. Confidence de classification < seuil
2. Erreurs de validation (données financières incorrectes)
3. Montant facture élevé (> HIGH_AMOUNT_ESCALATION)
4. Urgence haute
5. Catégorie "autre" (incompris)
6. Erreur pipeline (confidence=0.0)
"""
import os

from src.models import AgentAction, Classification, ExtractedInfo, ValidationResult

_SENSITIVE_CATEGORIES = {"facture"}

# Seuil configurable — aligné avec validator._HIGH_AMOUNT_WARNING
_HIGH_AMOUNT_ESCALATION = float(os.getenv("HIGH_AMOUNT_ESCALATION", "10000"))


def should_escalate(
    classification: Classification,
    validation: ValidationResult,
    extracted_info: ExtractedInfo,
) -> tuple[bool, str]:
    """
    Retourne (doit_escalader, raison).
    Logique purement déterministe — PAS de LLM.
    On collecte TOUTES les raisons pour aider l'opérateur humain.
    """
    reasons: list[str] = []

    # 1. Confiance trop faible (inclut confidence=0.0 → erreur LLM)
    if classification.needs_human_review:
        reasons.append(
            f"Confiance insuffisante ({classification.confidence:.0%}) "
            f"pour la catégorie '{classification.category}'"
        )

    # 2. Erreurs de validation
    if not validation.is_valid:
        reasons.append(f"Erreurs de validation : {'; '.join(validation.errors)}")

    # 3. Montant élevé sur une catégorie sensible
    if (
        classification.category in _SENSITIVE_CATEGORIES
        and extracted_info.montant is not None
        and extracted_info.montant > _HIGH_AMOUNT_ESCALATION
    ):
        reasons.append(
            f"Facture de montant élevé ({extracted_info.montant} {extracted_info.devise}) "
            f"— seuil : {_HIGH_AMOUNT_ESCALATION} €"
        )

    # 4. Urgence haute
    if extracted_info.urgency == "high":
        reasons.append("Email marqué urgent — traitement humain prioritaire")

    # 5. Catégorie non reconnue
    if classification.category == "autre" and classification.confidence > 0.0:
        reasons.append("Catégorie non reconnue par le système")

    if reasons:
        return True, " | ".join(reasons)
    return False, ""


def build_escalation_action(reason: str) -> AgentAction:
    return AgentAction(
        action_type="escalate",
        escalation_reason=reason,
        response=None,
    )
