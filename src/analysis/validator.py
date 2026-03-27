"""
Validation des données extraites.

⚠️  PAS DE LLM ICI.

Pourquoi ?
- Les règles comptables sont déterministes (TVA = 20%, délai de paiement légal = 30 jours…)
- Le LLM peut halluciner des validations → dangereux pour les données financières
- La logique métier doit être auditée et reproductible

C'est ici qu'on met l'intelligence métier, pas dans le LLM.
"""
from datetime import date

from src.models import ExtractedInfo, ValidationResult

# Règles métier France
_TVA_RATES_FR = {0.0, 5.5, 10.0, 20.0}
_MAX_INVOICE_AMOUNT = 100_000.0
_MAX_PAYMENT_DELAY_DAYS = 60  # Loi LME


def validate(info: ExtractedInfo) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # --- Montant ---
    if info.montant is not None:
        if info.montant <= 0:
            errors.append(f"Montant invalide : {info.montant} (doit être > 0)")
        elif info.montant > _MAX_INVOICE_AMOUNT:
            warnings.append(
                f"Montant élevé ({info.montant} {info.devise}) : vérification manuelle recommandée"
            )

    # --- TVA ---
    if info.tva_rate is not None:
        if info.tva_rate not in _TVA_RATES_FR:
            errors.append(
                f"Taux de TVA invalide : {info.tva_rate}% (valeurs acceptées : {sorted(_TVA_RATES_FR)})"
            )

    # --- Date ---
    if info.date_document is not None:
        today = date.today()
        days_diff = (today - info.date_document).days
        if days_diff > 365:
            warnings.append(f"Document daté de plus d'un an ({info.date_document})")
        if info.date_document > today:
            errors.append(f"Date dans le futur : {info.date_document}")

    # --- Urgence sans action définie ---
    if info.urgency == "high" and not info.action_required:
        warnings.append("Email urgent sans action clairement définie")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
