"""
Validation des données extraites.

⚠️  PAS DE LLM ICI.

Pourquoi ?
- Les règles comptables sont déterministes (TVA = 20%, délai de paiement légal = 60 jours…)
- Le LLM peut halluciner des validations → dangereux pour les données financières
- La logique métier doit être auditée et reproductible

C'est ici qu'on met l'intelligence métier, pas dans le LLM.
"""
from datetime import date

from src.models import ExtractedInfo, ValidationResult

# Règles métier France
_TVA_RATES_FR = {0.0, 5.5, 10.0, 20.0}
_TVA_TOLERANCE = 0.01           # tolérance float : 20.000001 est accepté
_MAX_INVOICE_AMOUNT = 100_000.0
_MAX_PAYMENT_DELAY_DAYS = 60    # Loi LME — délai légal inter-entreprises
_HIGH_AMOUNT_WARNING = 10_000.0 # seuil cohérent avec escalation.py


def _is_valid_tva(rate: float) -> bool:
    """Comparaison TVA avec tolérance float pour éviter les faux positifs."""
    return any(abs(rate - r) < _TVA_TOLERANCE for r in _TVA_RATES_FR)


def validate(info: ExtractedInfo) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    # --- Montant ---
    if info.montant is not None:
        if info.montant <= 0:
            errors.append(f"Montant invalide : {info.montant} (doit être > 0)")
        elif info.montant > _MAX_INVOICE_AMOUNT:
            warnings.append(
                f"Montant très élevé ({info.montant} {info.devise}) : vérification manuelle recommandée"
            )
        elif info.montant > _HIGH_AMOUNT_WARNING:
            warnings.append(
                f"Montant élevé ({info.montant} {info.devise}) : validation humaine requise"
            )

    # --- TVA ---
    if info.tva_rate is not None:
        if not _is_valid_tva(info.tva_rate):
            errors.append(
                f"Taux de TVA invalide : {info.tva_rate}% "
                f"(valeurs acceptées : {sorted(_TVA_RATES_FR)})"
            )

    # --- Date ---
    if info.date_document is not None:
        today = date.today()
        days_diff = (today - info.date_document).days
        if days_diff > 365:
            warnings.append(f"Document daté de plus d'un an ({info.date_document})")
        if info.date_document > today:
            errors.append(f"Date dans le futur : {info.date_document}")

    # --- Délai de paiement légal (Loi LME) ---
    # Si la date du document est connue, on vérifie que le délai restant est légal.
    # En l'absence de date d'échéance explicite, on utilise la date du document + 60j.
    if info.date_document is not None and not info.date_document > date.today():
        days_since_doc = (date.today() - info.date_document).days
        if days_since_doc > _MAX_PAYMENT_DELAY_DAYS:
            warnings.append(
                f"Document de {days_since_doc} jours : délai légal LME de {_MAX_PAYMENT_DELAY_DAYS} jours dépassé"
            )

    # --- Urgence sans action définie ---
    if info.urgency == "high" and not info.action_required:
        warnings.append("Email urgent sans action clairement définie")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
