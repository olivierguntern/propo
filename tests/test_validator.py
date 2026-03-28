"""
Tests exhaustifs de src/analysis/validator.py.

Couvre tous les chemins métier :
- Montants (négatif, zéro, normal, élevé, très élevé)
- TVA (tous les taux valides, taux invalides, précision float)
- Dates (futur, très ancienne, normale, délai LME)
- Urgence + action_required
- Cas sans données (None partout)
"""
from datetime import date, timedelta

import pytest

from src.analysis.validator import _HIGH_AMOUNT_WARNING, _MAX_PAYMENT_DELAY_DAYS, _TVA_RATES_FR, validate
from src.models import ExtractedInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def info(**kwargs) -> ExtractedInfo:
    return ExtractedInfo(**kwargs)


# ---------------------------------------------------------------------------
# Cas sans données
# ---------------------------------------------------------------------------

class TestNoData:
    def test_all_none_is_valid(self):
        result = validate(info())
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Montant
# ---------------------------------------------------------------------------

class TestAmount:
    def test_normal_amount_valid(self):
        assert validate(info(montant=1000.0)).is_valid

    def test_very_small_amount_valid(self):
        assert validate(info(montant=0.01)).is_valid

    def test_negative_amount_is_error(self):
        result = validate(info(montant=-1.0))
        assert not result.is_valid
        assert any("Montant invalide" in e for e in result.errors)

    def test_zero_amount_is_error(self):
        result = validate(info(montant=0.0))
        assert not result.is_valid

    def test_amount_at_high_threshold_warning(self):
        result = validate(info(montant=_HIGH_AMOUNT_WARNING + 1))
        assert result.is_valid  # warning, pas une erreur
        assert len(result.warnings) >= 1

    def test_amount_exactly_at_threshold_no_warning(self):
        result = validate(info(montant=_HIGH_AMOUNT_WARNING))
        # Le seuil est > donc exactement égal → pas de warning
        assert not any("élevé" in w for w in result.warnings)

    def test_very_high_amount_warning_not_error(self):
        result = validate(info(montant=150_000.0))
        assert result.is_valid
        assert len(result.warnings) >= 1

    def test_warning_includes_devise(self):
        result = validate(info(montant=20_000.0, devise="USD"))
        warning_text = " ".join(result.warnings)
        assert "USD" in warning_text


# ---------------------------------------------------------------------------
# TVA
# ---------------------------------------------------------------------------

class TestTVA:
    @pytest.mark.parametrize("rate", [0.0, 5.5, 10.0, 20.0])
    def test_valid_tva_rates(self, rate):
        result = validate(info(tva_rate=rate))
        assert result.is_valid, f"TVA {rate}% should be valid"

    def test_tva_with_float_precision_accepted(self):
        """20.000001 doit être accepté (tolérance 0.01)."""
        assert validate(info(tva_rate=20.000001)).is_valid

    def test_tva_with_negative_precision(self):
        assert validate(info(tva_rate=19.999999)).is_valid

    def test_tva_just_outside_tolerance_rejected(self):
        """20.02 dépasse la tolérance de 0.01 → erreur."""
        result = validate(info(tva_rate=20.02))
        assert not result.is_valid

    @pytest.mark.parametrize("invalid_rate", [15.0, 19.0, 21.0, 25.0, -1.0, 100.0])
    def test_invalid_tva_rates(self, invalid_rate):
        result = validate(info(tva_rate=invalid_rate))
        assert not result.is_valid
        assert any("TVA" in e for e in result.errors)

    def test_tva_error_mentions_valid_rates(self):
        result = validate(info(tva_rate=15.0))
        error_text = " ".join(result.errors)
        assert "20" in error_text  # doit indiquer les taux valides

    def test_tva_none_no_validation(self):
        """Aucun taux fourni → pas de validation TVA."""
        result = validate(info(tva_rate=None))
        assert result.is_valid


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

class TestDates:
    def test_today_is_valid(self):
        assert validate(info(date_document=date.today())).is_valid

    def test_yesterday_is_valid(self):
        assert validate(info(date_document=date.today() - timedelta(days=1))).is_valid

    def test_future_date_is_error(self):
        result = validate(info(date_document=date.today() + timedelta(days=1)))
        assert not result.is_valid
        assert any("futur" in e for e in result.errors)

    def test_far_future_is_error(self):
        result = validate(info(date_document=date(2030, 1, 1)))
        assert not result.is_valid

    def test_old_document_warning(self):
        old = date.today() - timedelta(days=400)
        result = validate(info(date_document=old))
        assert result.is_valid
        assert any("an" in w for w in result.warnings)

    def test_365_days_no_old_warning(self):
        exactly_365 = date.today() - timedelta(days=365)
        result = validate(info(date_document=exactly_365))
        # 365 jours = pas encore > 365
        assert not any("an" in w for w in result.warnings)

    def test_lme_delay_warning_at_61_days(self):
        old = date.today() - timedelta(days=_MAX_PAYMENT_DELAY_DAYS + 1)
        result = validate(info(date_document=old))
        assert any("LME" in w for w in result.warnings)

    def test_lme_delay_no_warning_at_60_days(self):
        exactly_60 = date.today() - timedelta(days=_MAX_PAYMENT_DELAY_DAYS)
        result = validate(info(date_document=exactly_60))
        assert not any("LME" in w for w in result.warnings)

    def test_lme_and_old_doc_both_present(self):
        """Un document de 400 jours → 2 warnings distincts."""
        very_old = date.today() - timedelta(days=400)
        result = validate(info(date_document=very_old))
        assert result.is_valid
        assert len(result.warnings) >= 2

    def test_no_lme_warning_for_future_date(self):
        """Date dans le futur → erreur, mais pas de warning LME."""
        future = date.today() + timedelta(days=10)
        result = validate(info(date_document=future))
        assert not result.is_valid
        assert not any("LME" in w for w in result.warnings)

    def test_none_date_no_validation(self):
        assert validate(info(date_document=None)).is_valid


# ---------------------------------------------------------------------------
# Urgence
# ---------------------------------------------------------------------------

class TestUrgency:
    def test_high_urgency_with_action_no_warning(self):
        result = validate(info(urgency="high", action_required="Rappeler le client"))
        assert not any("urgent" in w for w in result.warnings)

    def test_high_urgency_without_action_warning(self):
        result = validate(info(urgency="high", action_required=""))
        assert any("urgent" in w.lower() for w in result.warnings)

    def test_medium_urgency_no_warning(self):
        result = validate(info(urgency="medium", action_required=""))
        assert result.warnings == []

    def test_low_urgency_no_warning(self):
        result = validate(info(urgency="low"))
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Combinaisons multi-erreurs
# ---------------------------------------------------------------------------

class TestMultipleIssues:
    def test_multiple_errors_all_reported(self):
        result = validate(info(
            montant=-100.0,
            tva_rate=15.0,
            date_document=date(2030, 1, 1),
        ))
        assert not result.is_valid
        assert len(result.errors) >= 3

    def test_errors_and_warnings_independent(self):
        result = validate(info(
            montant=20_000.0,   # warning
            tva_rate=15.0,      # error
        ))
        assert not result.is_valid
        assert len(result.errors) >= 1
        assert len(result.warnings) >= 1
