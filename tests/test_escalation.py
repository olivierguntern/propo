"""
Tests exhaustifs de src/actions/escalation.py.

Couvre :
- Chaque condition d'escalade individuellement
- Combinaisons multi-conditions (toutes les raisons remontées)
- Valeurs limites (seuil confidence, seuil montant)
- Catégories sensibles vs non-sensibles
- Seuil configurable via env
- Cas normaux (pas d'escalade)
"""
import os
from datetime import date

import pytest

from src.actions.escalation import _HIGH_AMOUNT_ESCALATION, build_escalation_action, should_escalate
from src.models import Classification, ExtractedInfo, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cls(category="facture", confidence=0.9):
    return Classification(category=category, confidence=confidence, reasoning="test")


def val(valid=True, errors=None, warnings=None):
    return ValidationResult(is_valid=valid, errors=errors or [], warnings=warnings or [])


def info(urgency="low", montant=None, devise="EUR"):
    return ExtractedInfo(urgency=urgency, montant=montant, devise=devise)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidenceEscalation:
    def test_confidence_below_threshold_escalates(self):
        escalate, reason = should_escalate(cls(confidence=0.5), val(), info())
        assert escalate
        assert "Confiance" in reason
        assert "50%" in reason

    def test_confidence_at_threshold_does_not_escalate(self):
        # threshold = 0.70 → confidence=0.70 est >= threshold
        escalate, _ = should_escalate(cls(confidence=0.70), val(), info())
        assert not escalate

    def test_confidence_just_below_threshold_escalates(self):
        escalate, _ = should_escalate(cls(confidence=0.69), val(), info())
        assert escalate

    def test_zero_confidence_escalates(self):
        """confidence=0.0 = erreur LLM."""
        escalate, reason = should_escalate(cls(confidence=0.0), val(), info())
        assert escalate
        assert "0%" in reason

    def test_high_confidence_does_not_escalate(self):
        escalate, _ = should_escalate(cls(confidence=0.99), val(), info())
        assert not escalate


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationEscalation:
    def test_invalid_data_escalates(self):
        escalate, reason = should_escalate(
            cls(), val(valid=False, errors=["TVA invalide"]), info()
        )
        assert escalate
        assert "TVA invalide" in reason

    def test_multiple_errors_all_in_reason(self):
        escalate, reason = should_escalate(
            cls(), val(valid=False, errors=["TVA invalide", "Montant négatif"]), info()
        )
        assert escalate
        assert "TVA invalide" in reason
        assert "Montant négatif" in reason

    def test_valid_data_does_not_escalate(self):
        escalate, _ = should_escalate(cls(), val(valid=True), info())
        assert not escalate

    def test_warnings_alone_do_not_escalate(self):
        """Les warnings (pas les errors) ne déclenchent pas l'escalade."""
        escalate, _ = should_escalate(
            cls(), val(valid=True, warnings=["Montant élevé"]), info()
        )
        assert not escalate


# ---------------------------------------------------------------------------
# Montant élevé
# ---------------------------------------------------------------------------

class TestHighAmountEscalation:
    def test_high_amount_facture_escalates(self):
        escalate, reason = should_escalate(
            cls(category="facture"), val(), info(montant=_HIGH_AMOUNT_ESCALATION + 1)
        )
        assert escalate
        assert str(int(_HIGH_AMOUNT_ESCALATION)) in reason

    def test_amount_at_threshold_does_not_escalate(self):
        escalate, _ = should_escalate(
            cls(category="facture"), val(), info(montant=_HIGH_AMOUNT_ESCALATION)
        )
        assert not escalate

    def test_high_amount_devis_does_not_escalate(self):
        """Seule la catégorie 'facture' est sensible."""
        escalate, _ = should_escalate(
            cls(category="devis"), val(), info(montant=_HIGH_AMOUNT_ESCALATION + 10_000)
        )
        assert not escalate

    def test_high_amount_question_does_not_escalate(self):
        escalate, _ = should_escalate(
            cls(category="question_comptable"), val(), info(montant=999_999.0)
        )
        assert not escalate

    def test_none_amount_no_escalation(self):
        escalate, _ = should_escalate(cls(category="facture"), val(), info(montant=None))
        assert not escalate

    def test_reason_includes_devise(self):
        escalate, reason = should_escalate(
            cls(category="facture"), val(), info(montant=50_000.0, devise="USD")
        )
        assert escalate
        assert "USD" in reason


# ---------------------------------------------------------------------------
# Urgence
# ---------------------------------------------------------------------------

class TestUrgencyEscalation:
    def test_high_urgency_escalates(self):
        escalate, reason = should_escalate(cls(), val(), info(urgency="high"))
        assert escalate
        assert "urgent" in reason.lower()

    def test_medium_urgency_does_not_escalate(self):
        escalate, _ = should_escalate(cls(), val(), info(urgency="medium"))
        assert not escalate

    def test_low_urgency_does_not_escalate(self):
        escalate, _ = should_escalate(cls(), val(), info(urgency="low"))
        assert not escalate


# ---------------------------------------------------------------------------
# Catégorie "autre"
# ---------------------------------------------------------------------------

class TestCategoryAutreEscalation:
    def test_autre_with_positive_confidence_escalates(self):
        escalate, reason = should_escalate(
            cls(category="autre", confidence=0.8), val(), info()
        )
        assert escalate
        assert "non reconnue" in reason.lower()

    def test_autre_with_zero_confidence_escalates_via_confidence(self):
        """confidence=0 déclenche déjà l'escalade via le check confidence."""
        escalate, _ = should_escalate(
            cls(category="autre", confidence=0.0), val(), info()
        )
        assert escalate


# ---------------------------------------------------------------------------
# Multi-conditions
# ---------------------------------------------------------------------------

class TestMultipleEscalationReasons:
    def test_all_reasons_concatenated(self):
        """Toutes les conditions actives doivent apparaître dans le message."""
        escalate, reason = should_escalate(
            cls(confidence=0.5),
            val(valid=False, errors=["TVA invalide"]),
            info(urgency="high"),
        )
        assert escalate
        assert "Confiance" in reason
        assert "TVA invalide" in reason
        assert "urgent" in reason.lower()

    def test_reason_separator_present(self):
        escalate, reason = should_escalate(
            cls(confidence=0.5),
            val(valid=False, errors=["Erreur"]),
            info(),
        )
        assert " | " in reason


# ---------------------------------------------------------------------------
# Cas normaux (pas d'escalade)
# ---------------------------------------------------------------------------

class TestNoEscalation:
    def test_facture_low_amount_high_confidence_valid(self):
        escalate, _ = should_escalate(
            cls(category="facture", confidence=0.95),
            val(valid=True),
            info(urgency="low", montant=500.0),
        )
        assert not escalate

    def test_devis_normal_case(self):
        escalate, _ = should_escalate(
            cls(category="devis", confidence=0.88),
            val(valid=True),
            info(),
        )
        assert not escalate

    def test_question_comptable_normal(self):
        escalate, _ = should_escalate(
            cls(category="question_comptable", confidence=0.75),
            val(valid=True),
            info(),
        )
        assert not escalate

    def test_empty_reason_when_no_escalation(self):
        _, reason = should_escalate(cls(confidence=0.9), val(), info())
        assert reason == ""


# ---------------------------------------------------------------------------
# build_escalation_action
# ---------------------------------------------------------------------------

class TestBuildEscalationAction:
    def test_returns_escalate_action_type(self):
        action = build_escalation_action("raison test")
        assert action.action_type == "escalate"

    def test_reason_preserved(self):
        action = build_escalation_action("raison importante")
        assert action.escalation_reason == "raison importante"

    def test_response_is_none(self):
        action = build_escalation_action("raison")
        assert action.response is None


# ---------------------------------------------------------------------------
# Seuil configurable
# ---------------------------------------------------------------------------

class TestConfigurableThreshold:
    def test_custom_threshold_via_env(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.90")
        # confidence=0.85 < 0.90 → doit escalader
        escalate, _ = should_escalate(
            cls(confidence=0.85), val(), info()
        )
        assert escalate

    def test_low_threshold_via_env(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.50")
        # confidence=0.60 > 0.50 → ne doit pas escalader
        escalate, _ = should_escalate(
            cls(category="devis", confidence=0.60), val(), info()
        )
        assert not escalate
