"""
Tests unitaires — sans appels LLM (mocks).

On teste la logique métier déterministe :
- Règles de classification par mots-clés
- Validation des données
- Logique d'escalade
- Classification des documents

Les appels LLM sont mockés : on teste la logique, pas le modèle.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.classifier import _apply_keyword_rules
from src.analysis.validator import validate
from src.actions.document_classifier import classify_document
from src.actions.escalation import should_escalate
from src.models import Classification, ExtractedInfo, ValidationResult


# ---------------------------------------------------------------------------
# Tests classifier (règles mots-clés)
# ---------------------------------------------------------------------------

class TestKeywordClassifier:
    def test_detects_facture(self):
        result = _apply_keyword_rules("Veuillez trouver ci-joint notre facture n°2024-001")
        assert result is not None
        assert result.category == "facture"
        assert result.confidence >= 0.8

    def test_detects_devis(self):
        result = _apply_keyword_rules("Je souhaite obtenir un devis pour votre prestation")
        assert result is not None
        assert result.category == "devis"

    def test_returns_none_for_ambiguous(self):
        # Contient à la fois "facture" et "devis"
        result = _apply_keyword_rules("Concernant notre facture et le devis associé")
        assert result is None  # ambigu → LLM

    def test_returns_none_for_unknown(self):
        result = _apply_keyword_rules("Bonjour, j'espère que vous allez bien")
        assert result is None


# ---------------------------------------------------------------------------
# Tests validator
# ---------------------------------------------------------------------------

class TestValidator:
    def test_valid_invoice(self):
        info = ExtractedInfo(
            montant=2450.0,
            tva_rate=20.0,
            date_document=date(2024, 3, 28),
        )
        result = validate(info)
        assert result.is_valid

    def test_invalid_tva_rate(self):
        info = ExtractedInfo(montant=1000.0, tva_rate=15.0)
        result = validate(info)
        assert not result.is_valid
        assert any("TVA" in e for e in result.errors)

    def test_negative_amount(self):
        info = ExtractedInfo(montant=-100.0)
        result = validate(info)
        assert not result.is_valid

    def test_future_date_error(self):
        info = ExtractedInfo(date_document=date(2030, 1, 1))
        result = validate(info)
        assert not result.is_valid

    def test_high_amount_warning(self):
        info = ExtractedInfo(montant=50_000.0)
        result = validate(info)
        assert result.is_valid  # pas une erreur
        assert any("élevé" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def _make_classification(self, category="facture", confidence=0.9):
        return Classification(category=category, confidence=confidence, reasoning="test")

    def _make_validation(self, valid=True, errors=None):
        return ValidationResult(is_valid=valid, errors=errors or [])

    def _make_info(self, urgency="low", montant=None):
        return ExtractedInfo(urgency=urgency, montant=montant)

    def test_low_confidence_triggers_escalade(self):
        classif = self._make_classification(confidence=0.5)
        escalate, reason = should_escalate(classif, self._make_validation(), self._make_info())
        assert escalate
        assert "Confiance" in reason

    def test_invalid_data_triggers_escalade(self):
        classif = self._make_classification()
        validation = self._make_validation(valid=False, errors=["TVA invalide"])
        escalate, reason = should_escalate(classif, validation, self._make_info())
        assert escalate

    def test_high_urgency_triggers_escalade(self):
        classif = self._make_classification()
        escalate, reason = should_escalate(
            classif, self._make_validation(), self._make_info(urgency="high")
        )
        assert escalate

    def test_high_amount_facture_triggers_escalade(self):
        classif = self._make_classification(category="facture")
        escalate, reason = should_escalate(
            classif, self._make_validation(), self._make_info(montant=50_000.0)
        )
        assert escalate

    def test_normal_case_no_escalade(self):
        classif = self._make_classification(category="devis", confidence=0.9)
        escalate, _ = should_escalate(
            classif, self._make_validation(), self._make_info()
        )
        assert not escalate


# ---------------------------------------------------------------------------
# Tests document classifier
# ---------------------------------------------------------------------------

class TestDocumentClassifier:
    def test_classifies_by_filename(self):
        assert classify_document("facture_2024.pdf") == "facture"
        assert classify_document("devis_client.pdf") == "devis"
        assert classify_document("releve_mars.pdf") == "releve_bancaire"

    def test_classifies_by_content(self):
        content = "Total TTC : 1 200 EUR\nTVA 20%\nNuméro de facture : 001"
        assert classify_document("document_inconnu.pdf", content) == "facture"

    def test_unknown_returns_autre(self):
        assert classify_document("photo_vacances.jpg") == "autre"
