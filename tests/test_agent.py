"""
Tests unitaires — sans appels LLM (mocks).

On teste la logique métier déterministe :
- Règles de classification par mots-clés
- Validation des données (dont TVA float, délai LME)
- Logique d'escalade (toutes les raisons collectées)
- Classification des documents
- Utilitaires : safe_parse_json, normalize_literal, safe_float
- Robustesse : email reader FileNotFoundError, pipeline fallback
"""
import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.classifier import _apply_keyword_rules
from src.analysis.validator import validate
from src.actions.document_classifier import classify_document
from src.actions.escalation import should_escalate
from src.models import Classification, ExtractedInfo, ValidationResult
from src.utils import normalize_literal, safe_float, safe_parse_json


# ---------------------------------------------------------------------------
# Tests utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_safe_parse_json_valid(self):
        data = safe_parse_json('{"key": "value"}')
        assert data == {"key": "value"}

    def test_safe_parse_json_with_noise(self):
        data = safe_parse_json('Voici le résultat : {"key": "val"} fin.')
        assert data == {"key": "val"}

    def test_safe_parse_json_malformed(self):
        # JSON avec quote non échappée — doit retourner None sans lever
        data = safe_parse_json('{"key": "val"ue"}')
        assert data is None

    def test_safe_parse_json_no_json(self):
        assert safe_parse_json("pas de JSON ici") is None

    def test_normalize_literal_exact(self):
        assert normalize_literal("facture", ("facture", "devis"), "autre") == "facture"

    def test_normalize_literal_case_insensitive(self):
        assert normalize_literal("FACTURE", ("facture", "devis"), "autre") == "facture"

    def test_normalize_literal_unknown(self):
        assert normalize_literal("inconnu", ("facture", "devis"), "autre") == "autre"

    def test_normalize_literal_non_string(self):
        assert normalize_literal(42, ("facture",), "autre") == "autre"

    def test_safe_float_valid(self):
        assert safe_float("3.14", 0.0) == pytest.approx(3.14)

    def test_safe_float_invalid(self):
        assert safe_float("élevé", 0.0) == 0.0

    def test_safe_float_none(self):
        assert safe_float(None, 99.0) == 99.0


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
        # Contient à la fois "facture" et "devis" → ambigu → LLM
        result = _apply_keyword_rules("Concernant notre facture et le devis associé")
        assert result is None

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
            date_document=date.today(),
        )
        result = validate(info)
        assert result.is_valid

    def test_tva_with_float_precision(self):
        """20.000001 doit être accepté (tolérance float)."""
        info = ExtractedInfo(montant=1000.0, tva_rate=20.000001)
        result = validate(info)
        assert result.is_valid
        assert not any("TVA" in e for e in result.errors)

    def test_invalid_tva_rate(self):
        info = ExtractedInfo(montant=1000.0, tva_rate=15.0)
        result = validate(info)
        assert not result.is_valid
        assert any("TVA" in e for e in result.errors)

    def test_all_valid_tva_rates(self):
        for rate in (0.0, 5.5, 10.0, 20.0):
            info = ExtractedInfo(montant=500.0, tva_rate=rate)
            result = validate(info)
            assert result.is_valid, f"TVA {rate}% devrait être valide"

    def test_negative_amount(self):
        info = ExtractedInfo(montant=-100.0)
        result = validate(info)
        assert not result.is_valid

    def test_zero_amount(self):
        info = ExtractedInfo(montant=0.0)
        result = validate(info)
        assert not result.is_valid

    def test_future_date_error(self):
        info = ExtractedInfo(date_document=date(2030, 1, 1))
        result = validate(info)
        assert not result.is_valid

    def test_high_amount_warning(self):
        info = ExtractedInfo(montant=15_000.0)
        result = validate(info)
        assert result.is_valid  # pas une erreur bloquante
        assert any("élevé" in w for w in result.warnings)

    def test_lme_delay_warning(self):
        """Document de plus de 60 jours → warning LME."""
        old_date = date.today() - timedelta(days=61)
        info = ExtractedInfo(date_document=old_date)
        result = validate(info)
        assert any("LME" in w for w in result.warnings)

    def test_no_lme_warning_for_recent_doc(self):
        recent = date.today() - timedelta(days=30)
        info = ExtractedInfo(date_document=recent)
        result = validate(info)
        assert not any("LME" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def _cls(self, category="facture", confidence=0.9):
        return Classification(category=category, confidence=confidence, reasoning="test")

    def _val(self, valid=True, errors=None):
        return ValidationResult(is_valid=valid, errors=errors or [])

    def _info(self, urgency="low", montant=None):
        return ExtractedInfo(urgency=urgency, montant=montant)

    def test_low_confidence_triggers_escalade(self):
        escalate, reason = should_escalate(self._cls(confidence=0.5), self._val(), self._info())
        assert escalate
        assert "Confiance" in reason

    def test_zero_confidence_triggers_escalade(self):
        """confidence=0.0 = erreur LLM → escalade."""
        escalate, reason = should_escalate(self._cls(confidence=0.0), self._val(), self._info())
        assert escalate

    def test_invalid_data_triggers_escalade(self):
        escalate, reason = should_escalate(
            self._cls(), self._val(valid=False, errors=["TVA invalide"]), self._info()
        )
        assert escalate
        assert "TVA invalide" in reason

    def test_high_urgency_triggers_escalade(self):
        escalate, reason = should_escalate(self._cls(), self._val(), self._info(urgency="high"))
        assert escalate
        assert "urgent" in reason.lower()

    def test_high_amount_facture_triggers_escalade(self):
        escalate, reason = should_escalate(
            self._cls(category="facture"), self._val(), self._info(montant=50_000.0)
        )
        assert escalate

    def test_multiple_reasons_all_reported(self):
        """Toutes les raisons d'escalade doivent apparaître dans le message."""
        escalate, reason = should_escalate(
            self._cls(confidence=0.5),
            self._val(valid=False, errors=["TVA invalide"]),
            self._info(urgency="high"),
        )
        assert escalate
        assert "Confiance" in reason
        assert "TVA invalide" in reason
        assert "urgent" in reason.lower()

    def test_normal_case_no_escalade(self):
        escalate, _ = should_escalate(
            self._cls(category="devis", confidence=0.9), self._val(), self._info()
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


# ---------------------------------------------------------------------------
# Tests email reader
# ---------------------------------------------------------------------------

class TestEmailReader:
    def test_file_not_found(self):
        from src.ingestion.email_reader import load_mock_emails
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_mock_emails("non_existant_path.json")

    def test_invalid_json(self, tmp_path):
        from src.ingestion.email_reader import load_mock_emails
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalide json }", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON invalide"):
            load_mock_emails(bad_file)

    def test_skips_malformed_entries(self, tmp_path):
        from src.ingestion.email_reader import load_mock_emails
        data = [
            {"id": "ok", "from": "a@b.fr", "subject": "S", "body": "B",
             "received_at": "2024-01-01T10:00:00", "attachments": []},
            {"id": "bad"},  # entrée malformée → skippée
        ]
        f = tmp_path / "emails.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        emails = load_mock_emails(f)
        assert len(emails) == 1
        assert emails[0].id == "ok"
