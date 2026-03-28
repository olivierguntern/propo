"""
Tests de src/analysis/extractor.py.

Tous les appels LLM sont mockés.
Couvre :
- Extraction complète avec tous les champs
- Champs optionnels manquants → valeurs par défaut
- Normalisation urgency (INVALID → "low")
- Normalisation devise (CHF → "EUR" si hors enum, mais CHF est dans l'enum)
- Montant sous forme de chaîne française "2 450,00"
- Date invalide → None sans crash
- JSON malformé → ExtractedInfo minimal avec message d'erreur
- Erreur LLM → ExtractedInfo minimal
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.extractor import extract_info
from src.models import ExtractedInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_llm(text: str):
    return patch(
        "src.analysis.extractor.anthropic.Anthropic",
        return_value=MagicMock(
            messages=MagicMock(
                create=MagicMock(
                    return_value=MagicMock(content=[MagicMock(text=text)])
                )
            )
        ),
    )


# ---------------------------------------------------------------------------
# Extraction réussie
# ---------------------------------------------------------------------------

class TestExtractInfoSuccess:
    @patch("src.utils.time.sleep")
    def test_full_extraction(self, mock_sleep):
        json_resp = """{
            "montant": 2450.0,
            "devise": "EUR",
            "date_document": "2024-03-28",
            "client_name": "Dupont SAS",
            "urgency": "low",
            "action_required": "Enregistrer la facture",
            "tva_rate": 20.0
        }"""
        with _patch_llm(json_resp):
            result = extract_info("email texte")
        assert result.montant == pytest.approx(2450.0)
        assert result.devise == "EUR"
        assert result.date_document == date(2024, 3, 28)
        assert result.client_name == "Dupont SAS"
        assert result.urgency == "low"
        assert result.action_required == "Enregistrer la facture"
        assert result.tva_rate == pytest.approx(20.0)

    @patch("src.utils.time.sleep")
    def test_null_fields_become_none(self, mock_sleep):
        json_resp = """{
            "montant": null,
            "devise": "EUR",
            "date_document": null,
            "client_name": null,
            "urgency": "low",
            "action_required": "",
            "tva_rate": null
        }"""
        with _patch_llm(json_resp):
            result = extract_info("email texte")
        assert result.montant is None
        assert result.date_document is None
        assert result.client_name is None
        assert result.tva_rate is None

    @patch("src.utils.time.sleep")
    def test_all_urgency_values(self, mock_sleep):
        for urgency in ("low", "medium", "high"):
            json_resp = f'{{"urgency": "{urgency}", "action_required": "test"}}'
            with _patch_llm(json_resp):
                result = extract_info("texte")
            assert result.urgency == urgency

    @patch("src.utils.time.sleep")
    def test_urgency_normalized_from_invalid(self, mock_sleep):
        """LLM retourne 'urgent' → normalisé en 'low' (valeur par défaut)."""
        json_resp = '{"urgency": "urgent", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.urgency == "low"

    @patch("src.utils.time.sleep")
    def test_urgency_case_insensitive(self, mock_sleep):
        json_resp = '{"urgency": "HIGH", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.urgency == "high"


# ---------------------------------------------------------------------------
# Montant
# ---------------------------------------------------------------------------

class TestAmountExtraction:
    @patch("src.utils.time.sleep")
    def test_montant_as_integer(self, mock_sleep):
        json_resp = '{"montant": 1000, "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.montant == pytest.approx(1000.0)

    @patch("src.utils.time.sleep")
    def test_montant_as_french_string(self, mock_sleep):
        """Format français "2 450,00" → 2450.0."""
        json_resp = '{"montant": "2 450,00", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.montant == pytest.approx(2450.0)

    @patch("src.utils.time.sleep")
    def test_montant_as_dot_string(self, mock_sleep):
        json_resp = '{"montant": "1234.56", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.montant == pytest.approx(1234.56)

    @patch("src.utils.time.sleep")
    def test_montant_invalid_string_becomes_none(self, mock_sleep):
        """'non communiqué' → safe_float retourne None."""
        json_resp = '{"montant": "non communiqué", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.montant is None


# ---------------------------------------------------------------------------
# Devise
# ---------------------------------------------------------------------------

class TestDeviseExtraction:
    @patch("src.utils.time.sleep")
    def test_eur_devise(self, mock_sleep):
        json_resp = '{"devise": "EUR", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.devise == "EUR"

    @patch("src.utils.time.sleep")
    def test_usd_devise(self, mock_sleep):
        json_resp = '{"devise": "USD", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.devise == "USD"

    @patch("src.utils.time.sleep")
    def test_unknown_devise_defaults_to_eur(self, mock_sleep):
        """Devise inconnue → normalisée vers EUR."""
        json_resp = '{"devise": "BTC", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.devise == "EUR"

    @patch("src.utils.time.sleep")
    def test_devise_case_insensitive(self, mock_sleep):
        json_resp = '{"devise": "eur", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.devise == "EUR"


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

class TestDateExtraction:
    @patch("src.utils.time.sleep")
    def test_valid_iso_date(self, mock_sleep):
        json_resp = '{"date_document": "2024-03-28", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.date_document == date(2024, 3, 28)

    @patch("src.utils.time.sleep")
    def test_invalid_date_format_becomes_none(self, mock_sleep):
        json_resp = '{"date_document": "28/03/2024", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.date_document is None

    @patch("src.utils.time.sleep")
    def test_impossible_date_becomes_none(self, mock_sleep):
        json_resp = '{"date_document": "2024-13-45", "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.date_document is None

    @patch("src.utils.time.sleep")
    def test_integer_date_becomes_none(self, mock_sleep):
        json_resp = '{"date_document": 20240328, "action_required": "test"}'
        with _patch_llm(json_resp):
            result = extract_info("texte")
        assert result.date_document is None


# ---------------------------------------------------------------------------
# Fallbacks sur erreurs
# ---------------------------------------------------------------------------

class TestExtractInfoFallback:
    @patch("src.utils.time.sleep")
    def test_json_parse_failure_returns_minimal_info(self, mock_sleep):
        with _patch_llm("Impossible d'extraire les informations"):
            result = extract_info("texte")
        assert isinstance(result, ExtractedInfo)
        assert "Extraction échouée" in result.action_required
        assert result.montant is None

    @patch("src.utils.time.sleep")
    def test_llm_connection_error_returns_minimal_info(self, mock_sleep):
        import anthropic
        with patch("src.analysis.extractor.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = anthropic.APIConnectionError(
                request=MagicMock()
            )
            result = extract_info("texte")
        assert isinstance(result, ExtractedInfo)
        assert "Extraction échouée" in result.action_required

    @patch("src.utils.time.sleep")
    def test_rate_limit_retries_then_succeeds(self, mock_sleep):
        import anthropic
        success_resp = MagicMock(
            content=[MagicMock(text='{"montant": 500.0, "action_required": "ok"}')]
        )
        rate_err = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch("src.analysis.extractor.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = [rate_err, success_resp]
            result = extract_info("texte")
        assert result.montant == pytest.approx(500.0)
        assert mock_sleep.called
