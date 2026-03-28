"""
Tests de src/analysis/classifier.py.

Stratégie de mock :
- On patche `src.analysis.classifier.anthropic.Anthropic` pour simuler le LLM.
- On patche `src.utils.time.sleep` pour éviter les vrais délais de retry.

Couvre :
- Règles mots-clés (toutes les catégories, ambiguïté, cas limites)
- Chemin LLM : réponse valide, JSON bruité, catégorie normalisée,
               confidence invalide, JSON malformé
- Gestion erreurs : réseau, rate-limit (retry), erreur fatale
"""
from unittest.mock import MagicMock, patch

import pytest

from src.analysis.classifier import _apply_keyword_rules, classify_email
from src.models import Classification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _patch_llm(text: str):
    """Contexte : patche l'appel LLM pour retourner `text`."""
    return patch(
        "src.analysis.classifier.anthropic.Anthropic",
        return_value=MagicMock(
            messages=MagicMock(
                create=MagicMock(return_value=_make_llm_response(text))
            )
        ),
    )


# ---------------------------------------------------------------------------
# Règles mots-clés
# ---------------------------------------------------------------------------

class TestKeywordRules:
    def test_detects_facture(self):
        result = _apply_keyword_rules("Bonjour, voici notre facture n°001")
        assert result is not None
        assert result.category == "facture"

    def test_detects_invoice_english(self):
        result = _apply_keyword_rules("Please find attached our invoice for services")
        assert result is not None
        assert result.category == "facture"

    def test_detects_devis(self):
        result = _apply_keyword_rules("Je souhaite un devis pour votre prestation")
        assert result is not None
        assert result.category == "devis"

    def test_detects_relance(self):
        result = _apply_keyword_rules("Suite à ma relance, toujours en attente de réponse")
        assert result is not None
        assert result.category == "relance"

    def test_detects_document_upload(self):
        result = _apply_keyword_rules("Veuillez trouver ci-joint les documents demandés")
        assert result is not None
        assert result.category == "document_upload"

    def test_single_keyword_sufficient(self):
        result = _apply_keyword_rules("facture")
        assert result is not None
        assert result.category == "facture"

    def test_returns_none_for_ambiguous_two_categories(self):
        """'facture' et 'devis' → ambigu → None."""
        result = _apply_keyword_rules("La facture correspond au devis initial")
        assert result is None

    def test_returns_none_for_empty_text(self):
        assert _apply_keyword_rules("") is None

    def test_returns_none_for_unknown_content(self):
        assert _apply_keyword_rules("Bonjour, j'espère que vous allez bien") is None

    def test_case_insensitive(self):
        result = _apply_keyword_rules("FACTURE N°001 CI-JOINT")
        # "FACTURE" et "CI-JOINT" → ambigu (facture + document_upload) → None
        assert result is None

    def test_keyword_count_in_reasoning(self):
        result = _apply_keyword_rules("facture facture")
        assert result is not None
        assert "2" in result.reasoning

    def test_confidence_is_0_85(self):
        result = _apply_keyword_rules("voici notre facture")
        assert result is not None
        assert result.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Chemin LLM : réponses valides
# ---------------------------------------------------------------------------

class TestClassifyEmailLLM:
    @patch("src.utils.time.sleep")
    def test_llm_returns_valid_json(self, mock_sleep):
        json_resp = '{"category": "question_comptable", "confidence": 0.88, "reasoning": "Question TVA"}'
        with _patch_llm(json_resp):
            result = classify_email("Comment fonctionne la TVA intracommunautaire ?")
        assert result.category == "question_comptable"
        assert result.confidence == pytest.approx(0.88)
        assert result.reasoning == "Question TVA"

    @patch("src.utils.time.sleep")
    def test_llm_json_with_noise_prefix(self, mock_sleep):
        noisy = 'Voici ma réponse : {"category": "devis", "confidence": 0.75, "reasoning": "ok"}'
        with _patch_llm(noisy):
            result = classify_email("texte sans mot-clé")
        assert result.category == "devis"

    @patch("src.utils.time.sleep")
    def test_llm_category_normalized_uppercase(self, mock_sleep):
        """LLM retourne 'FACTURE' → normalisé en 'facture'."""
        json_resp = '{"category": "FACTURE", "confidence": 0.9, "reasoning": "test"}'
        with _patch_llm(json_resp):
            result = classify_email("texte sans mot-clé")
        assert result.category == "facture"

    @patch("src.utils.time.sleep")
    def test_llm_unknown_category_defaults_to_autre(self, mock_sleep):
        json_resp = '{"category": "inconnue", "confidence": 0.7, "reasoning": "test"}'
        with _patch_llm(json_resp):
            result = classify_email("texte sans mot-clé")
        assert result.category == "autre"

    @patch("src.utils.time.sleep")
    def test_llm_invalid_confidence_string_defaults_to_zero(self, mock_sleep):
        json_resp = '{"category": "facture", "confidence": "élevée", "reasoning": "test"}'
        with _patch_llm(json_resp):
            result = classify_email("texte sans mot-clé")
        assert result.confidence == pytest.approx(0.0)

    @patch("src.utils.time.sleep")
    def test_llm_malformed_json_returns_fallback(self, mock_sleep):
        with _patch_llm("Désolé, je ne peux pas répondre"):
            result = classify_email("texte sans mot-clé")
        assert result.category == "autre"
        assert result.confidence == pytest.approx(0.0)
        assert result.needs_human_review


# ---------------------------------------------------------------------------
# Gestion erreurs LLM
# ---------------------------------------------------------------------------

class TestClassifyEmailErrors:
    @patch("src.utils.time.sleep")
    def test_connection_error_returns_fallback(self, mock_sleep):
        import anthropic
        with patch("src.analysis.classifier.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = anthropic.APIConnectionError(
                request=MagicMock()
            )
            result = classify_email("texte sans mot-clé")
        assert result.category == "autre"
        assert result.confidence == pytest.approx(0.0)
        assert result.needs_human_review

    @patch("src.utils.time.sleep")
    def test_rate_limit_retries_then_succeeds(self, mock_sleep):
        """RateLimitError puis succès → retry fonctionne."""
        import anthropic
        success_response = _make_llm_response(
            '{"category": "facture", "confidence": 0.9, "reasoning": "ok"}'
        )
        rate_err = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with patch("src.analysis.classifier.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = [
                rate_err,
                success_response,
            ]
            result = classify_email("texte sans mot-clé")
        assert result.category == "facture"
        assert mock_sleep.called  # un sleep entre les tentatives

    @patch("src.utils.time.sleep")
    def test_authentication_error_propagates(self, mock_sleep):
        """AuthenticationError = fatale → pas de retry, lève l'exception."""
        import anthropic
        with patch("src.analysis.classifier.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = anthropic.AuthenticationError(
                message="Invalid key",
                response=MagicMock(status_code=401, headers={}),
                body={},
            )
            with pytest.raises(anthropic.AuthenticationError):
                classify_email("texte sans mot-clé")
        mock_sleep.assert_not_called()

    def test_keyword_rule_bypasses_llm(self):
        """Si les règles mots-clés matchent, le LLM n'est jamais appelé."""
        with patch("src.analysis.classifier._call_llm") as mock_call:
            result = classify_email("Veuillez trouver ci-joint notre facture")
            mock_call.assert_not_called()
        assert result is not None
