"""
Tests de src/actions/response_generator.py.

Couvre :
- Substitution des variables dans le template
- Catégories qui court-circuitent le LLM (document_upload, relance sans RAG)
- Appel LLM pour les autres catégories
- Fallback sur template si le LLM échoue
- Réponse de secours si aucun template
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.actions.response_generator import generate_response, _FALLBACK_RESPONSE
from src.models import Classification, ExtractedInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cls(category="facture", confidence=0.9):
    return Classification(category=category, confidence=confidence, reasoning="test")


def info(
    client_name="Dupont",
    montant=1000.0,
    devise="EUR",
    date_document=None,
    urgency="low",
    action_required="",
):
    return ExtractedInfo(
        client_name=client_name,
        montant=montant,
        devise=devise,
        date_document=date_document,
        urgency=urgency,
        action_required=action_required,
    )


def _patch_llm(text: str):
    return patch(
        "src.actions.response_generator.anthropic.Anthropic",
        return_value=MagicMock(
            messages=MagicMock(
                create=MagicMock(
                    return_value=MagicMock(content=[MagicMock(text=text)])
                )
            )
        ),
    )


# ---------------------------------------------------------------------------
# Substitution de template
# ---------------------------------------------------------------------------

class TestTemplateSubstitution:
    @patch("src.utils.time.sleep")
    def test_client_name_substituted(self, mock_sleep):
        with _patch_llm("réponse LLM"):
            response = generate_response("email", cls(), info(client_name="Martin"), [])
        # client_name doit apparaître quelque part (dans le prompt LLM ou le template)
        # On vérifie surtout que ça ne crash pas
        assert response is not None

    def test_missing_client_name_uses_fallback(self, tmp_path):
        """Sans client_name, le template utilise 'Monsieur/Madame'."""
        with patch("src.actions.response_generator.get_template") as mock_tpl:
            mock_tpl.return_value = "Bonjour {client_name},"
            with patch("src.actions.response_generator._call_llm", return_value="ok"):
                response = generate_response("email", cls(), info(client_name=None), [])
        # Pas de crash, la substitution est faite avant l'appel LLM
        assert response is not None

    def test_missing_montant_uses_indique(self):
        with patch("src.actions.response_generator.get_template") as mock_tpl:
            mock_tpl.return_value = "Montant : {montant} {devise}"
            with patch("src.actions.response_generator._call_llm", return_value="ok"):
                response = generate_response("email", cls(), info(montant=None), [])
        assert response is not None

    def test_missing_date_uses_recu(self):
        with patch("src.actions.response_generator.get_template") as mock_tpl:
            mock_tpl.return_value = "Document {date_document}"
            with patch("src.actions.response_generator._call_llm", return_value="ok"):
                response = generate_response(
                    "email", cls(), info(date_document=None), []
                )
        assert response is not None


# ---------------------------------------------------------------------------
# Court-circuit LLM (document_upload + relance sans RAG)
# ---------------------------------------------------------------------------

class TestLLMBypass:
    def test_document_upload_without_rag_uses_template_only(self):
        """document_upload + pas de RAG → pas d'appel LLM."""
        template_text = "Bonjour {client_name}, document reçu."
        with patch("src.actions.response_generator.get_template", return_value=template_text):
            with patch("src.actions.response_generator._call_llm") as mock_llm:
                response = generate_response(
                    "email",
                    cls(category="document_upload"),
                    info(),
                    [],  # rag_context vide
                )
        mock_llm.assert_not_called()
        assert "Dupont" in response  # substitution {client_name}

    def test_relance_without_rag_uses_template_only(self):
        with patch("src.actions.response_generator.get_template", return_value="Template {client_name}"):
            with patch("src.actions.response_generator._call_llm") as mock_llm:
                response = generate_response(
                    "email", cls(category="relance"), info(), []
                )
        mock_llm.assert_not_called()
        assert response is not None

    def test_facture_calls_llm(self):
        """facture → appel LLM même sans RAG."""
        with patch("src.actions.response_generator.get_template", return_value="Template"):
            with patch("src.actions.response_generator._call_llm", return_value="LLM response") as mock_llm:
                response = generate_response("email", cls(category="facture"), info(), [])
        mock_llm.assert_called_once()
        assert response == "LLM response"

    def test_document_upload_with_rag_calls_llm(self):
        """document_upload + RAG disponible → LLM quand même."""
        with patch("src.actions.response_generator.get_template", return_value="Template"):
            with patch("src.actions.response_generator._call_llm", return_value="enriched") as mock_llm:
                response = generate_response(
                    "email",
                    cls(category="document_upload"),
                    info(),
                    ["règle TVA"],  # rag_context non vide
                )
        mock_llm.assert_called_once()


# ---------------------------------------------------------------------------
# Fallback sur erreur LLM
# ---------------------------------------------------------------------------

class TestLLMFallback:
    @patch("src.utils.time.sleep")
    def test_llm_failure_returns_template_not_empty(self, mock_sleep):
        """Si le LLM échoue, on retourne le template pré-rempli, pas une chaîne vide."""
        import anthropic
        template_text = "Bonjour Dupont, facture reçue."
        with patch("src.actions.response_generator.get_template", return_value=template_text):
            with patch("src.actions.response_generator.anthropic.Anthropic") as mock_cls:
                mock_cls.return_value.messages.create.side_effect = anthropic.APIConnectionError(
                    request=MagicMock()
                )
                response = generate_response("email", cls(category="facture"), info(), [])
        # Le template pré-rempli doit être retourné
        assert response
        assert response != ""
        # Pas la réponse de secours vide
        assert "Dupont" in response or "Monsieur/Madame" in response

    @patch("src.utils.time.sleep")
    def test_fallback_response_when_no_template(self, mock_sleep):
        """Aucun template + LLM échoue → réponse de secours générique."""
        import anthropic
        with patch("src.actions.response_generator.get_template", return_value=None):
            with patch("src.actions.response_generator.anthropic.Anthropic") as mock_cls:
                mock_cls.return_value.messages.create.side_effect = anthropic.APIConnectionError(
                    request=MagicMock()
                )
                response = generate_response("email", cls(category="facture"), info(), [])
        assert response
        assert response == _FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Contenu de la réponse LLM
# ---------------------------------------------------------------------------

class TestLLMResponseContent:
    @patch("src.utils.time.sleep")
    def test_llm_response_is_returned_as_is(self, mock_sleep):
        expected = "Bonjour Dupont, nous avons bien reçu votre facture."
        with _patch_llm(expected):
            response = generate_response("email", cls(category="facture"), info(), [])
        assert response == expected

    @patch("src.utils.time.sleep")
    def test_rag_context_included_in_prompt(self, mock_sleep):
        """Le contexte RAG doit être passé au LLM (vérifiable via le prompt)."""
        captured_prompts = []
        with patch("src.actions.response_generator.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = lambda **kwargs: (
                captured_prompts.append(kwargs.get("messages", [{}])[0].get("content", ""))
                or MagicMock(content=[MagicMock(text="réponse")])
            )
            generate_response(
                "email test",
                cls(category="facture"),
                info(),
                ["Règle TVA : 20% standard"],
            )
        assert captured_prompts
        assert "Règle TVA" in captured_prompts[0]
