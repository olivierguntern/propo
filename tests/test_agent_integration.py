"""
Tests d'intégration du pipeline complet (src/agent.py).

Tous les appels LLM et RAG sont mockés.
On teste le pipeline de bout en bout : Email → AgentResult.

Couvre :
- Pipeline nominal : catégorie keyword → réponse générée
- Classification LLM → confiance faible → escalade
- Validation échoue → escalade
- Montant élevé facture → escalade
- Urgence haute → escalade
- document_upload + pièce jointe → action classify_document
- confidence=0.0 (erreur LLM) → court-circuit + escalade immédiate
- Exception inattendue → filet de sécurité (AgentResult escalade)
- Métriques : total, escalade, catégories
- Métriques thread-safe : concurrence simple
"""
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.models import Attachment, Classification, Email, ExtractedInfo, ValidationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent(knowledge_base_dir):
    """Agent avec base de connaissances temporaire, RAG désactivé."""
    with patch("src.agent.VectorStore") as mock_vs_cls:
        mock_vs = MagicMock()
        mock_vs.query.return_value = []
        mock_vs_cls.return_value = mock_vs
        from src.agent import AccountingAgent
        a = AccountingAgent(knowledge_base_dir=knowledge_base_dir)
    return a


def _make_email(
    category_hint="facture",
    urgency="low",
    has_attachment=False,
    email_id="test_001",
):
    """Crée un Email selon le scénario voulu."""
    body_map = {
        "facture": "Veuillez trouver ci-joint notre facture n°001.",
        "devis": "Nous souhaitons obtenir un devis pour vos services.",
        "question_comptable": "J'ai une question sur la TVA applicable.",
        "relance": "Suite à notre relance, nous attendons une réponse.",
        "document_upload": "Veuillez trouver ci-joint les documents.",
        "neutre": "Bonjour, cordialement.",
    }
    atts = (
        [Attachment(filename="facture.pdf", content_type="application/pdf", text="TVA 20%")]
        if has_attachment
        else []
    )
    return Email(
        id=email_id,
        from_address="client@example.fr",
        subject=body_map.get(category_hint, "Test"),
        body=body_map.get(category_hint, "Bonjour"),
        received_at=datetime(2024, 3, 28, 10, 0, tzinfo=timezone.utc),
        attachments=atts,
    )


# ---------------------------------------------------------------------------
# Pipeline nominal (règles mots-clés → pas de LLM classif)
# ---------------------------------------------------------------------------

class TestNominalPipeline:
    @patch("src.agent.generate_response", return_value="Réponse générée automatiquement")
    @patch("src.agent.extract_info")
    def test_facture_pipeline_success(self, mock_extract, mock_generate, agent):
        mock_extract.return_value = ExtractedInfo(
            montant=500.0, tva_rate=20.0, client_name="Dupont", urgency="low"
        )
        email = _make_email("facture")
        result = agent.process(email)

        assert result.email_id == "test_001"
        assert result.classification.category == "facture"
        assert result.action.action_type == "respond"
        assert result.action.response == "Réponse générée automatiquement"
        assert result.validation.is_valid

    @patch("src.agent.generate_response", return_value="Réponse devis")
    @patch("src.agent.extract_info")
    def test_devis_pipeline_success(self, mock_extract, mock_generate, agent):
        mock_extract.return_value = ExtractedInfo(urgency="low")
        result = agent.process(_make_email("devis"))
        assert result.classification.category == "devis"
        assert result.action.action_type == "respond"

    @patch("src.agent.generate_response", return_value="Document reçu")
    @patch("src.agent.extract_info")
    def test_document_upload_with_attachment(self, mock_extract, mock_generate, agent):
        mock_extract.return_value = ExtractedInfo(urgency="low")
        email = _make_email("document_upload", has_attachment=True)
        result = agent.process(email)
        assert result.action.action_type == "classify_document"
        assert result.action.document_category is not None
        assert result.action.response == "Document reçu"


# ---------------------------------------------------------------------------
# Escalades
# ---------------------------------------------------------------------------

class TestEscalations:
    @patch("src.agent.extract_info")
    def test_low_confidence_escalates(self, mock_extract, agent):
        mock_extract.return_value = ExtractedInfo(urgency="low")
        with patch("src.agent.classify_email") as mock_cls:
            mock_cls.return_value = Classification(
                category="autre", confidence=0.4, reasoning="Incertain"
            )
            result = agent.process(_make_email("neutre"))
        assert result.action.action_type == "escalate"
        assert result.action.escalation_reason is not None

    @patch("src.agent.extract_info")
    def test_validation_error_escalates(self, mock_extract, agent):
        mock_extract.return_value = ExtractedInfo(
            montant=1000.0, tva_rate=15.0, urgency="low"  # TVA invalide
        )
        email = _make_email("facture")
        result = agent.process(email)
        assert result.action.action_type == "escalate"
        assert "TVA" in result.action.escalation_reason

    @patch("src.agent.extract_info")
    def test_high_amount_facture_escalates(self, mock_extract, agent):
        mock_extract.return_value = ExtractedInfo(
            montant=50_000.0, tva_rate=20.0, urgency="low"
        )
        result = agent.process(_make_email("facture"))
        assert result.action.action_type == "escalate"

    @patch("src.agent.extract_info")
    def test_urgent_email_escalates(self, mock_extract, agent):
        mock_extract.return_value = ExtractedInfo(urgency="high", action_required="URGENT")
        result = agent.process(_make_email("facture"))
        assert result.action.action_type == "escalate"
        assert "urgent" in result.action.escalation_reason.lower()

    def test_zero_confidence_short_circuits_extraction(self, agent):
        """confidence=0.0 → extract_info ne doit pas être appelé."""
        with patch("src.agent.classify_email") as mock_cls, \
             patch("src.agent.extract_info") as mock_ext:
            mock_cls.return_value = Classification(
                category="autre", confidence=0.0, reasoning="Erreur LLM"
            )
            result = agent.process(_make_email("neutre"))
        mock_ext.assert_not_called()
        assert result.action.action_type == "escalate"


# ---------------------------------------------------------------------------
# Filet de sécurité (exception inattendue)
# ---------------------------------------------------------------------------

class TestSafetyNet:
    def test_unexpected_exception_returns_escalation_result(self, agent):
        with patch("src.agent.classify_email", side_effect=RuntimeError("crash inattendu")):
            result = agent.process(_make_email("neutre", email_id="crash_001"))
        # Le pipeline ne doit pas propager l'exception
        assert result is not None
        assert result.email_id == "crash_001"
        assert result.action.action_type == "escalate"
        assert "RuntimeError" in result.action.escalation_reason

    def test_safety_net_result_is_valid_agentresult(self, agent):
        with patch("src.agent.classify_email", side_effect=ValueError("test error")):
            result = agent.process(_make_email("neutre"))
        # Tous les champs requis présents
        assert result.classification is not None
        assert result.extracted_info is not None
        assert result.validation is not None
        assert isinstance(result.processing_time_ms, int)

    def test_safety_net_logs_error(self, agent):
        with patch("src.agent.classify_email", side_effect=Exception("boom")):
            with patch("src.agent.log_error") as mock_log_error:
                agent.process(_make_email("neutre", email_id="err_001"))
        mock_log_error.assert_called_once()
        args = mock_log_error.call_args[0]
        assert args[0] == "err_001"


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_processed_count_increments(self, agent):
        from src.monitoring.logger import metrics
        before = metrics.total_processed
        with patch("src.agent.extract_info", return_value=ExtractedInfo()), \
             patch("src.agent.generate_response", return_value="ok"):
            agent.process(_make_email("facture"))
        assert metrics.total_processed == before + 1

    def test_escalated_count_increments_on_escalation(self, agent):
        from src.monitoring.logger import metrics
        before_escalated = metrics.escalated
        with patch("src.agent.classify_email") as mock_cls:
            mock_cls.return_value = Classification(
                category="autre", confidence=0.0, reasoning="erreur"
            )
            agent.process(_make_email("neutre"))
        assert metrics.escalated == before_escalated + 1

    def test_category_counter_updated(self, agent):
        from src.monitoring.logger import metrics
        with patch("src.agent.extract_info", return_value=ExtractedInfo()), \
             patch("src.agent.generate_response", return_value="ok"):
            agent.process(_make_email("devis"))
        assert metrics.by_category.get("devis", 0) >= 1


# ---------------------------------------------------------------------------
# Thread-safety des métriques
# ---------------------------------------------------------------------------

class TestMetricsThreadSafety:
    def test_concurrent_processing_no_race_condition(self, agent):
        """10 threads traitent en parallèle → le compteur doit être exact."""
        from src.monitoring.logger import metrics
        before = metrics.total_processed
        errors: list[Exception] = []

        def process_email(idx: int):
            try:
                with patch("src.agent.classify_email") as mock_cls, \
                     patch("src.agent.extract_info", return_value=ExtractedInfo()), \
                     patch("src.agent.generate_response", return_value="ok"):
                    mock_cls.return_value = Classification(
                        category="devis", confidence=0.9, reasoning="test"
                    )
                    agent.process(_make_email("devis", email_id=f"thread_{idx}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=process_email, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Erreurs dans les threads : {errors}"
        assert metrics.total_processed >= before + 10
