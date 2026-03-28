"""
Fixtures partagées entre tous les modules de test.
"""
import json
import os
from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

# Clé API bidon pour les tests — évite KeyError sans appels réels
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.70")
os.environ.setdefault("HIGH_AMOUNT_ESCALATION", "10000")


# ---------------------------------------------------------------------------
# Modèles de base
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_email():
    from src.models import Attachment, Email
    return Email(
        id="email_test_001",
        from_address="client@example.fr",
        subject="Facture n°2024-042",
        body="Bonjour, veuillez trouver ci-joint notre facture de 2 450 EUR TTC.",
        received_at=datetime(2024, 3, 28, 10, 0, 0, tzinfo=timezone.utc),
        attachments=[
            Attachment(
                filename="facture_2024_042.pdf",
                content_type="application/pdf",
                text="FACTURE N° 2024-042\nTotal TTC : 2 450,00 EUR\nTVA 20%",
            )
        ],
    )


@pytest.fixture
def email_factory():
    """Factory pour créer des emails custom dans les tests."""
    from src.models import Email

    def _make(
        id="email_test",
        from_address="test@example.fr",
        subject="Test",
        body="Corps du message",
        received_at=None,
        attachments=None,
    ):
        return Email(
            id=id,
            from_address=from_address,
            subject=subject,
            body=body,
            received_at=received_at or datetime(2024, 3, 28, 10, 0, 0, tzinfo=timezone.utc),
            attachments=attachments or [],
        )

    return _make


@pytest.fixture
def classification_ok():
    from src.models import Classification
    return Classification(category="facture", confidence=0.92, reasoning="Test")


@pytest.fixture
def classification_low_conf():
    from src.models import Classification
    return Classification(category="autre", confidence=0.4, reasoning="Incertain")


@pytest.fixture
def classification_zero():
    from src.models import Classification
    return Classification(category="autre", confidence=0.0, reasoning="Erreur LLM")


@pytest.fixture
def extracted_info_ok():
    from src.models import ExtractedInfo
    return ExtractedInfo(
        montant=2450.0,
        devise="EUR",
        date_document=date(2024, 3, 28),
        client_name="Dupont SAS",
        urgency="low",
        action_required="Enregistrer la facture",
        tva_rate=20.0,
    )


@pytest.fixture
def extracted_info_minimal():
    from src.models import ExtractedInfo
    return ExtractedInfo()


@pytest.fixture
def validation_ok():
    from src.models import ValidationResult
    return ValidationResult(is_valid=True)


@pytest.fixture
def validation_error():
    from src.models import ValidationResult
    return ValidationResult(is_valid=False, errors=["TVA invalide : 15.0%"])


# ---------------------------------------------------------------------------
# Mock Anthropic client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_response():
    """Retourne une factory pour créer des réponses LLM mockées."""
    def _make(text: str):
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        return msg
    return _make


@pytest.fixture
def mock_anthropic_client(mock_llm_response):
    """Client Anthropic mocké prêt à l'emploi."""
    client = MagicMock()
    client.messages.create.return_value = mock_llm_response(
        '{"category": "facture", "confidence": 0.9, "reasoning": "test"}'
    )
    return client


# ---------------------------------------------------------------------------
# Fichiers temporaires
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_emails_file(tmp_path):
    """Crée un fichier JSON d'emails valide dans un répertoire temporaire."""
    data = [
        {
            "id": "email_001",
            "from": "client@example.fr",
            "subject": "Facture",
            "body": "Bonjour, voici notre facture.",
            "received_at": "2024-03-28T10:00:00",
            "attachments": [],
        },
        {
            "id": "email_002",
            "from": "autre@example.fr",
            "subject": "Devis",
            "body": "Demande de devis.",
            "received_at": "2024-03-28T11:00:00",
            "attachments": [
                {"filename": "doc.pdf", "content_type": "application/pdf", "text": "contenu"},
            ],
        },
    ]
    f = tmp_path / "emails.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    return f


@pytest.fixture
def knowledge_base_dir(tmp_path):
    """Base de connaissances minimale pour les tests d'intégration."""
    kb = tmp_path / "knowledge_base"
    kb.mkdir()

    (kb / "accounting_rules.json").write_text(
        json.dumps([
            {
                "id": "tva_standard",
                "title": "TVA standard",
                "content": "Le taux de TVA standard est 20%.",
                "source": "accounting_rules",
            }
        ]),
        encoding="utf-8",
    )
    (kb / "response_templates.json").write_text(
        json.dumps([
            {
                "id": "tpl_facture",
                "category": "facture",
                "template": "Bonjour {client_name}, facture {montant} {devise} reçue.",
            },
            {
                "id": "tpl_escalate",
                "category": "escalate",
                "template": "Bonjour {client_name}, nous revenons vers vous.",
            },
        ]),
        encoding="utf-8",
    )
    (kb / "faq.json").write_text(json.dumps([]), encoding="utf-8")
    return str(kb)
