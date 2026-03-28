"""
Tests de src/ingestion/email_reader.py et document_extractor.py.

Couvre :
- Chargement fichier valide (2 emails, avec et sans pièces jointes)
- FileNotFoundError avec message clair
- JSON invalide → ValueError
- Entrée malformée skippée, les autres continuent
- email_to_text : plain text, avec pièce jointe texte, plusieurs pièces jointes
- extract_text_from_pdf : module absent, PDF valide, PDF corrompu
- extract_text_from_file : dispatch par extension
"""
import json
from datetime import datetime, timezone

import pytest

from src.ingestion.email_reader import email_to_text, load_mock_emails
from src.models import Attachment, Email


# ---------------------------------------------------------------------------
# load_mock_emails
# ---------------------------------------------------------------------------

class TestLoadMockEmails:
    def test_loads_valid_file(self, mock_emails_file):
        emails = load_mock_emails(mock_emails_file)
        assert len(emails) == 2

    def test_email_fields_parsed_correctly(self, mock_emails_file):
        emails = load_mock_emails(mock_emails_file)
        e = emails[0]
        assert e.id == "email_001"
        assert e.from_address == "client@example.fr"
        assert e.subject == "Facture"
        assert e.body == "Bonjour, voici notre facture."
        assert isinstance(e.received_at, datetime)

    def test_attachment_parsed(self, mock_emails_file):
        emails = load_mock_emails(mock_emails_file)
        e = emails[1]  # email_002 a une pièce jointe
        assert len(e.attachments) == 1
        att = e.attachments[0]
        assert att.filename == "doc.pdf"
        assert att.content_type == "application/pdf"
        assert att.text == "contenu"

    def test_email_without_attachment(self, mock_emails_file):
        emails = load_mock_emails(mock_emails_file)
        assert emails[0].attachments == []

    def test_file_not_found_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_mock_emails(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{ invalide", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON invalide"):
            load_mock_emails(f)

    def test_malformed_entry_skipped_others_kept(self, tmp_path):
        data = [
            {
                "id": "ok",
                "from": "a@b.fr",
                "subject": "S",
                "body": "B",
                "received_at": "2024-01-01T10:00:00",
                "attachments": [],
            },
            {"id": "bad_no_from"},  # manque "from", "subject", "body"
        ]
        f = tmp_path / "emails.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        emails = load_mock_emails(f)
        assert len(emails) == 1
        assert emails[0].id == "ok"

    def test_all_malformed_returns_empty_list(self, tmp_path):
        data = [{"invalid": True}, {"also": "bad"}]
        f = tmp_path / "emails.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        emails = load_mock_emails(f)
        assert emails == []

    def test_empty_attachments_list(self, tmp_path):
        data = [{
            "id": "x",
            "from": "x@x.fr",
            "subject": "X",
            "body": "body",
            "received_at": "2024-01-01T10:00:00",
            "attachments": [],
        }]
        f = tmp_path / "emails.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        emails = load_mock_emails(f)
        assert emails[0].attachments == []

    def test_accepts_string_path(self, mock_emails_file):
        emails = load_mock_emails(str(mock_emails_file))
        assert len(emails) == 2


# ---------------------------------------------------------------------------
# email_to_text
# ---------------------------------------------------------------------------

class TestEmailToText:
    def _make_email(self, body="Bonjour", subject="Test", attachments=None):
        return Email(
            id="test",
            from_address="x@x.fr",
            subject=subject,
            body=body,
            received_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            attachments=attachments or [],
        )

    def test_includes_from(self):
        email = self._make_email()
        text = email_to_text(email)
        assert "x@x.fr" in text

    def test_includes_subject(self):
        email = self._make_email(subject="Ma facture")
        text = email_to_text(email)
        assert "Ma facture" in text

    def test_includes_body(self):
        email = self._make_email(body="Contenu important")
        text = email_to_text(email)
        assert "Contenu important" in text

    def test_includes_attachment_text(self):
        att = Attachment(filename="doc.pdf", content_type="application/pdf", text="Texte PDF")
        email = self._make_email(attachments=[att])
        text = email_to_text(email)
        assert "Texte PDF" in text
        assert "doc.pdf" in text

    def test_attachment_without_text_excluded(self):
        att = Attachment(filename="doc.pdf", content_type="application/pdf", text=None)
        email = self._make_email(attachments=[att])
        text = email_to_text(email)
        assert "doc.pdf" not in text

    def test_multiple_attachments_all_included(self):
        atts = [
            Attachment(filename="a.pdf", content_type="application/pdf", text="texte A"),
            Attachment(filename="b.pdf", content_type="application/pdf", text="texte B"),
        ]
        email = self._make_email(attachments=atts)
        text = email_to_text(email)
        assert "texte A" in text
        assert "texte B" in text

    def test_no_attachments(self):
        email = self._make_email()
        text = email_to_text(email)
        assert "Pièce jointe" not in text


# ---------------------------------------------------------------------------
# document_extractor
# ---------------------------------------------------------------------------

class TestDocumentExtractor:
    def test_extract_non_pdf_returns_decoded_text(self):
        from src.ingestion.document_extractor import extract_text_from_file
        content = "Contenu texte".encode("utf-8")
        result = extract_text_from_file("rapport.txt", content)
        assert result == "Contenu texte"

    def test_extract_pdf_module_absent(self):
        from src.ingestion.document_extractor import extract_text_from_pdf
        import src.ingestion.document_extractor as mod
        original = mod._PYPDF2_AVAILABLE
        mod._PYPDF2_AVAILABLE = False
        result = extract_text_from_pdf(b"fake pdf")
        mod._PYPDF2_AVAILABLE = original
        assert "non installé" in result

    def test_extract_pdf_corrupted_returns_error_string(self):
        from src.ingestion.document_extractor import extract_text_from_pdf
        import src.ingestion.document_extractor as mod
        if not mod._PYPDF2_AVAILABLE:
            pytest.skip("PyPDF2 non disponible")
        result = extract_text_from_pdf(b"ceci n'est pas un PDF")
        assert isinstance(result, str)
        # Doit retourner une chaîne d'erreur, pas lever une exception
        assert "Erreur" in result or result == ""

    def test_extract_file_dispatch_pdf(self):
        from src.ingestion.document_extractor import extract_text_from_file
        import src.ingestion.document_extractor as mod
        import unittest.mock as mock
        with mock.patch.object(mod, "extract_text_from_pdf", return_value="pdf content") as mock_pdf:
            result = extract_text_from_file("document.pdf", b"bytes")
        mock_pdf.assert_called_once()
        assert result == "pdf content"
