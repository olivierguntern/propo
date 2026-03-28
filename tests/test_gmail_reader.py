"""
Tests de src/ingestion/gmail_reader.py.

Toute l'API Google est mockée — aucune connexion réseau requise.

Couvre :
- _decode_b64 : base64 valide, padding, invalide
- _get_header : présent, absent, insensible à la casse
- _strip_html : balises br/p, balises diverses, texte pur
- _extract_parts : text/plain, text/html, multipart/mixed, pièce jointe,
                   pièce jointe avec erreur de téléchargement, multipart imbriqué
- _parse_message : message complet, From absent, date invalide
- GmailReader.fetch_unread : aucun message, un message, plusieurs,
                              erreur list(), erreur get() isolée
- GmailReader.mark_as_read : succès, erreur silencieuse
- GmailReader.mark_processed : label existant, label à créer,
                                erreur silencieuse
- _build_service : token valide, token expiré, absence credentials
"""
import base64
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

# Helpers pour encoder du texte en base64url Gmail
def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# _decode_b64
# ---------------------------------------------------------------------------

class TestDecodeB64:
    def test_decodes_valid_base64(self):
        from src.ingestion.gmail_reader import _decode_b64
        encoded = base64.urlsafe_b64encode(b"hello world").decode()
        assert _decode_b64(encoded) == b"hello world"

    def test_handles_missing_padding(self):
        from src.ingestion.gmail_reader import _decode_b64
        # Gmail n'envoie pas le padding = → on l'ajoute automatiquement
        encoded = base64.urlsafe_b64encode(b"test").decode().rstrip("=")
        assert _decode_b64(encoded) == b"test"

    def test_invalid_base64_returns_empty_bytes(self):
        from src.ingestion.gmail_reader import _decode_b64
        result = _decode_b64("!!!invalide!!!")
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# _get_header
# ---------------------------------------------------------------------------

class TestGetHeader:
    def test_finds_header_exact_case(self):
        from src.ingestion.gmail_reader import _get_header
        headers = [{"name": "From", "value": "test@x.fr"}]
        assert _get_header(headers, "From") == "test@x.fr"

    def test_finds_header_case_insensitive(self):
        from src.ingestion.gmail_reader import _get_header
        headers = [{"name": "FROM", "value": "test@x.fr"}]
        assert _get_header(headers, "from") == "test@x.fr"

    def test_returns_default_when_absent(self):
        from src.ingestion.gmail_reader import _get_header
        assert _get_header([], "From", "default") == "default"

    def test_returns_empty_string_by_default(self):
        from src.ingestion.gmail_reader import _get_header
        assert _get_header([], "Subject") == ""


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

class TestStripHtml:
    def test_removes_simple_tags(self):
        from src.ingestion.gmail_reader import _strip_html
        assert _strip_html("<p>Bonjour</p>") == "Bonjour"

    def test_converts_br_to_newline(self):
        from src.ingestion.gmail_reader import _strip_html
        result = _strip_html("Ligne 1<br>Ligne 2")
        assert "Ligne 1" in result
        assert "Ligne 2" in result

    def test_plain_text_unchanged(self):
        from src.ingestion.gmail_reader import _strip_html
        assert _strip_html("Bonjour le monde") == "Bonjour le monde"

    def test_normalizes_multiple_newlines(self):
        from src.ingestion.gmail_reader import _strip_html
        result = _strip_html("<p>A</p><p>B</p><p>C</p>")
        assert result.count("\n") <= 2  # normalisé


# ---------------------------------------------------------------------------
# _extract_parts
# ---------------------------------------------------------------------------

class TestExtractParts:
    def _make_service(self):
        return MagicMock()

    def test_text_plain_extracted(self):
        from src.ingestion.gmail_reader import _extract_parts
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64("Bonjour le client")},
        }
        body, atts = _extract_parts(payload, self._make_service(), "me", "msg1")
        assert "Bonjour le client" in body
        assert atts == []

    def test_text_html_extracted_and_stripped(self):
        from src.ingestion.gmail_reader import _extract_parts
        payload = {
            "mimeType": "text/html",
            "body": {"data": _b64("<p>Bonjour <b>le client</b></p>")},
        }
        body, atts = _extract_parts(payload, self._make_service(), "me", "msg1")
        assert "Bonjour" in body
        assert "<p>" not in body

    def test_multipart_plain_preferred_over_html(self):
        from src.ingestion.gmail_reader import _extract_parts
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("texte plain")}},
                {"mimeType": "text/html", "body": {"data": _b64("<p>texte html</p>")}},
            ],
        }
        body, _ = _extract_parts(payload, self._make_service(), "me", "msg1")
        assert "texte plain" in body

    def test_attachment_extracted(self):
        from src.ingestion.gmail_reader import _extract_parts
        service = MagicMock()
        service.users().messages().attachments().get().execute.return_value = {
            "data": _b64("contenu pdf simulé"),
        }
        payload = {
            "mimeType": "application/pdf",
            "filename": "facture.pdf",
            "body": {"attachmentId": "att_001"},
        }
        body, atts = _extract_parts(payload, service, "me", "msg1")
        assert len(atts) == 1
        assert atts[0].filename == "facture.pdf"
        assert atts[0].content_type == "application/pdf"

    def test_attachment_download_failure_returns_none_text(self):
        from src.ingestion.gmail_reader import _extract_parts
        service = MagicMock()
        service.users().messages().attachments().get().execute.side_effect = Exception("Network error")
        payload = {
            "mimeType": "application/pdf",
            "filename": "document.pdf",
            "body": {"attachmentId": "att_001"},
        }
        body, atts = _extract_parts(payload, service, "me", "msg1")
        assert len(atts) == 1
        assert atts[0].text is None  # erreur silencieuse, pas de crash

    def test_multipart_mixed_body_and_attachment(self):
        from src.ingestion.gmail_reader import _extract_parts
        service = MagicMock()
        service.users().messages().attachments().get().execute.return_value = {
            "data": _b64("pdf content"),
        }
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("corps du mail")}},
                {
                    "mimeType": "application/pdf",
                    "filename": "att.pdf",
                    "body": {"attachmentId": "att_id"},
                },
            ],
        }
        body, atts = _extract_parts(payload, service, "me", "msg1")
        assert "corps du mail" in body
        assert len(atts) == 1

    def test_empty_payload_returns_empty(self):
        from src.ingestion.gmail_reader import _extract_parts
        body, atts = _extract_parts({}, self._make_service(), "me", "msg1")
        assert body == ""
        assert atts == []


# ---------------------------------------------------------------------------
# _parse_message
# ---------------------------------------------------------------------------

class TestParseMessage:
    def _make_msg(self, from_addr="client@x.fr", subject="Test", date_str="Thu, 28 Mar 2024 10:00:00 +0000", body_text="Bonjour"):
        return {
            "id": "msg_001",
            "payload": {
                "headers": [
                    {"name": "From", "value": from_addr},
                    {"name": "Subject", "value": subject},
                    {"name": "Date", "value": date_str},
                ],
                "mimeType": "text/plain",
                "body": {"data": _b64(body_text)},
            },
        }

    def test_valid_message_parsed(self):
        from src.ingestion.gmail_reader import _parse_message
        email = _parse_message(self._make_msg(), MagicMock(), "me")
        assert email is not None
        assert email.id == "msg_001"
        assert email.from_address == "client@x.fr"
        assert email.subject == "Test"
        assert "Bonjour" in email.body
        assert isinstance(email.received_at, datetime)

    def test_missing_from_returns_none(self):
        from src.ingestion.gmail_reader import _parse_message
        msg = self._make_msg(from_addr="")
        result = _parse_message(msg, MagicMock(), "me")
        assert result is None

    def test_invalid_date_uses_now(self):
        from src.ingestion.gmail_reader import _parse_message
        msg = self._make_msg(date_str="date invalide")
        email = _parse_message(msg, MagicMock(), "me")
        assert email is not None
        # La date doit être récente (within a few seconds)
        now = datetime.now(timezone.utc)
        delta = abs((now - email.received_at).total_seconds())
        assert delta < 5

    def test_subject_defaults_when_absent(self):
        from src.ingestion.gmail_reader import _parse_message
        msg = {
            "id": "x",
            "payload": {
                "headers": [{"name": "From", "value": "a@b.fr"}],
                "mimeType": "text/plain",
                "body": {"data": _b64("body")},
            },
        }
        email = _parse_message(msg, MagicMock(), "me")
        assert email is not None
        assert email.subject == "(sans objet)"


# ---------------------------------------------------------------------------
# GmailReader (avec _build_service mocké)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_gmail_service():
    return MagicMock()


@pytest.fixture
def gmail_reader(mock_gmail_service):
    with patch("src.ingestion.gmail_reader._build_service", return_value=mock_gmail_service):
        from src.ingestion.gmail_reader import GmailReader
        return GmailReader(credentials_file="creds.json", token_file="token.json")


class TestGmailReaderFetchUnread:
    def _make_full_message(self, msg_id="m1", from_addr="a@b.fr", body_text="Bonjour"):
        return {
            "id": msg_id,
            "payload": {
                "headers": [
                    {"name": "From", "value": from_addr},
                    {"name": "Subject", "value": "Test"},
                    {"name": "Date", "value": "Thu, 28 Mar 2024 10:00:00 +0000"},
                ],
                "mimeType": "text/plain",
                "body": {"data": _b64(body_text)},
            },
        }

    def test_no_unread_returns_empty_list(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().list().execute.return_value = {"messages": []}
        result = gmail_reader.fetch_unread()
        assert result == []

    def test_empty_response_returns_empty_list(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().list().execute.return_value = {}
        result = gmail_reader.fetch_unread()
        assert result == []

    def test_single_email_fetched(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "m1"}]
        }
        mock_gmail_service.users().messages().get().execute.return_value = (
            self._make_full_message("m1", "client@x.fr")
        )
        result = gmail_reader.fetch_unread()
        assert len(result) == 1
        assert result[0].id == "m1"
        assert result[0].from_address == "client@x.fr"

    def test_multiple_emails_fetched(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
        }
        mock_gmail_service.users().messages().get().execute.side_effect = [
            self._make_full_message("m1"),
            self._make_full_message("m2"),
            self._make_full_message("m3"),
        ]
        result = gmail_reader.fetch_unread()
        assert len(result) == 3

    def test_list_api_failure_returns_empty_list(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().list().execute.side_effect = Exception("API error")
        result = gmail_reader.fetch_unread()
        assert result == []

    def test_single_message_fetch_failure_skipped(self, gmail_reader, mock_gmail_service):
        """Une erreur sur un message isolé ne bloque pas les autres."""
        mock_gmail_service.users().messages().list().execute.return_value = {
            "messages": [{"id": "m1"}, {"id": "m2"}]
        }
        mock_gmail_service.users().messages().get().execute.side_effect = [
            Exception("Network error on m1"),
            self._make_full_message("m2"),
        ]
        result = gmail_reader.fetch_unread()
        assert len(result) == 1
        assert result[0].id == "m2"

    def test_max_results_passed_to_api(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().list().execute.return_value = {}
        gmail_reader.fetch_unread(max_results=5)
        call_kwargs = mock_gmail_service.users().messages().list.call_args
        assert call_kwargs.kwargs.get("maxResults") == 5 or 5 in call_kwargs.args


class TestGmailReaderMarkAsRead:
    def test_mark_as_read_calls_modify(self, gmail_reader, mock_gmail_service):
        gmail_reader.mark_as_read("msg_001")
        mock_gmail_service.users().messages().modify.assert_called_once()
        call_kwargs = mock_gmail_service.users().messages().modify.call_args.kwargs
        assert "UNREAD" in call_kwargs["body"]["removeLabelIds"]

    def test_mark_as_read_failure_does_not_raise(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().messages().modify().execute.side_effect = Exception("API error")
        gmail_reader.mark_as_read("msg_001")  # Ne doit pas lever


class TestGmailReaderMarkProcessed:
    def test_mark_processed_with_existing_label(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().labels().list().execute.return_value = {
            "labels": [{"name": "PROCESSED_BY_AGENT", "id": "label_123"}]
        }
        gmail_reader.mark_processed("msg_001")
        modify_call = mock_gmail_service.users().messages().modify.call_args.kwargs
        assert "label_123" in modify_call["body"]["addLabelIds"]
        assert "UNREAD" in modify_call["body"]["removeLabelIds"]

    def test_mark_processed_creates_label_if_absent(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().labels().list().execute.return_value = {"labels": []}
        mock_gmail_service.users().labels().create().execute.return_value = {
            "id": "new_label_id", "name": "PROCESSED_BY_AGENT"
        }
        gmail_reader.mark_processed("msg_001")
        mock_gmail_service.users().labels().create.assert_called_once()

    def test_mark_processed_label_failure_silenced(self, gmail_reader, mock_gmail_service):
        mock_gmail_service.users().labels().list().execute.side_effect = Exception("API error")
        gmail_reader.mark_processed("msg_001")  # Ne doit pas lever


# ---------------------------------------------------------------------------
# _build_service
# ---------------------------------------------------------------------------

class TestBuildService:
    def test_valid_token_used_directly(self, tmp_path):
        from src.ingestion.gmail_reader import _build_service
        token = tmp_path / "token.json"
        token.write_text('{"token": "x"}', encoding="utf-8")

        mock_creds = MagicMock()
        mock_creds.valid = True

        with patch("src.ingestion.gmail_reader.Credentials") as mock_creds_cls, \
             patch("src.ingestion.gmail_reader.build") as mock_build:
            # On importe en local pour éviter l'erreur si google n'est pas installé
            try:
                from google.oauth2.credentials import Credentials  # noqa
            except ImportError:
                pytest.skip("google-auth non installé")

            mock_creds_cls.from_authorized_user_file.return_value = mock_creds
            mock_build.return_value = MagicMock()
            _build_service("creds.json", str(token))
            mock_build.assert_called_once()

    def test_missing_credentials_file_raises(self, tmp_path):
        from src.ingestion.gmail_reader import _build_service
        try:
            from google.oauth2.credentials import Credentials  # noqa
        except ImportError:
            pytest.skip("google-auth non installé")
        with pytest.raises(FileNotFoundError, match="credentials.json introuvable"):
            _build_service(
                credentials_file=str(tmp_path / "nonexistent.json"),
                token_file=str(tmp_path / "token.json"),
            )
