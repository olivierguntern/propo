"""
Ingestion des emails depuis Gmail via l'API officielle.

Setup (une seule fois) :
  1. Créer un projet GCP, activer l'API Gmail
  2. Télécharger credentials.json (OAuth 2.0 Desktop)
  3. Première exécution → navigateur OAuth → token.json sauvegardé

Scopes requis : gmail.modify (lecture + marquage lu)
"""
import base64
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ingestion.document_extractor import extract_text_from_file
from src.models import Attachment, Email

_logger = logging.getLogger("agent")

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Imports Google — optionnels (pas installés en CI sans les dépendances)
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build as _google_build
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    _logger.warning(
        "google-api-python-client non installé — GmailReader indisponible. "
        "Installer avec : pip install google-api-python-client google-auth-oauthlib"
    )


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------

def _build_service(
    credentials_file: str = "credentials.json",
    token_file: str = "token.json",
):
    """
    Construit le service Gmail avec OAuth2.
    - Si token.json existe et est valide → l'utilise directement.
    - Si expiré → le rafraîchit silencieusement.
    - Si absent → lance le flux OAuth (navigateur).
    """
    if not _GOOGLE_AVAILABLE:
        raise ImportError(
            "google-api-python-client non installé. "
            "Lancer : pip install google-api-python-client google-auth-oauthlib"
        )

    creds = None

    if Path(token_file).exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            _logger.info("Gmail: refreshing expired OAuth token")
            creds.refresh(Request())
        else:
            if not Path(credentials_file).exists():
                raise FileNotFoundError(
                    f"credentials.json introuvable : {credentials_file}\n"
                    "Téléchargez-le depuis Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        Path(token_file).write_text(creds.to_json(), encoding="utf-8")
        _logger.info("Gmail: OAuth token saved to %s", token_file)

    return _google_build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Parsing des messages
# ---------------------------------------------------------------------------

def _decode_b64(data: str) -> bytes:
    """Décode le base64url Gmail (padding automatique)."""
    try:
        return base64.urlsafe_b64decode(data + "==")
    except Exception as exc:
        _logger.warning("Base64 decode error: %s", exc)
        return b""


def _get_header(headers: list[dict], name: str, default: str = "") -> str:
    """Extrait une valeur de header par nom (insensible à la casse)."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", default)
    return default


def _extract_parts(
    payload: dict,
    service: Any,
    user_id: str,
    msg_id: str,
) -> tuple[str, list[Attachment]]:
    """
    Parcourt récursivement le payload MIME Gmail.
    Retourne (texte_du_corps, liste_pièces_jointes).

    Priorité du corps : text/plain > text/html (stripped).
    """
    body_text = ""
    attachments: list[Attachment] = []
    mime_type = payload.get("mimeType", "")

    # --- Partie texte ---
    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            body_text = _decode_b64(data).decode("utf-8", errors="replace")

    elif mime_type == "text/html":
        # Fallback HTML → on strip les balises basiquement
        data = payload.get("body", {}).get("data", "")
        if data:
            raw_html = _decode_b64(data).decode("utf-8", errors="replace")
            body_text = _strip_html(raw_html)

    # --- Partie multipart : descente récursive ---
    elif mime_type.startswith("multipart/"):
        plain_text = ""
        html_text = ""
        for part in payload.get("parts", []):
            sub_body, sub_atts = _extract_parts(part, service, user_id, msg_id)
            attachments.extend(sub_atts)
            if part.get("mimeType") == "text/plain" and not plain_text:
                plain_text = sub_body
            elif part.get("mimeType") == "text/html" and not html_text:
                html_text = sub_body
            elif sub_body and not plain_text and not html_text:
                plain_text = sub_body
        body_text = plain_text or html_text

    # --- Pièce jointe ---
    filename = payload.get("filename", "")
    attachment_id = payload.get("body", {}).get("attachmentId")

    if filename and attachment_id:
        att_text: str | None = None
        try:
            att_data = (
                service.users()
                .messages()
                .attachments()
                .get(userId=user_id, messageId=msg_id, id=attachment_id)
                .execute()
            )
            content_bytes = _decode_b64(att_data.get("data", ""))
            att_text = extract_text_from_file(filename, content_bytes) or None
        except Exception as exc:
            _logger.warning("Attachment download failed (%s): %s", filename, exc)

        attachments.append(
            Attachment(
                filename=filename,
                content_type=mime_type,
                text=att_text,
            )
        )

    return body_text, attachments


def _strip_html(html: str) -> str:
    """Strip HTML tags basique (pas de dépendance BeautifulSoup)."""
    import re
    # Remplace <br> et <p> par des sauts de ligne
    html = re.sub(r"<br\s*/?>|</p>", "\n", html, flags=re.IGNORECASE)
    # Supprime toutes les autres balises
    html = re.sub(r"<[^>]+>", "", html)
    # Normalise les espaces
    return re.sub(r"\n{3,}", "\n\n", html).strip()


def _parse_message(msg: dict, service: Any, user_id: str) -> Email | None:
    """
    Convertit un message Gmail brut (format "full") en modèle Email.
    Retourne None si le message est invalide.
    """
    payload = msg.get("payload", {})
    headers = payload.get("headers", [])

    from_address = _get_header(headers, "From")
    subject = _get_header(headers, "Subject", "(sans objet)")
    date_str = _get_header(headers, "Date")

    if not from_address:
        _logger.warning("Skipping message %s: missing From header", msg.get("id", "?"))
        return None

    # Parse de la date RFC 2822
    received_at: datetime
    try:
        from email.utils import parsedate_to_datetime
        received_at = parsedate_to_datetime(date_str)
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
    except Exception:
        _logger.warning("Could not parse date %r for message %s — using now()", date_str, msg.get("id"))
        received_at = datetime.now(timezone.utc)

    body, attachments = _extract_parts(payload, service, user_id, msg["id"])

    return Email(
        id=msg["id"],
        from_address=from_address,
        subject=subject,
        body=body,
        received_at=received_at,
        attachments=attachments,
    )


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class GmailReader:
    """
    Ingestion des emails non lus depuis Gmail.

    Utilisation :
        reader = GmailReader()
        emails = reader.fetch_unread(max_results=20)
        for email in emails:
            result = agent.process(email)
            reader.mark_processed(email.id)
    """

    def __init__(
        self,
        credentials_file: str | None = None,
        token_file: str | None = None,
        user_id: str | None = None,
    ):
        self._credentials_file = credentials_file or os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
        self._token_file = token_file or os.getenv("GMAIL_TOKEN_FILE", "token.json")
        self._user_id = user_id or os.getenv("GMAIL_USER_ID", "me")
        self._service = _build_service(self._credentials_file, self._token_file)

    def fetch_unread(
        self,
        max_results: int | None = None,
        label: str = "INBOX",
    ) -> list[Email]:
        """
        Récupère les emails non lus.
        Les erreurs par message sont logguées et skippées (le batch continue).
        """
        max_results = max_results or int(os.getenv("GMAIL_MAX_RESULTS", "50"))

        try:
            response = (
                self._service.users()
                .messages()
                .list(
                    userId=self._user_id,
                    labelIds=[label, "UNREAD"],
                    maxResults=max_results,
                )
                .execute()
            )
        except Exception as exc:
            _logger.error("Gmail list messages failed: %s", exc)
            return []

        message_refs = response.get("messages", [])
        if not message_refs:
            _logger.info("Gmail: no unread messages in %s", label)
            return []

        emails: list[Email] = []
        for ref in message_refs:
            msg_id = ref["id"]
            try:
                msg = (
                    self._service.users()
                    .messages()
                    .get(userId=self._user_id, id=msg_id, format="full")
                    .execute()
                )
                email = _parse_message(msg, self._service, self._user_id)
                if email:
                    emails.append(email)
            except Exception as exc:
                _logger.error("Failed to fetch message %s: %s", msg_id, exc)

        _logger.info("Gmail: fetched %d/%d emails successfully", len(emails), len(message_refs))
        return emails

    def mark_as_read(self, email_id: str) -> None:
        """Retire le label UNREAD d'un email."""
        try:
            self._service.users().messages().modify(
                userId=self._user_id,
                id=email_id,
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()
        except Exception as exc:
            _logger.warning("Failed to mark email %s as read: %s", email_id, exc)

    def mark_processed(
        self,
        email_id: str,
        label_name: str | None = None,
    ) -> None:
        """
        Marque un email comme traité :
        - retire UNREAD
        - ajoute le label PROCESSED_BY_AGENT (créé si inexistant)
        """
        label_name = label_name or os.getenv("GMAIL_PROCESSED_LABEL", "PROCESSED_BY_AGENT")
        label_id = self._get_or_create_label(label_name)

        body: dict[str, list[str]] = {"removeLabelIds": ["UNREAD"]}
        if label_id:
            body["addLabelIds"] = [label_id]

        try:
            self._service.users().messages().modify(
                userId=self._user_id,
                id=email_id,
                body=body,
            ).execute()
        except Exception as exc:
            _logger.warning("Failed to mark email %s as processed: %s", email_id, exc)

    def _get_or_create_label(self, name: str) -> str | None:
        """Retourne l'ID d'un label Gmail, le crée s'il n'existe pas."""
        try:
            labels = (
                self._service.users().labels().list(userId=self._user_id).execute()
            )
            for label in labels.get("labels", []):
                if label["name"].lower() == name.lower():
                    return label["id"]

            created = (
                self._service.users()
                .labels()
                .create(
                    userId=self._user_id,
                    body={"name": name, "labelListVisibility": "labelShow"},
                )
                .execute()
            )
            _logger.info("Gmail: created label '%s' (id=%s)", name, created["id"])
            return created["id"]

        except Exception as exc:
            _logger.warning("Failed to get/create label '%s': %s", name, exc)
            return None
