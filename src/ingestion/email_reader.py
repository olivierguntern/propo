"""
Ingestion des emails.
En prod : connexion IMAP/API (Gmail, Outlook).
Ici : lecture depuis un fichier JSON mock.
"""
import json
from datetime import datetime
from pathlib import Path

from src.models import Attachment, Email


def load_mock_emails(path: str | Path = "data/mock_emails/emails.json") -> list[Email]:
    """Charge des emails depuis le fichier mock JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    emails = []
    for raw in data:
        attachments = [
            Attachment(
                filename=a["filename"],
                content_type=a["content_type"],
                text=a.get("text"),
            )
            for a in raw.get("attachments", [])
        ]
        emails.append(
            Email(
                id=raw["id"],
                from_address=raw["from"],
                subject=raw["subject"],
                body=raw["body"],
                received_at=datetime.fromisoformat(raw["received_at"]),
                attachments=attachments,
            )
        )
    return emails


def email_to_text(email: Email) -> str:
    """Concatène sujet + corps + texte des pièces jointes pour analyse LLM."""
    parts = [
        f"De : {email.from_address}",
        f"Sujet : {email.subject}",
        f"Corps :\n{email.body}",
    ]
    for att in email.attachments:
        if att.text:
            parts.append(f"\n--- Pièce jointe : {att.filename} ---\n{att.text}")
    return "\n\n".join(parts)
