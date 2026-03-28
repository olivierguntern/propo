"""
Script de démonstration — traite les emails et affiche les résultats.

Modes :
  python demo.py            → emails mock (data/mock_emails/emails.json)
  python demo.py --gmail    → emails non lus depuis Gmail (OAuth2 requis)

Prérequis :
  cp .env.example .env
  # Renseigner ANTHROPIC_API_KEY dans .env
  # Pour Gmail : placer credentials.json à la racine du projet
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERREUR : ANTHROPIC_API_KEY manquant dans .env")
    sys.exit(1)

from src.agent import AccountingAgent
from src.monitoring.logger import metrics


def _print_result(result) -> None:
    cat = result.classification.category
    conf = result.classification.confidence
    action = result.action.action_type

    print(f"  -> Categorie  : {cat} (confiance : {conf:.0%})")
    print(f"  -> Action     : {action}")

    if action == "escalate":
        print(f"  [!] Escalade  : {result.action.escalation_reason}")
    elif result.action.response:
        preview = result.action.response[:120].replace("\n", " ")
        print(f"  [>] Reponse   : {preview}...")

    if result.action.document_category:
        print(f"  [D] Document  : {result.action.document_category}")

    for err in result.validation.errors:
        print(f"  [x] Erreur    : {err}")
    for warn in result.validation.warnings:
        print(f"  [!] Warning   : {warn}")

    print(f"  [t] Traite en : {result.processing_time_ms} ms")


def run_mock():
    from src.ingestion.email_reader import load_mock_emails
    print("Mode : emails mock")
    print("=" * 60)
    agent = AccountingAgent()
    emails = load_mock_emails()
    for email in emails:
        print(f"\n[{email.id}] {email.subject}")
        print(f"  De : {email.from_address}")
        result = agent.process(email)
        _print_result(result)


def run_gmail():
    print("Mode : Gmail (emails non lus)")
    print("=" * 60)
    try:
        from src.ingestion.gmail_reader import GmailReader
    except ImportError:
        print("ERREUR : google-api-python-client non installe.")
        print("  pip install google-api-python-client google-auth-oauthlib")
        sys.exit(1)

    agent = AccountingAgent()
    reader = GmailReader()
    emails = reader.fetch_unread()

    if not emails:
        print("Aucun email non lu dans la boite de reception.")
        return

    for email in emails:
        print(f"\n[{email.id}] {email.subject}")
        print(f"  De : {email.from_address}")
        result = agent.process(email)
        _print_result(result)
        reader.mark_processed(email.id)
        print(f"  [v] Marque comme traite")


def main():
    parser = argparse.ArgumentParser(description="Agent comptable Dougs")
    parser.add_argument("--gmail", action="store_true", help="Utiliser Gmail au lieu des emails mock")
    args = parser.parse_args()

    print("=" * 60)
    print("  Agent comptable Dougs")
    print("=" * 60)

    if args.gmail:
        run_gmail()
    else:
        run_mock()

    print("\n" + "=" * 60)
    print("  Metriques de session")
    print("=" * 60)
    summary = metrics.summary()
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
