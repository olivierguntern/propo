"""
Script de démonstration — traite les 5 emails mock et affiche les résultats.

Usage :
    cp .env.example .env
    # Renseigner ANTHROPIC_API_KEY dans .env
    python demo.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    print("⚠  ANTHROPIC_API_KEY manquant dans .env — arrêt.")
    sys.exit(1)

from src.agent import AccountingAgent
from src.ingestion.email_reader import load_mock_emails
from src.monitoring.logger import metrics


def main():
    print("=" * 60)
    print("  Agent comptable Dougs — Démonstration")
    print("=" * 60)

    agent = AccountingAgent()
    emails = load_mock_emails()

    for email in emails:
        print(f"\n📧  [{email.id}] {email.subject}")
        print(f"    De : {email.from_address}")

        result = agent.process(email)

        print(f"    → Catégorie  : {result.classification.category} "
              f"(confiance : {result.classification.confidence:.0%})")
        print(f"    → Action     : {result.action.action_type}")

        if result.action.action_type == "escalate":
            print(f"    ⚠  Escalade  : {result.action.escalation_reason}")
        elif result.action.response:
            preview = result.action.response[:120].replace("\n", " ")
            print(f"    ✉  Réponse   : {preview}...")

        if not result.validation.is_valid:
            for err in result.validation.errors:
                print(f"    ✗ Erreur     : {err}")
        for warn in result.validation.warnings:
            print(f"    ⚡ Warning    : {warn}")

        print(f"    ⏱  Traité en : {result.processing_time_ms} ms")

    print("\n" + "=" * 60)
    print("  Métriques de session")
    print("=" * 60)
    summary = metrics.summary()
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
