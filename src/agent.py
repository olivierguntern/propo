"""
Orchestrateur principal de l'agent.

Pipeline :
  Email → [Ingestion] → [Classification] → [Extraction] → [Validation]
       → [Décision escalade] → [RAG] → [Action] → [Log]

Principes de conception :
- Chaque étape est indépendante et testable séparément
- LLM uniquement là où le langage naturel est nécessaire
- Validation et règles métier = code Python pur
- Escalade automatique si confiance < seuil ou données invalides
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.actions.document_classifier import classify_document
from src.actions.escalation import build_escalation_action, should_escalate
from src.actions.response_generator import generate_response
from src.analysis.classifier import classify_email
from src.analysis.extractor import extract_info
from src.analysis.validator import validate
from src.ingestion.email_reader import email_to_text
from src.models import AgentAction, AgentResult, Email
from src.monitoring.logger import log_result, metrics
from src.rag.knowledge_base import load_knowledge
from src.rag.vector_store import VectorStore


class AccountingAgent:
    def __init__(self, knowledge_base_dir: str = "data/knowledge_base"):
        self._vector_store = VectorStore()
        # Indexation au démarrage
        docs = load_knowledge(knowledge_base_dir)
        self._vector_store.index_documents(docs)

    def process(self, email: Email) -> AgentResult:
        """Traite un email de bout en bout."""
        start_ms = int(time.time() * 1000)
        full_text = email_to_text(email)

        # 1. Classification
        classification = classify_email(full_text)

        # 2. Extraction des informations clés (LLM)
        extracted_info = extract_info(full_text)

        # 3. Validation métier (sans LLM)
        validation = validate(extracted_info)

        # 4. Décision d'escalade (sans LLM)
        escalate, reason = should_escalate(classification, validation, extracted_info)
        if escalate:
            action = build_escalation_action(reason)
            rag_used = False
        else:
            # 5. Enrichissement RAG
            rag_docs = self._vector_store.query(full_text, n_results=3)
            rag_used = bool(rag_docs)

            # 6. Action selon catégorie
            if classification.category == "document_upload" and email.attachments:
                att = email.attachments[0]
                doc_type = classify_document(att.filename, att.text or "")
                response = generate_response(full_text, classification, extracted_info, rag_docs)
                action = AgentAction(
                    action_type="classify_document",
                    document_category=doc_type,
                    response=response,
                )
            else:
                response = generate_response(full_text, classification, extracted_info, rag_docs)
                action = AgentAction(
                    action_type="respond",
                    response=response,
                )

        elapsed_ms = int(time.time() * 1000) - start_ms

        result = AgentResult(
            email_id=email.id,
            classification=classification,
            extracted_info=extracted_info,
            validation=validation,
            action=action,
            processing_time_ms=elapsed_ms,
            rag_context_used=rag_used if not escalate else False,
        )

        log_result(result)
        return result
