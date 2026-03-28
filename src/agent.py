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
- try/except global : un email ne doit jamais bloquer le pipeline
"""
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

from src.actions.document_classifier import classify_document
from src.actions.escalation import build_escalation_action, should_escalate
from src.actions.response_generator import generate_response
from src.analysis.classifier import classify_email
from src.analysis.extractor import extract_info
from src.analysis.validator import validate
from src.ingestion.email_reader import email_to_text
from src.models import AgentAction, AgentResult, Classification, Email, ExtractedInfo, ValidationResult
from src.monitoring.logger import log_error, log_result, metrics
from src.rag.knowledge_base import load_knowledge
from src.rag.vector_store import VectorStore

_logger = logging.getLogger("agent")


class AccountingAgent:
    def __init__(self, knowledge_base_dir: str = "data/knowledge_base"):
        self._vector_store = VectorStore()
        docs = load_knowledge(knowledge_base_dir)
        self._vector_store.index_documents(docs)

    def process(self, email: Email) -> AgentResult:
        """
        Traite un email de bout en bout.
        Garantit toujours un AgentResult : en cas d'erreur inattendue,
        retourne une escalade d'urgence plutôt que de lever une exception.
        """
        start_ms = int(time.time() * 1000)
        full_text = email_to_text(email)

        try:
            return self._process_inner(email, full_text, start_ms)
        except Exception as exc:
            # Filet de sécurité : aucun email ne doit être silencieusement perdu
            elapsed_ms = int(time.time() * 1000) - start_ms
            _logger.error("Unhandled error processing email %s: %s", email.id, exc, exc_info=True)
            log_error(email.id, exc)

            result = AgentResult(
                email_id=email.id,
                classification=Classification(
                    category="autre",
                    confidence=0.0,
                    reasoning=f"Erreur pipeline: {type(exc).__name__}",
                ),
                extracted_info=ExtractedInfo(action_required="Erreur pipeline — révision manuelle urgente"),
                validation=ValidationResult(is_valid=False, errors=[str(exc)]),
                action=AgentAction(
                    action_type="escalate",
                    escalation_reason=f"Erreur inattendue dans le pipeline : {type(exc).__name__}: {exc}",
                ),
                processing_time_ms=elapsed_ms,
                rag_context_used=False,
            )
            log_result(result)
            return result

    def _process_inner(self, email: Email, full_text: str, start_ms: int) -> AgentResult:
        # 1. Classification
        classification = classify_email(full_text)

        # Optimisation : si on sait déjà qu'on va escalader (confidence=0),
        # on court-circuite l'extraction LLM pour éviter un appel inutile.
        if classification.confidence == 0.0:
            elapsed_ms = int(time.time() * 1000) - start_ms
            action = build_escalation_action(classification.reasoning)
            result = AgentResult(
                email_id=email.id,
                classification=classification,
                extracted_info=ExtractedInfo(action_required="Classification échouée"),
                validation=ValidationResult(is_valid=False, errors=["Classification LLM échouée"]),
                action=action,
                processing_time_ms=elapsed_ms,
                rag_context_used=False,
            )
            log_result(result)
            return result

        # 2. Extraction des informations clés (LLM)
        extracted_info = extract_info(full_text)

        # 3. Validation métier (sans LLM)
        validation = validate(extracted_info)

        # 4. Décision d'escalade (sans LLM)
        escalate, reason = should_escalate(classification, validation, extracted_info)
        rag_used = False

        if escalate:
            action = build_escalation_action(reason)
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
                action = AgentAction(action_type="respond", response=response)

        elapsed_ms = int(time.time() * 1000) - start_ms
        result = AgentResult(
            email_id=email.id,
            classification=classification,
            extracted_info=extracted_info,
            validation=validation,
            action=action,
            processing_time_ms=elapsed_ms,
            rag_context_used=rag_used,
        )
        log_result(result)
        return result
